#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "parser.h"
#include "list.h"
#include "utils.h"
#include "option_list.h"
#include "cuda.h"
#include "tree.h"

#include "activation_layer.h"
#include "batchnorm_layer.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "detection_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "softmax_layer.h"
#include "shortcut_layer.h"
#include "region_layer.h"
#include "reorg_layer.h"
#include "route_layer.h"

//网络与各个层之间的承接变量、中间变量
typedef struct size_param
{
	int batch;
	int inputs;
	int h, w, c;
	int index;
	int time_steps;
	network net;
}size_params;

learning_rate_policy get_policy(char *s)
{
	if (strcmp(s, "random") == 0)      return RANDOM;
	if (strcmp(s, "poly") == 0)        return POLY;
	if (strcmp(s, "constant") == 0)    return CONSTANT;
	if (strcmp(s, "step") == 0)        return STEP;
	if (strcmp(s, "exp") == 0)         return EXP;
	if (strcmp(s, "sigmoid") == 0)     return SIG;
	if (strcmp(s, "steps") == 0)       return STEPS;

	fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
	return CONSTANT;
}


//通过字符,转换到枚举值

LAYER_TYPE string_to_layer_type(char * type)
{

	if (strcmp(type, "[shortcut]") == 0) return SHORTCUT;
	if (strcmp(type, "[crop]") == 0) return CROP;
	if (strcmp(type, "[cost]") == 0) return COST;
	if (strcmp(type, "[detection]") == 0) return DETECTION;
	if (strcmp(type, "[region]") == 0) return REGION;
	if (strcmp(type, "[local]") == 0) return LOCAL;
	if (strcmp(type, "[conv]") == 0
		|| strcmp(type, "[convolutional]") == 0) return CONVOLUTIONAL;
	if (strcmp(type, "[deconv]") == 0
		|| strcmp(type, "[deconvolutional]") == 0) return DECONVOLUTIONAL;
	if (strcmp(type, "[activation]") == 0) return ACTIVE;
	if (strcmp(type, "[net]") == 0
		|| strcmp(type, "[network]") == 0) return NETWORK;
	if (strcmp(type, "[crnn]") == 0) return CRNN;
	if (strcmp(type, "[gru]") == 0) return GRU;
	if (strcmp(type, "[rnn]") == 0) return RNN;
	if (strcmp(type, "[conn]") == 0
		|| strcmp(type, "[connected]") == 0) return CONNECTED;
	if (strcmp(type, "[max]") == 0
		|| strcmp(type, "[maxpool]") == 0) return MAXPOOL;
	if (strcmp(type, "[reorg]") == 0) return REORG;
	if (strcmp(type, "[avg]") == 0
		|| strcmp(type, "[avgpool]") == 0) return AVGPOOL;
	if (strcmp(type, "[dropout]") == 0) return DROPOUT;
	if (strcmp(type, "[lrn]") == 0
		|| strcmp(type, "[normalization]") == 0) return NORMALIZATION;
	if (strcmp(type, "[batchnorm]") == 0) return BATCHNORM;
	if (strcmp(type, "[soft]") == 0
		|| strcmp(type, "[softmax]") == 0) return SOFTMAX;
	if (strcmp(type, "[route]") == 0) return ROUTE;

	// 如果没有一个匹配上，说明配置文件中存在不能识别的网络层名称，
	// 返回BLANK（这时应该去检查下配置文件，看看是否有拼写错误）
	return BLANK;
}

/*
* 释放section的内存,section里有两个指针元素,type和options,都有申请内存
* options是链表,链表每一个节点的val都保存着一个键值对
* 要注意这个键值对的释放方式,其实这个键值对的key和val是指向同一块内存的不同地方,key是这块内存的起点,而val只是指向这块内存的后面位置( '=' 后面)
* 所以对于这个键值对,只能释放前面的key,具体看申请内存可以看option_list.c中的read_option()函数,就是line的内存申请
* 释放顺序:从内到外,先释放子元素,然后释放section,嵌套内存的释放,不能遗漏子元素的内存释放,否则你遗漏子元素的释放,并且还释放了最外层section的内存,那遗漏的子元素就会慢慢造成内存泄漏
*/
void free_section(section *s)
{
	//这个随便释放
	free(s->type);
	node *n = s->options->front;
	while (n)
	{
		kvp *tmp = (kvp*)n->val;
		free(tmp->key);

		//先利用n来获取下一个节点,因为下面需要释放节点n
		node *next = n->next;

		free(n);
		n = next;
	}
	//free_list(s->options);
	free(s->options);
	free(s);
}

//解析网络通用参数,其实就是与option中各个节点进行比对,如果在配置文件里有这个参数,那么就读进来,否则就用默认值
void parse_net_option(list *options, network *net)
{
	//这里读进来的batch是人为定义的一轮batch的数量,下面会把这里的batch 除以subdivisions,来作为真实的batch
	net->batch = option_find_int(options, "batch", 1);
	net->learning_rate = option_find_float(options, "learning_rate", 0.001);
	net->momentum = option_find_float(options, "momentum", 0.9);
	net->decay = option_find_float(options, "decay", 0.0001);
	int subdivs = option_find_int(options, "subdivisions", 1);
	net->time_steps = option_find_int(options, "time_steps", 1);
	net->notruth = option_find_int(options, "notruth", 0);

	//这里计算得到网络训练时真实的batch数量
	net->batch /= subdivs;
	net->batch *= net->time_steps;
	net->subdivisions = subdivs;

	//todo:
	net->adam = option_find_int(options, "adam", 0);
	if (net->adam)
	{
		net->B1 = option_find_float(options, "B1", 0.9);
		net->B2 = option_find_float(options, "B2", 0.999);
		net->eps = option_find_float(options, "eps", 0.00000001);
	}

	net->h = option_find_int(options, "height", 0);
	net->w = option_find_int(options, "width", 0);
	net->c = option_find_int(options, "channels", 0);

	// 一张输入图片的元素个数，如果网络配置文件没有指定，则默认值为net->h * net->w * net->c
	net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
	net->max_crop = option_find_int_quiet(options, "max_crop", net->w * 2);
	net->min_crop = option_find_int_quiet(options, "min_crop", net->w);
	net->center = option_find_int_quiet(options, "center", 0);

	net->angle = option_find_float_quiet(options, "angle", 0);
	net->aspect = option_find_float_quiet(options, "aspect", 1);
	net->saturation = option_find_float_quiet(options, "saturation", 1);
	net->exposure = option_find_float_quiet(options, "exposure", 1);
	net->hue = option_find_float_quiet(options, "hue", 0);

	//如果相关的输入图片信息有为0的,直接退出程序
	if (!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

	//以下注释是我从某个配置文件里复制的,用于理解接下来较为重要的参数的意义
	//learning_rate = 0.001                 //初始学习率
	//burn_in = 1000                        //当学习的batch次数还没超过burn_in的值,这一阶段从0(近乎0)开始递增,在burn_in次的时候,rate变成我们预设的learning_rate
	//max_batches = 500200                  //最多学习max_batches轮
	//policy = steps                        //调整学习率的方法,有很多方法比如CONSTANT, STEP, EXP, POLY，STEPS, SIG, RANDOM,yolo2用的是steps
	//steps = 400000, 450000                //steps的调整策略就是学习batch次数超过一定值就调整一次,然后steps这个参数后的数组就是每超过数组中的一个值就调整一次
	//scales = .1, .1                       //搭配steps参数,steps的第一次调整用到的缩放比例,就是这个数组的第一个值

	//获取调整学习率的方法枚举字符串
	char *policy_s = option_find_str(options, "policy", "constant");
	net->policy = get_policy(policy_s);
	net->burn_in = option_find_int_quiet(options, "burn_in", 0);
	net->power = option_find_float_quiet(options, "power", 4);

	//这里就是根据调整学习率的方法,来个性化地获调整学习率的特殊参数
	//关于不同的调整学习率的方法,详细可以看链接,这是我自找的,可能不是很完整:https://blog.csdn.net/u014090429/article/details/103736594
	if (net->policy == STEP)
	{
		//因为只调整依次,所以只需要一组参数,训练batc次数达到step,则用scale来缩放学习率以作调整
		net->step = option_find_int(options, "step", 1);
		net->scale = option_find_float(options, "scale", 1);
	}
	else if (net->policy == STEPS)
	{
		//STEPS必须steps和scales两个参数同时存在
		char *l = option_find(options, "steps");
		char *p = option_find(options, "scales");
		if (!l || !p) error("STEPS policy must have steps and scales in cfg file");

		int len = strlen(l);
		int n = 1;
		int i;
		//确定需要调整几次
		for (i = 0; i < len; ++i) {
			if (l[i] == ',') ++n;
		}
		int *steps = calloc(n, sizeof(int));
		float *scales = calloc(n, sizeof(float));
		for (i = 0; i < n; ++i) {
			//atoi,atof: 将字符串转换为int和float,如果中途有非数字字符的话,就会停止,这里有 ',' 隔开,所以没事
			int step = atoi(l);
			float scale = atof(p);

			//strchr函数功能解释:返回在参数 str 所指向的字符串中搜索第一次出现字符（一个无符号字符）的位置。
			//在这里就是将l,p这两个指针指向下一个','逗号后面一位的位置,也就是根据 ',' 来分别取值
			l = strchr(l, ',') + 1;
			p = strchr(p, ',') + 1;
			
			steps[i] = step;
			scales[i] = scale;
		}
		//赋值给网络的成员
		net->scales = scales;
		net->steps = steps;
		net->num_steps = n;
	}
	else if (net->policy == EXP)
	{
		net->gamma = option_find_float(options, "gamma", 1);
	}
	else if (net->policy == SIG)
	{
		net->gamma = option_find_float(options, "gamma", 1);
		net->step = option_find_int(options, "step", 1);
	}
	else if (net->policy == POLY|| net->policy == RANDOM)
	{
		//todo:源码不操作的,目前不知道为啥不操作,后续再看
	}
	net->max_batches = option_find_int(options, "max_batches", 0);

}

int is_network(section *s)
{
	return (strcmp(s->type, "[net]") == 0) || 
		(strcmp(s->type, "[network]") == 0);
}

//解析卷积层参数并创建卷积层对象
convolutional_layer parse_convolutional(list *options,size_params params)
{
	// 获取卷积核个数，
	int n = option_find_int(options, "filters", 1);
	// 获取卷积核尺寸
	int size = option_find_int(options, "size", 1);
	// 卷积窗口滑动步长
	int stride = option_find_int(options, "stride", 1);
	// 是否在输入图像四周补0,若需要补0,值为1；若配置文件中没有指定，则设为0,不补0
	int pad = option_find_int_quiet(options, "pad", 0);
	// 四周补0的长读，下面这句代码多余，有if(pad)这句就够了
	int padding = option_find_int_quiet(options, "padding", 0);
	if (pad) padding = size / 2;   // 如若需要补0,补0长度为卷积核一半长度（往下取整），这对应same补0策略

	//获取当前层的激活函数类型,默认 logistic
	char *activation_s = option_find_str(options, "activation", "logistic");
	ACTIVATION activation = get_activation(activation_s);

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch = params.batch;

	//上一层的输出,也就是这层的输入图像的h,w,c都必须大于0 ,否则报错
	if (!(h && w && c)) error("layer bdfore convolution layer mast output image.");

	// 是否进行规范化，1表示进行规范化，若配置文件中没有指定，则设为0,即默认不进行规范化
	int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
	// 是否对权重进行二值化，1表示进行二值化，若配置文件中没有指定，则设为0,即默认不进行二值化
	int binary = option_find_int_quiet(options, "binary", 0);
	// 是否对权重以及输入进行二值化，1表示是，若配置文件中没有指定，则设为0,即默认不进行二值化
	int xnor = option_find_int_quiet(options, "xnor", 0);

	//以上已经获取到了构建一层卷积层的所有参数，现在可以用这些参数构建卷积层了
	convolutional_layer layer = make_convolutional_layer(batch, h, w, c, n, size, stride, padding, activation, batch_normalize, binary, xnor, params.net.adam);
	
	//todo:下面的参数目前不知道什么意思
	layer.flipped = option_find_int_quiet(options, "flipped", 0);
	layer.dot = option_find_float_quiet(options, "dot", 0);
	if (params.net.adam) {
		layer.B1 = params.net.B1;
		layer.B2 = params.net.B2;
		layer.eps = params.net.eps;
	}

	return layer;
}

//构建激活函数层
activation_layer parse_activation(list *options, size_params params)
{
	//默认激活函数是线性的linear
	char *activation_s = option_find_str(options, "activation", "linear");
	ACTIVATION activation = get_activation(activation_s);

	layer l = make_activation_layer(params.batch, params.inputs, activation);

	l.out_h = params.h;
	l.out_w = params.w;
	l.out_c = params.c;
	l.h = params.h;
	l.w = params.w;
	l.c = params.c;

	return l;
}

connected_layer parse_connected(list *options, size_params params)
{
	int output = option_find_int(options, "output", 1);
	char *activation_s = option_find_str(options, "activation", "logistic");
	ACTIVATION activation = get_activation(activation_s);
	int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

	connected_layer layer = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize);

	return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
	//获取评价函数类型,默认使用平方差
	char *type_s = option_find_str(options, "type", "sse");
	COST_TYPE type = get_cost_type(type_s);

	float scale = option_find_float_quiet(options, "scale", 1);
	cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
	layer.ratio = option_find_float_quiet(options, "ratio", 0);
	layer.thresh = option_find_float_quiet(options, "thresh", 0); 

	return layer;
}

//yolo1的损失层,暂时不用
//detection_layer parse_detection(list *options, size_param params)
//{
//
//}

//softmax函数本质就是将一个K维的任意实数向量压缩（映射）成另一个K维的实数向量，其中向量中的每个元素取值都介于（0，1）之间。
//一般用于神经网络的最后一层,做分类的映射层
softmax_layer parse_softmax(list *options, size_params params)
{
	int groups = option_find_int_quiet(options, "groups", 1);

	softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);

	// softmax的温度参数，温度参数对于softmax还是比较重要的，当temperature很大时，即趋于正无穷时，所有的激活值对应的激活概率趋近于相同
	// （激活概率差异性较小）；而当temperature很低时，即趋于0时，不同的激活值对应的激活概率差异也就越大。
	// 可以参考博客：http://www.cnblogs.com/maybe2030/p/5678387.html?utm_source=tuicool&utm_medium=referral
	layer.temperature = option_find_float_quiet(options, "temperature", 1);

	//TODO
	char *tree_file = option_find_str(options, "tree", 0);
	if (tree_file) layer.softmax_tree = read_tree(tree_file);
	return layer;
}

normalization_layer parse_normalization(list *options, size_params params)
{
	float alpha = option_find_float(options, "alpha", .0001);
	float beta = option_find_float(options, "beta", .75);
	//kappa?背靠背?哈哈
	//TODO:
	float kappa = option_find_float(options, "kappa", 1);
	int size = option_find_int(options, "size", 5);
	layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
	return l;
}

layer parse_batchnorm(list *options, size_params params)
{
	layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c);
	return l;
}

//最大池化层
maxpool_layer parse_maxpool(list *options, size_params params)
{
	int stride = option_find_int(options, "stride", 1);
	int size = option_find_int(options, "size", stride);
	int padding = option_find_int_quiet(options, "padding", (size - 1) / 2);

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch = params.batch;
	//最大池化层前面的层的输出必须是正确的图
	if (!(h && w && c)) error("Layer before maxpool layer must output image.");

	maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride, padding);
	return layer;
}

//连接层
//也就是把两个c h w都相同的两个层相加成一个相同c h w的层。
//同位置的元素相加的吧
layer parse_shortcut(list *options, size_params params, network net)
{
	char *l = option_find(options, "from");
	int index = atoi(l);
	if (index < 0) index = params.index + index;

	int batch = params.batch;
	layer from = net.layers[index];

	layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

	char *activation_s = option_find_str(options, "activation", "linear");
	ACTIVATION activation = get_activation(activation_s);
	s.activation = activation;
	return s;
}

layer parse_region(list *options, size_params params)
{
	int coords = option_find_int(options, "coords", 4);
	int classes = option_find_int(options, "classes", 20);
	int num = option_find_int(options, "num", 1);

	layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords);
	assert(l.outputs == params.inputs);

	l.log = option_find_int_quiet(options, "log", 0);
	l.sqrt = option_find_int_quiet(options, "sqrt", 0);

	l.softmax = option_find_int(options, "softmax", 0);
	l.max_boxes = option_find_int_quiet(options, "max", 30);
	l.jitter = option_find_float(options, "jitter", .2);
	l.rescore = option_find_int_quiet(options, "rescore", 0);

	l.thresh = option_find_float(options, "thresh", .5);
	l.classfix = option_find_int_quiet(options, "classfix", 0);
	l.absolute = option_find_int_quiet(options, "absolute", 0);
	l.random = option_find_int_quiet(options, "random", 0);

	l.coord_scale = option_find_float(options, "coord_scale", 1);
	l.object_scale = option_find_float(options, "object_scale", 1);
	l.noobject_scale = option_find_float(options, "noobject_scale", 1);
	l.class_scale = option_find_float(options, "class_scale", 1);
	l.bias_match = option_find_int_quiet(options, "bias_match", 0);

	char *tree_file = option_find_str(options, "tree", 0);
	if (tree_file) l.softmax_tree = read_tree(tree_file);
	char *map_file = option_find_str(options, "map", 0);
	if (map_file) l.map = read_map(map_file);

	char *a = option_find_str(options, "anchors", 0);
	if (a) {
		int len = strlen(a);
		int n = 1;
		int i;
		for (i = 0; i < len; ++i) {
			if (a[i] == ',') ++n;
		}
		for (i = 0; i < n; ++i) {
			float bias = atof(a);
			l.biases[i] = bias;
			a = strchr(a, ',') + 1;
		}
	}
	return l;
}


layer parse_reorg(list *options, size_params params)
{
	int stride = option_find_int(options, "stride", 1);
	int reverse = option_find_int_quiet(options, "reverse", 0);
	int flatten = option_find_int_quiet(options, "flatten", 0);
	int extra = option_find_int_quiet(options, "extra", 0);

	int batch, h, w, c;
	h = params.h;
	w = params.w;
	c = params.c;
	batch = params.batch;
	if (!(h && w && c)) error("Layer before reorg layer must output image.");

	layer layer = make_reorg_layer(batch, w, h, c, stride, reverse, flatten, extra);
	return layer;
}

route_layer parse_route(list *options, size_params params, network net)
{
	char *l = option_find(options, "layers");
	int len = strlen(l);
	if (!l) error("Route Layer must specify input layers");
	int n = 1;
	int i;
	for (i = 0; i < len; ++i) {
		if (l[i] == ',') ++n;
	}

	int *layers = calloc(n, sizeof(int));
	int *sizes = calloc(n, sizeof(int));
	for (i = 0; i < n; ++i) {
		int index = atoi(l);
		l = strchr(l, ',') + 1;
		if (index < 0) index = params.index + index;
		layers[i] = index;
		sizes[i] = net.layers[index].outputs;
	}
	int batch = params.batch;

	route_layer layer = make_route_layer(batch, n, layers, sizes);

	convolutional_layer first = net.layers[layers[0]];
	layer.out_w = first.out_w;
	layer.out_h = first.out_h;
	layer.out_c = first.out_c;
	for (i = 1; i < n; ++i) {
		int index = layers[i];
		convolutional_layer next = net.layers[index];
		if (next.out_w == first.out_w && next.out_h == first.out_h) {
			layer.out_c += next.out_c;
		}
		else {
			layer.out_h = layer.out_w = layer.out_c = 0;
		}
	}

	return layer;
}
network parse_network_cfg(char * filename)
{
	list *sections = read_cfg(filename);
	printf_cfg_list(sections);

	//读取配置文件第一部分的节点,里面都是网络的通用参数
	node *n = sections->front;
	if (!n)
	{
		error("config file has no section");
	}
	network net = make_network(sections->size - 1);
	
	//所用显卡号
	//暂时还不同gpu
	net.gpu_index = gpu_index;

	//提取通用参数层的变量
	section *s = (section*)n->val;
	list *options = s->options;
	if (!is_network(s)) error("first section must be [net] or [network]");
	parse_net_option(options, &net);
	
	size_params params;
	params.h = net.h;
	params.w = net.w;
	params.c = net.c;
	params.inputs = net.inputs;
	params.batch = net.batch;
	params.time_steps = net.time_steps;
	params.net = net;

	size_t workspace_size = 0;
	//切换到下一层参数
	n = n->next;
	int count = 0;
	free_section(s);

	//打印网络结构时,作为表头
	fprintf(stderr, "layer     filters    size              input                output\n");
	while (n)
	{
		params.index = count;
		fprintf(stderr, "%5d", count);
		s = (section*)n->val;
		options = s->options;

		//构建各种层,塞入net中
		layer l = { 0 };
		LAYER_TYPE lt = string_to_layer_type(s->type);

		if (lt == CONVOLUTIONAL) {
			//yolo2 num.1
			l = parse_convolutional(options, params);
		}
		else if (lt == DECONVOLUTIONAL) {
			//l = parse_deconvolutional(options, params);
		}
		else if (lt == LOCAL) {
			//l = parse_local(options, params);
		}
		else if (lt == ACTIVE) {
			l = parse_activation(options, params);
		}
		else if (lt == RNN) {
			//l = parse_rnn(options, params);
		}
		else if (lt == GRU) {
			//l = parse_gru(options, params);
		}
		else if (lt == CRNN) {
			//l = parse_crnn(options, params);
		}
		else if (lt == CONNECTED) {
			l = parse_connected(options, params);
		}
		else if (lt == CROP) {
			//l = parse_crop(options, params);
		}
		else if (lt == COST) {
			l = parse_cost(options, params);
		}
		else if (lt == REGION) {
			//yolo2 num.2
			l = parse_region(options, params);
		}
		else if (lt == DETECTION) {
			//l = parse_detection(options, params);
		}
		else if (lt == SOFTMAX) {
			l = parse_softmax(options, params);
			net.hierarchy = l.softmax_tree;
		}
		else if (lt == NORMALIZATION) {
			l = parse_normalization(options, params);
		}
		else if (lt == BATCHNORM) {
			//yolo2 num.3
			l = parse_batchnorm(options, params);
		}
		else if (lt == MAXPOOL) {
			//yolo2 num.4
			l = parse_maxpool(options, params);
		}
		else if (lt == REORG) {
			//yolo2 num.5
			l = parse_reorg(options, params);
		}
		else if (lt == AVGPOOL) {
			//l = parse_avgpool(options, params);
		}
		else if (lt == ROUTE) {
			//yolo2 num.6
			l = parse_route(options, params, net);
		}
		else if (lt == SHORTCUT) {
			l = parse_shortcut(options, params, net);
		}
		else if (lt == DROPOUT) {
			//l = parse_dropout(options, params);
			//l.output = net.layers[count - 1].output;
			//l.delta = net.layers[count - 1].delta;

		}
		else {
			fprintf(stderr, "Type not recognized: %s\n", s->type);
		}

		//获取每一层的一些通用参数,如果配置文件里有的话
		//TODO:以下参数的注释
		l.truth = option_find_int_quiet(options, "truth", 0);
		l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
		l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
		l.dontload = option_find_int_quiet(options, "dontload", 0);
		l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
		l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
		l.smooth = option_find_float_quiet(options, "smooth", 0);

		//提示配置文件中未使用的参数
		option_unused(options);

		net.layers[count] = l;

		//寻找最大占内存的那个数,应该是每一层都用同一块内存,为了避免反复申请内存,就直接选一个最大的内存块
		if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;

		//释放局部配置文件链表
		free_section(s);

		//指向下一层的配置文件
		n = n->next;
		++count;
		// 构建每一层之后，如果之后还有层，则更新params.h,params.w,params.c及params.inputs为上一层相应的输出参数
		if (n) {
			//params.h = l.out_h;
			//params.w = l.out_w;
			//params.c = l.out_c;
			//params.inputs = l.outputs;
			
			//TODO:这里做测试
			params.h = 1;
			params.w = 1;
			params.c = 1;
			params.inputs = 1;
		}
	}

	
	//这时候sections就是个空壳,就剩指针变量了,直接释放
	free_list(sections);

	layer out = get_network_output_layer(net);
	net.outputs = out.outputs;
	net.truths = out.outputs;
	if (net.layers[net.n - 1].truths) net.truths = net.layers[net.n - 1].truths;
	net.output = out.output;
	net.input = calloc(net.inputs*net.batch, sizeof(float));
	//net.truth = calloc(net.truths*net.batch, sizeof(float));

	if (workspace_size) {
		//printf("%ld\n", workspace_size);

		net.workspace = calloc(1, workspace_size);

	}
	return net;
}

list * read_cfg(char * filename)
{
	FILE *file = fopen(filename,"r");
	if (file == 0)
		file_error(filename);
	
	char *line;
	int nu = 0;

	list *sections = make_list();

	section *current = 0;

	while ((line = fgetl(file)) != 0)
	{
		nu++;
		strip(line);
		switch (line[0])
		{
		case '[' :
			//中括号开头,不是网络基础信息层,就是具体的某一层
			//所以就要申请内存,存放这一层所有的配置到一个 section里,以便放到 list sections里
			current = malloc(sizeof(section));
			list_insert(sections, current);

			current->options = make_list();
			current->type = line;
			break;
		case '\0':
		case '#' :
		case ';':
			free(line);
			break;
		default:
			if (!read_option(line,current->options))
			{
				fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
				free(line);
			}
			break;
		}
		
	}

	fclose(file);
	return sections;
}

void load_weights(network * net, char * filename)
{
	//todo
}
