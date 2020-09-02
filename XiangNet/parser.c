#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "parser.h"
#include "list.h"
#include "utils.h"
#include "option_list.h"
#include "cuda.h"

//网络与各个层之间的承接变量、中间变量
typedef struct size_param
{
	int batch;
	int inputs;
	int h, w, c;
	int index;
	int time_steps;
	network net;
}size_param;

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
	network net = make_network(n - 1);
	
	//所用显卡号
	//暂时还不同gpu
	net.gpu_index = gpu_index;

	size_param param;
	
	//提取当前层的变量
	section *s = (section*)n->val;
	list *option = s->options;
	if (!is_network(s)) error("first section must be [net] or [network]");
	
	parse_net_option(option, &net);
	


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
