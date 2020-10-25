#include "region_layer.h"
#include "activations.h"
#include "utils.h"
#include "box.h"
#include "blas.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//yolo2的损失层


layer make_region_layer(int batch, int h, int w, int n, int classes, int coords)
{
	layer l = { 0 };
	l.type = REGION;

	/// 以下众多参数含义参考layer.h中的注释
	l.n = n;                                                ///< 一个cell（网格）中预测多少个矩形框（box）
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = n*(classes + coords + 1);                         ///< region_layer输出的通道数
	l.out_w = l.w;                                          ///< region_layer层的输入和输出尺寸一致，通道数也一样，也就是这一层并不改变输入数据的维度
	l.out_h = l.h;
	l.out_c = l.c;
	l.classes = classes;                                    ///< 物体类别种数（训练数据集中所拥有的物体类别总数）
	l.coords = coords;                                      ///< 定位一个物体所需的参数个数（一般值为4,包括矩形中心点坐标x,y以及长宽w,h）
	l.cost = calloc(1, sizeof(float));                      ///< 目标函数值，为单精度浮点型指针
	l.biases = calloc(n * 2, sizeof(float));
	l.bias_updates = calloc(n * 2, sizeof(float));
	l.outputs = h*w*n*(classes + coords + 1);               ///< 一张训练图片经过region_layer层后得到的输出元素个数（等于网格数*每个网格预测的矩形框数*每个矩形框的参数个数）
	l.inputs = l.outputs;                                   ///< 一张训练图片输入到reigon_layer层的元素个数（注意是一张图片，对于region_layer，输入和输出的元素个数相等）
	/**
	* 每张图片含有的真实矩形框参数的个数（30表示一张图片中最多有30个ground truth矩形框，每个真实矩形框有
	* 5个参数，包括x,y,w,h四个定位参数，以及物体类别）,注意30是darknet程序内写死的，实际上每张图片可能
	* 并没有30个真实矩形框，也能没有这么多参数，但为了保持一致性，还是会留着这么大的存储空间，只是其中的
	* 值未空而已.
	*/
	l.truths = 30 * (5);
	l.delta = calloc(batch*l.outputs, sizeof(float));       ///< l.delta,l.input,l.output三个参数的大小是一样的
	/**
	* region_layer的输出维度为l.out_w*l.out_h，等于输入的维度，输出通道数为l.out_c，等于输入通道数，
	* 且通道数等于n*(classes+coords+1)。那region_layer的输出l.output中到底存储了什么呢？存储了
	* 所有网格（grid cell）中预测矩形框（box）的所有信息。看Yolo论文就知道，Yolo检测模型最终将图片
	* 划分成了S*S（论文中为7*7）个网格，每个网格中预测B个（论文中B=2）矩形框，最后一层输出的就是这些
	* 网格中所包含的所有预测矩形框信息。目标检测模型中，作者用矩形框来表示并定位检测到的物体，每个矩形框中
	* 包含了矩形框定位信息x,y,w,h，含有物体的自信度信息c，以及属于各类的概率（如果有20类，那么就有矩形框
	* 中所包含物体属于这20类的概率）。注意了，这里的实现与论文中的描述有不同，首先参数固然可能不同（比如
	* 并不像论文中那样每个网格预测2个box，也有可能更多），更为关键的是，输出维度的计算方式不同，论文中提到
	* 最后一层输出的维度为一个S_w*S_c*(B*5+C)的tensor（作者在论文中是S*S，这里我写成S_w，S_c是考虑到
	* 网格划分维度不一定S_w=S_c=S，不过貌似作者用的都是S_w=S_c的，比如7*7,13*13，总之明白就可以了），
	* 实际上，这里有点不同，输出的维度应该为S_w*S_c*B*(5+C),C为类别数目，比如共有20类；5是因为有4个定位
	* 信息，外加一个自信度信息c，共有5个参数。也即每个矩形框都包含一个属于各类的概率，并不是所有矩形框共有
	* 一组属于各类的概率，这点可以从l.outputs的计算方式中看出（可以对应上，l.out_w = S_w, l.out_c = S_c,
	* l.out_c = B*(5+C)）。知道输出到底存储什么之后，接下来要搞清是怎么存储的，毕竟输出的是一个三维张量，
	* 但实现中是用一个一维数组来存储的，详细的注释可以参考下面forward_region_layer()以及entry_index()
	* 函数的注释，这个东西仅用文字还是比较难叙述的，应该借助图来说明～
	*/
	l.output = calloc(batch*l.outputs, sizeof(float));
	int i;
	for (i = 0; i < n * 2; ++i) {
		l.biases[i] = .5;
	}

	l.forward = forward_region_layer;
	l.backward = backward_region_layer;

	fprintf(stderr, "detection\n");
	srand(0);

	return l;
}


/**
* @brief 计算某个矩形框中某个参数在l.output中的索引。一个矩形框包含了x,y,w,h,c,C1,C2...,Cn信息，
*        前四个用于定位，第五个为矩形框含有物体的自信度信息c，即矩形框中存在物体的概率为多大，而C1到Cn
*        为矩形框中所包含的物体分别属于这n类物体的概率。本函数负责获取该矩形框首个定位信息也即x值在
*        l.output中索引、获取该矩形框自信度信息c在l.output中的索引、获取该矩形框分类所属概率的首个
*        概率也即C1值的索引，具体是获取矩形框哪个参数的索引，取决于输入参数entry的值，这些在
*        forward_region_layer()函数中都有用到，由于l.output的存储方式，当entry=0时，就是获取矩形框
*        x参数在l.output中的索引；当entry=4时，就是获取矩形框自信度信息c在l.output中的索引；当
*        entry=5时，就是获取矩形框首个所属概率C1在l.output中的索引，具体可以参考forward_region_layer()
*        中调用本函数时的注释.
* @param l 当前region_layer
* @param batch 当前照片是整个batch中的第几张，因为l.output中包含整个batch的输出，所以要定位某张训练图片
*              输出的众多网格中的某个矩形框，当然需要该参数.
* @param location 这个参数，说实话，感觉像个鸡肋参数，函数中用这个参数获取n和loc的值，这个n就是表示网格中
*                 的第几个预测矩形框（比如每个网格预测5个矩形框，那么n取值范围就是从0~4），loc就是某个
*                 通道上的元素偏移（region_layer输出的通道数为l.out_c = (classes + coords + 1)），
*                 这样说可能没有说明白，这都与l.output的存储结构相关，见下面详细注释以及其他说明。总之，
*                 查看一下调用本函数的父函数orward_region_layer()就知道了，可以直接输入n和j*l.w+i的，
*                 没有必要输入location，这样还得重新计算一次n和loc.
* @param entry 切入点偏移系数，关于这个参数，就又要扯到l.output的存储结构了，见下面详细注释以及其他说明.
* @details l.output这个参数的存储内容以及存储方式已经在多个地方说明了，再多的文字都不及图文说明，此处再
*          简要罗嗦几句，更为具体的参考图文说明。l.output中存储了整个batch的训练输出，每张训练图片都会输出
*          l.out_w*l.out_h个网格，每个网格会预测l.n个矩形框，每个矩形框含有l.classes+l.coords+1个参数，
*          而最后一层的输出通道数为l.n*(l.classes+l.coords+1)，可以想象下最终输出的三维张量是个什么样子的。
*          展成一维数组存储时，l.output可以首先分成batch个大段，每个大段存储了一张训练图片的所有输出；进一步细分，
*          取其中第一大段分析，该大段中存储了第一张训练图片所有输出网格预测的矩形框信息，每个网格预测了l.n个矩形框，
*          存储时，l.n个矩形框是分开存储的，也就是先存储所有网格中的第一个矩形框，而后存储所有网格中的第二个矩形框，
*          依次类推，如果每个网格中预测5个矩形框，则可以继续把这一大段分成5个中段。继续细分，5个中段中取第
*          一个中段来分析，这个中段中按行（有l.out_w*l.out_h个网格，按行存储）依次存储了这张训练图片所有输出网格中
*          的第一个矩形框信息，要注意的是，这个中段存储的顺序并不是挨个挨个存储每个矩形框的所有信息，
*          而是先存储所有矩形框的x，而后是所有的y,然后是所有的w,再是h，c，最后的的概率数组也是拆分进行存储，
*          并不是一下子存储完一个矩形框所有类的概率，而是先存储所有网格所属第一类的概率，再存储所属第二类的概率，
*          具体来说这一中段首先存储了l.out_w*l.out_h个x，然后是l.out_w*l.out_c个y，依次下去，
*          最后是l.out_w*l.out_h个C1（属于第一类的概率，用C1表示，下面类似），l.out_w*l.outh个C2,...,
*          l.out_w*l.out_c*Cn（假设共有n类），所以可以继续将中段分成几个小段，依次为x,y,w,h,c,C1,C2,...Cn
*          小段，每小段的长度都为l.out_w*l.out_c.
*          现在回过来看本函数的输入参数，batch就是大段的偏移数（从第几个大段开始，对应是第几张训练图片），
*          由location计算得到的n就是中段的偏移数（从第几个中段开始，对应是第几个矩形框），
*          entry就是小段的偏移数（从几个小段开始，对应具体是那种参数，x,c还是C1），而loc则是最后的定位，
*          前面确定好第几大段中的第几中段中的第几小段的首地址，loc就是从该首地址往后数loc个元素，得到最终定位
*          某个具体参数（x或c或C1）的索引值，比如l.output中存储的数据如下所示（这里假设只存了一张训练图片的输出，
*          因此batch只能为0；并假设l.out_w=l.out_h=2,l.classes=2）：
*          xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2-#-xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2，
*          n=0则定位到-#-左边的首地址（表示每个网格预测的第一个矩形框），n=1则定位到-#-右边的首地址（表示每个网格预测的第二个矩形框）
*          entry=0,loc=0获取的是x的索引，且获取的是第一个x也即l.out_w*l.out_h个网格中第一个网格中第一个矩形框x参数的索引；
*          entry=4,loc=1获取的是c的索引，且获取的是第二个c也即l.out_w*l.out_h个网格中第二个网格中第一个矩形框c参数的索引；
*          entry=5,loc=2获取的是C1的索引，且获取的是第三个C1也即l.out_w*l.out_h个网格中第三个网格中第一个矩形框C1参数的索引；
*          如果要获取第一个网格中第一个矩形框w参数的索引呢？如果已经获取了其x值的索引，显然用x的索引加上3*l.out_w*l.out_h即可获取到，
*          这正是delta_region_box()函数的做法；
*          如果要获取第三个网格中第一个矩形框C2参数的索引呢？如果已经获取了其C1值的索引，显然用C1的索引加上l.out_w*l.out_h即可获取到，
*          这正是delta_region_class()函数中的做法；
*          由上可知，entry=0时,即偏移0个小段，是获取x的索引；entry=4,是获取自信度信息c的索引；entry=5，是获取C1的索引.
*          l.output的存储方式大致就是这样，个人觉得说的已经很清楚了，但可视化效果终究不如图文说明～
*/
int entry_index(layer l, int batch, int location, int entry)
{
	//location:按每个网格一个预测框的顺序排下来，第几个预测框，比如2*2的输出结果图，4个网格，每个网格2个预测框，
	//         那么所有预测框排列顺序就是tl1,tr1,bl1,br1,tl2,tr2,bl2,br2,如果location=5,那么就是指tl2这个预测框
	//entry:代表一组结果数组里，取哪一类结果，比如总共有x,y,w,h,c,C1,C2，这7类结果，entry代表的就是第几类，
	//      entry=0就是指x类的首地址，entry=3就是指w类的首地址

	//n:就是每个网格对应的第几个预测矩形框
	int n = location / (l.w * l.h);
	//loc:每一个小类里的索引，比如loc=2,在上面的例子里就是第三个网格中的矩形框相应的预测结果索引，
	//    拿x来说，就是xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2中的xxxx的第三个x

	int loc = location % (l.w*l.h);
	return batch*l.outputs + n*l.w*l.h*(l.coords + l.classes + 1) + entry*l.w*l.h + loc;
}

/** 获取某个矩形框的4个定位信息（根据输入的矩形框索引从l.output中获取该矩形框的定位信息x,y,w,h）.
* @param x region_layer的输出，即l.output，包含所有batch预测得到的矩形框信息
* @param biases
* @param n
* @param index 矩形框的首地址（索引，矩形框中存储的首个参数x在l.output中的索引）
* @param i 第几行（region_layer维度为l.out_w*l.out_c，通道数为）
* @param j
* @param w
* @param h
* @param stride个数
*/
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
	//解码过程,将预测值编码成归一化的坐标信息
	// 这里的x已经经过softmax函数处理了
	box b;
	b.x = (i + x[index + 0 * stride]) / w;
	b.y = (j + x[index + 1 * stride]) / h;

	//论文中的解码公式:exp(tw)*pw:这里的tw就是网络预测的值,而这个pw就是先验框的宽度预设值,在这里就是biases[2 * n]
	b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
	b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
	return b;
}

//todo
float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
	box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
	float iou = box_iou(pred, truth);

	//编码过程,
	// 将归一化的坐标信息编码成yolo2的需求:tx,ty是基于cell左上角的偏移量,取值[0,1],
	//							tw,th也是yolo2的特殊编码方式,
	//							解码过程是:label.w = pw * exp(tw),这里,label.是归一化坐标,pw是先验框的w值,tw是网络预测值
	float tx = (truth.x*w - i);// 就是在"label.x = (i + tx)/w"这个公式中求tx
	float ty = (truth.y*h - j);
	float tw = log(truth.w*w / biases[2 * n]); // 这里的编码方式是解码过程的倒推,就是"label.w = pw * exp(tw)"这个公式中求tw
	float th = log(truth.h*h / biases[2 * n + 1]);

	//上面的代码已经将标签编码成yolo的结果形式,然后再跟网络预测进行损失计算
	//损失就是标签值(编码后)减去预测值
	delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
	delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);

	delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
	delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);

	//注意,上面的stride的理解需要看结构信息的内存结构
	// xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C1C1C1C1C1C2C2C2C2C2C2-##-xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C2C1C2C1C2C1C2C1C2C1C2
	// 看,上面就是l.output里保存的预测信息,是   1.每个元素排在一起;2.每一个等级的先验框排在一起;3.然后再是一个batch
	// 所以这里的stride应该是l.w*l.h,因为一个等级的先验框有 "l.w*l.h" 个
	return iou;
}

void delta_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, int stride, float *avg_cat)
{
	int i, n;
	if (hier) {
		//以下是yolo9000用的一些转换手法
		/*float pred = 1;
		while (class >= 0) {
			pred *= output[index + stride*class];
			int g = hier->group[class];
			int offset = hier->group_offset[g];
			for (i = 0; i < hier->group_size[g]; ++i) {
				delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
			}
			delta[index + stride*class] = scale * (1 - output[index + stride*class]);

			class = hier->parent[class];
		}
		*avg_cat += pred;*/
	}
	else {
		for (n = 0; n < classes; ++n) {
			//损失就是,
			// 如果当前的n的类型是标签(class)的类型,那么目标就是1,所以损失就是1 - 当前类型概率,
			// 如果当前n的类型不是标签(class)的类型,那么目标就是0,所以损失就是 0 - 当前类型概率,
			delta[index + stride*n] = scale * (((n == class) ? 1 : 0) - output[index + stride*n]);
			if (n == class) *avg_cat += output[index + stride*n];
		}
	}
}


/**
* @param l
* @param net
* @details 本函数多次调用了entry_index()函数，且使用的参数不尽相同，尤其是最后一个参数，通过最后一个参数，
*          可以确定出region_layer输出l.output的数据存储方式。为方便叙述，假设本层输出参数l.w = 2, l.h= 3,
*          l.n = 2, l.classes = 2, l.coords = 4, l.c = l.n * (l.coords + l.classes + 1) = 21,
*          l.output中存储了所有矩形框的信息参数，每个矩形框包括4条定位信息参数x,y,w,h，一条自信度（confidience）
*          参数c，以及所有类别的概率C1,C2（本例中，假设就只有两个类别，l.classes=2），那么一张样本图片最终会有
*          l.w*l.h*l.n个矩形框（l.w*l.h即为最终图像划分层网格的个数，每个网格预测l.n个矩形框），那么
*          l.output中存储的元素个数共有l.w*l.h*l.n*(l.coords + 1 + l.classes)，这些元素全部拉伸成一维数组
*          的形式存储在l.output中，存储的顺序为：
*          xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C1C1C1C1C1C2C2C2C2C2C2-##-xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C2C1C2C1C2C1C2C1C2C1C2
*          文字说明如下：-##-隔开分成两段，左右分别是代表所有网格的第1个box和第2个box（因为l.n=2，表示每个网格预测两个box），
*          总共有l.w*l.h个网格，且存储时，把所有网格的x,y,w,h,c信息聚到一起再拼接起来，因此xxxxxx及其他信息都有l.w*l.h=6个，
*          因为每个有l.classes个物体类别，而且也是和xywh一样，每一类都集中存储，先存储l.w*l.h=6个C1类，而后存储6个C2类，
*         更为具体的注释可以函数中的语句注释（注意不是C1C2C1C2C1C2C1C2C1C2C1C2的模式，而是将所有的类别拆开分别集中存储）。
* @details 自信度参数c表示的是该矩形框内存在物体的概率，而C1，C2分别表示矩形框内存在物体时属于物体1和物体2的概率，
*          因此c*C1即得矩形框内存在物体1的概率，c*C2即得矩形框内存在物体2的概率
*/
void forward_region_layer(const layer l, network net)
{
	// 损失的计算,以预测2种类别的数据为例
	// yolo2是针对先验框来做预测的,预测的维度有:4维位置信息(xywh),1维置信度,2维类别信息,按照yolo2定好的每个网格有5个先验框的话,
	// 那每个网格就有(4+1+2)*5=35维,也就是35个通道,每个网格里的所有维度都会涉及到损失的计算,具体的损失计算看下面
	// 
	// 以下损失是具体某个先验框来说,基于一次前向训练来说无非就是在外循环加上batch,w,h这三层,遍历一个batch的所有网格

	// 先验框分负责预测物体的和不负责预测物体的,
	// 负责预测物体的先验框:与gd的iou最高的先验框就是负责预测该gd的先验框
	// 不负责预测物体的先验框:除了负责预测物体之外的所有先验框
	// 
	// 负责预测物体的先验框的损失计算:
	// 1.预测框的xywh与gd的xywh的损失计算,采用mse损失
	// 2.置信度的损失计算,
	//		a.如果没有打开rescore这个参数,则是预测框置信度和1之间的mse损失;
	//  	b.如果打开了rescore这个参数,那么就是以预测框与gd的iou为目标,计算预测框置信度与这个iou之间的mse损失
	// 3.类别的损失计算,计算预测框分类结果与gd类别结果的mse损失,比如类别2个类别维度[0.5,0.88],gd的类别维度[0,1],计算对应类别的mse
	// 不负责预测物体的先验框的损失计算
	// 4.当参与训练的图片数量没有达到12800之前(论文值),计算预测框与先验框的xywh的mse损失(先验框的位置是固定从配置文件里读来的)
	//		a.这一步的目的就是让网络先学习先验框的位置信息,加速收敛
	// 5.参与训练图片数量超过12800之后就不再计算xywh的损失了,但是分情况计算置信度损失
	//		a.先验框的预测置信度超过0.6(论文值),则不计算置信度损失
	//		b.先验框的预测置信度不超过0.6,则计算置信度的mse损失,置信度目标是0,因为这时候判定正常情况是不存在物体的,你有一定的置信度,那么给与损失

	int i, j, b, t, n;
	//损失层的输出等于输入
	memcpy(l.output, l.inputs, l.batch * l.outputs * sizeof(float));
	//将所有敏感图的元素都初始化为0
	memset(l.delta, 0, l.batch * l.outputs * sizeof(float));
	
	//下面循环体是对每个预测框的结果进行解码
	// xy是基于cell的中心点的偏移量,经过logistic激活函数,也就是sigmoid函数后,被限定在[0,1]
	for (b = 0; b < l.batch; ++b) {
		// 注意region_layer层中的l.n含义是每个cell grid（网格）中预测的矩形框个数（不是卷积层中卷积核的个数）
		for (n = 0; n < l.n; ++n) {
			// 获取 某一中段首个x的地址
			int index = entry_index(l, b, n*l.w*l.h, 0);
			// 注意第二个参数是2*l.w*l.h，也就是从index+l.output处开始，对之后2*l.w*l.h个元素进行logistic激活函数处理，也就是对
			// 一个中段内所有的x,y进行logistic函数处理，这里没有搞明白的是，为什么对x,y进行激活函数处理？后面的w,h呢？
			// 解答:如果不用sigmoid激活函数将偏移量限定在[0,1],那么就是使用正常的偏移量,
			// 而预测框目标的中心可能在图片的任何地方,不限定的话,这个偏移量可能会很大,最终导致模型难收敛
			// 所以为了让预测框的中心点落在cell内,就对偏移量做sigmoid函数处理
			// 而预测框的w和h是可以大于1的,比如很大的框,超出了当前的网格
			activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
			
			// 和上面一样，此处是获取一个中段内首个自信度信息c值的地址，而后对该中段内所有的c值（该中段内共有l.w*l.h个c值）进行logistic激活函数处理
			// 置信度是[0,1]问题,用logistic激活函数(sigmoid函数)处理
			index = entry_index(l, b, n*l.w*l.h, 4);
			activate_array(l.output + index, l.w*l.h, LOGISTIC);
		}
	}

	if (l.softmax_tree) {
		//
	}
	else if (l.softmax)
	{
		int index = entry_index(l, 0, 0, 5);

		//对所有class结果进行softmax,也就是让所有类型的和为1 ,每一个类型的exp值各自除以所有类型exp的总和
		softmax_cpu(net.input + index, l.classes, l.batch*l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
	}

	//如果是非训练过程,则直接退出,因为不需要计算损失啊
	if (!net.train)return;

	//avg开头的变量都是为了打印的平均参数
	float avg_iou = 0;	
	float recall = 0;	//召回率
	float avg_cat = 0;
	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;

	//这四层循环就是遍历每个网格的每个先验框
	//每个先验框都进行置信度的损失计算
	//而且初期迭代(12800次之前),还会计算每个预测框与先验框之间的xywh的损失
	for ( b = 0; b < l.batch; b++)
	{

		//1.-----------------
		//计算负样本的置信度损失
		//计算迭代初期所有预测框与先验框的坐标损失
		//计算
		for ( j = 0; j < l.h; j++)
		{
			for ( i = 0; i < l.w; i++)
			{
				for ( n = 0; n < l.n; n++)//遍历每个网络的所有先验框
				{
					//获取当前先验框的xywh结果的索引
					int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);

					//预测框
					box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);

					//从标签中找出与当前预测框iou最大的框
					float best_iou = 0;
					//这个30是内部定死的,yolo2是最多支持30个标签的
					for (t = 0; t < 30; t++)
					{
						//t * 5是因为一个标签有5个元素,xywh和class
						//b * l.truths 是因为net.truth里面是一个batch的图片,而l.truths是指一张图片拥有的真是标签数量
						//net.truth里的内存结构应该是提前定死的,即使你一张图片里只有一个标,但是都会申请30个标的内存空间
						box truth = float_to_box(net.truth + t * 5 + b * l.truths, 1);

						//如果读到一个空的标签,证明该图片的标签已经读到头了,则跳出,这句话就验证了"net.truth里的内存结构应该是提前定死的"这句话
						if (!truth.x) break;

						//网络预测框与真实标签的iou
						float iou = box_iou(pred, truth);
						
						if (iou > best_iou)
						{
							best_iou = iou;
						}
					}

					int obj_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4);
					avg_anyobj += l.output[obj_index];//用于输出

					l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
					if (best_iou > l.thresh)
					{
						l.delta[obj_index] = 0;
					}
					
					//初期迭代,会让所有网络预测框与先验框计算损失,以此来靠近先验框
					if (*net.seen < 12800)
					{
						box truth = { 0 };
						truth.x = (i + 0.5) / l.w;
						truth.y = (j + 0.5) / l.h;
						//这个l.biases是w和h的值,在make_region_layer()构造函数里面会默认赋值为0.5
						//但是parser.c文件中的解析region层的函数最后又对l.biases进行赋值,赋值的内容就是配置文件里面anchor参数后的值
						//也就是说,尽管构造函数里面默认赋值为0.5,但是只要配置文件里指明了anchor的参数,这里计算与先验框的损失
						truth.w = l.biases[2 * n] / l.w;
						truth.h = l.biases[2 * n + 1] / l.h;
						
						//todo:scale是干嘛的??? 应该就是一个权重,平衡该位置产生的损失在整个损失函数里的占比
						delta_region_box(truth, l.output, l.biases, n, obj_index, i, j, l.w, l.h, l.delta, 0.01, l.w*l.h);
					}
				}
			}
		}

		//2.-----------------
		//根据标签计算负责检测该标签的预测框的各部分损失:xywh,置信度,class
		for (t = 0; t < 30; t++)
		{
			//从net.truth中取出一个标签
			box truth = float_to_box(net.truth + t * 5 + b * l.truths, 1);

			if (!truth.x)break;
			float best_iou = 0;
			int best_n = 0;
			//根据标签的坐标信息,结合iou最大的原则,找到负责预测这个标签的预测框
			i = truth.x * l.w;
			j = truth.y * l.h;

			//将truth移到图片左上角,因为后续计算iou的时候,会把预测框和标签框的左上角都移到(0,0)去计算iou
			box shift_truth = truth;
			shift_truth.x = 0;
			shift_truth.y = 0;
			for ( n = 0; n < l.n; n++)
			{
				int box_index = entry_index(l, b, j * l.w + i + l.w * l.h * n, 0);
				box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, 1);
				
				//是否使用配置文件里的anchor信息
				if (l.bias_match)
				{
					pred.w = l.biases[2 * n] / l.w;
					pred.h = l.biases[2 * n + 1] / l.h;
				}

				pred.x = 0;
				pred.y = 0;

				float iou = box_iou(pred, shift_truth);
				if (best_iou < iou)
				{
					best_iou = iou;
					best_n = n;//记录了当前cell中最佳的预测框,用于与标签框计算损失
				}
			}

			//到这一步就知道具体哪个预测框要跟当前标签计算损失
			//这里知道该预测框在l.output中的内存索引就定位到了具体的预测框
			int box_index = entry_index(l, b, j*l.w + i + best_n * l.w*l.h, 0);

			//计算坐标损失
			//
			float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
			//注意:上面这个权重的意义:l.coord_scale *  (2 - truth.w*truth.h):
			//						  l.coord_scale = 1,
			//						  (2 - truth.w*truth.h):具体物理意义不知道,反正就是预测框越大,则损失越小
			//用于打印,iou大于0.5证明网络找到了该标签
			if (iou > .5) recall += 1;
			avg_iou += iou;
			

			//计算置信度损失,其实就是在上面的"box_index"基础上后移4位,就到了置信度的位置
			int obj_index = entry_index(l, b, j*l.w + i + best_n * l.w*l.h, 4);
			avg_obj += l.output[obj_index];//用于记录平均的置信度
			l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
			if (l.rescore)//如果定义了这个参数,则按以下的方式计算置信度损失
			{
				l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
			}

			//当前标签的类型
			int class = net.truth[t * 5 + b*l.truths + 4];//加4就是第5个数是类型,前四个是xywh

			//计算class损失
			if (l.map) class = l.map[class];//这个是跟yolo9000有关的,暂时用不到
			int class_index = entry_index(l, b, j*l.w + i + best_n * l.w*l.h, 5);//其实就是在上面的"box_index"基础上后移5位,就到了类别的位置
			delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat);
			++count;
			++class_count;
		}
	}

	//好像具体的l.cost这个损失函数的值已经不重要了,反向传播往回传的是l.delta这个值,里面包含了每一个元素的损失
	//在network.c这个文件里的void backward_network(network net)函数中,
	// 有这么一句代码:net.delta = prev.delta;
	// 这句代码是把前一层网络的局部损失指针赋给net.delta指针
	// 而region层的反向传播函数又会让当前region层的delta复制给net.delta,就相当于复制给前一层的网络的l.delta,以此达到反向传播中的链式法则

	//因为yolo2损失函数是上面所有项的平方和,而上面只是计算了每一项的值,所以这里需要平方求和得出最终的l.cost
	(*l.cost) = pow(mag_array(l.delta,l.outputs * l.batch), 2);//这里有点小奇怪,mag_array函数,是求数组的平方和开根号的,但是函数里面已经求了平方和了
	printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, count);

}

void backward_region_layer(const layer l, network net)
{
	//不知道为什么github上的源码把这句话注释了,注释了的话,损失怎么传回去?我暂时打开注释
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}
