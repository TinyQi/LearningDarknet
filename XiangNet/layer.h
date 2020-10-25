#ifndef LAYER_H
#define LAYER_H

#include "activations.h"
#include "tree.h"

//前向声明,只是为了typedf struct layer ,这样之后在layer结构体里声明layer变量就不需要家sruuct关键词
struct layer;
typedef struct layer layer;

struct network;
typedef struct network network;

typedef enum {
	CONVOLUTIONAL,//卷积
	DECONVOLUTIONAL,//反卷积?
	CONNECTED,//连接层
	MAXPOOL,//最大池化
	SOFTMAX,//SOFTMAX
	DETECTION,//检测?
	DROPOUT,//抛弃层
	CROP,//?
	ROUTE,//?
	COST,//损失层
	NORMALIZATION,//归一化层
	AVGPOOL,//平均池化层
	LOCAL,//?
	SHORTCUT,//?
	ACTIVE,//激活层
	RNN,//循环神经网络层?
	GRU,//?
	CRNN,//?
	BATCHNORM,//标准化层
	NETWORK,//?
	XNOR,//与或非?
	REGION,//区域层,预测层?
	REORG,//?
	BLANK               // 表示未识别的网络层名称
} LAYER_TYPE;

typedef enum {
	SSE, MASKED, L1, SMOOTH
} COST_TYPE;


typedef struct layer
{
	LAYER_TYPE type;
	COST_TYPE cost_type;
	ACTIVATION activation;

	//成员函数,用指针函数实现
	void(*forward)(struct layer, network);
	void(*backward)(struct layer, network);
	void(*update)(struct layer, int a,float b,float c,float d);


	int classes;
	float jitter;


	tree *softmax_tree;

	int flipped;				//?
	float dot;					//?

	float B1;					//?
	float B2;					//?
	float eps;					//?

	int out_h;
	int out_w;
	int out_c;

	int w, h, c;

	float ratio;				//?
	float thresh;				//?

	float temperature;			//softmax层的温度参数,该值越大,不同的激活值差异越小

	int truth;
	int onlyforward;
	int stopbackward;
	int dontload;
	int dontloadscales;
	float learning_rate_scale;
	float smooth;

	// net.workspace的元素个数，为所有层中最大的l.out_h*l.out_w*l.size*l.size*l.c
	// （在make_convolutional_layer()计算得到workspace_size的大小，在parse_network_cfg()中动态分配内存，此值对应未使用gpu时的情况）
	size_t workspace_size;     


	// 该层对应一张输入图片的输出元素个数（一般在各网络层构建函数中赋值，比如make_connected_layer()）
	int outputs;

	// 存储该层所有的输出，维度为l.out_h * l.out_w * l.out_c * l.batch，可知包含整个batch输入图片的输出，一般在构建具体网络层时动态分配内存（比如make_maxpool_layer()中）。
	// 按行存储：每张图片按行铺排成一大行，图片间再并成一行。
	float * output;

	//< 根据region_layer.c判断，这个变量表示一张图片含有的真实值的个数，对于检测模型来说，一个真实的标签含有5个值，
	//< 包括类型对应的编号以及定位矩形框用到的w,h,x,y四个参数，且在darknet中，固定每张图片最大处理30个矩形框（可查看max_boxes参数），
	//< 因此，在region_layer.c的make_region_layer()函数中，赋值为30*5
	int truths;                 

	//卷积核的个数,相当于该层的输出通道数
	int n;

	int bias_match;//判断先验框(anchor)参数从配置文件里读取正确

	//TODO:?
	int binary;
	//TODO:?
	int xnor;
	
	//每个batch含有的图片数
	int batch;

	//滑窗步长
	int stride;

	//卷积核的尺寸
	int size;

	//输入图片四周补0的长度
	int pad;

	//是否进行BN(规范化)
	int batch_normalize;

	//当前层的权重系数(就是卷积参数)
	float *weights;

	//权重更新值,数量对应weights
	float *weight_updates;

	//所有权重参数个数
	int nweights;

	//偏置,就是 Wx+B 里面的B
	float *biases;

	//偏置更新值
	float *bias_updates;

	//所有偏置的数量
	int nbiases;
	
	// 一张输入图片所含的元素个数（一般在各网络层构建函数中赋值，比如make_connected_layer()），第一层的值等于l.h*l.w*l.c，
	// 之后的每一层都是由上一层的输出自动推算得到的（参见parse_network_cfg()，在构建每一层后，会更新params.inputs为上一层的l.outputs）
	int inputs;

	//局部损失,残差
	float *delta;

	//--------TODO上-------------
	float * binary_input;
	float * binary_weights;
	char * cweights;
	//batch norm的缩放参数数组指针
	float * scales;
	//batch norm的缩放参数的更新值数组指针
	float * scale_updates;
	//batch norm的均值参数数组指针
	float * mean;
	//batch norm的方差参数数组指针
	float * variance;
	//均值的指数加权移动平均系数
	float * rolling_mean;
	//方差的指数加权移动平均系数
	float * rolling_variance;

	float * variance_delta;
	float * mean_delta;

	float * x;
	float * x_norm;

	int adam;

	float * m;
	float * v;
	float * bias_m;
	float * scale_m;
	float * bias_v;
	float * scale_v;
	//--------TODO下-------------


	float scale;               // 在dropout层中，该变量是一个比例因子，取值为保留概率的倒数（darknet实现用的是inverted dropout），用于缩放输入元素的值
							   // （在网上随便搜索关于dropout的博客，都会提到inverted dropout），在make_dropout_layer()函数中赋值

	float * cost;             // 目标函数值，该参数不是所有层都有的，一般在网络最后一层拥有，用于计算最后的cost，比如识别模型中的cost_layer层，
							  // 检测模型中的region_layer层
							  
	int coords;                 // 这个参数一般用在检测模型中，且不是所有层都有这个参数，一般在检测模型最后一层有，比如region_layer层，该参数的含义
								// 是定位一个物体所需的参数个数，一般为4个，包括物体所在矩形框中心坐标x,y两个参数以及矩形框长宽w,h两个参数，
								// 可以在darknet/cfg文件夹下，执行grep coords *.cfg，会搜索出所有使用该参数的模型，并可看到该值都设置位4
								
	int log;
	int sqrt;
	int softmax;

	int max_boxes;              /// 每张图片最多含有的标签矩形框数（参看：data.c中的load_data_detection()，其输入参数boxes就是指这个参数），
								/// 什么意思呢？就是每张图片中最多打了max_boxes个标签物体，模型预测过程中，可能会预测出很多的物体，但实际上，
								/// 图片中打上标签的真正存在的物体最多就max_boxes个，预测多出来的肯定存在false positive，需要滤出与筛选，
								/// 可参看region_layer.c中forward_region_layer()函数的第二个for循环中的注释
								
	int rescore;
	int classfix;
	int absolute;
	int random;
	float coord_scale;
	float object_scale;
	float noobject_scale;
	float class_scale;

	/**
	* 这个参数用的不多，仅在region_layer.c中使用，该参数的作用是用于不同数据集间类别编号的转换，更为具体的，
	* 是coco数据集中80类物体编号与联合数据集中9000+物体类别编号之间的转换，可以对比查看data/coco.names与
	* data/9k.names以及data/coco9k.map三个文件（旧版的darknet可能没有，新版的darknet才有coco9k.map这个文件），
	* 可以发现，coco.names中每一个物体类别都可以在9k.names中找到,且coco.names中每个物体类别名称在9k.names
	* 中所在的行数就是coco9k.map中的编号值（减了1,因为在程序数组中编号从0开始），也就是这个map将coco数据集中
	* 的类别编号映射到联和数据集9k中的类别编号（这个9k数据集是一个联和多个数据集的大数集，其名称分类被层级划分，
	* ）（注意两个文件中物体的类别名称大部分都相同，有小部分存在小差异，虽然有差异，但只是两个数据集中使用的名称有所差异而已，
	* 对应的物体是一样的，比如在coco.names中摩托车的名称为motorbike，在联合数据集9k.names，其名称为motorcycle）.
	*/
	int   * map;
}layer;





















#endif // LAYER_H


