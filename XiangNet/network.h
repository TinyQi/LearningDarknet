#ifndef NETWORK_H
#define NETWORK_H
#include "layer.h"

typedef enum {
	CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network
{
	int n;                          // 网络总层数
	int batch;                      //一个batch含有的图片张数,此处的batch与subdivisions相乘,才是配置文件里的batch,也就是说配置文件里的batch虽然意思是一轮训练是用这么多张图
									//但是内部其实是分subdivisions轮,每轮图片位此处的batch数量

	int *seen;                      //字面意思,已经看过的图片,也就是目前已经处理过的图片数量
	float epoch;
	int subdivisions;               //再细分,batch后还会再除以这个值,才是一次训练的所有图片
	float momentum;                 //动量
	float decay;                    //权重衰减值
	
	layer *layers;                  //网络中的所有层

	float *cost;					//损失
	int gpu_index;					//所用gpu的卡号

	
	int notruth;					//?

	int adam;						//?我目前理解为是否开adam优化
	float B1;						//?adam优化相关的参数
	float B2;						//?adam优化相关的参数
	float eps;						//?adam优化相关的参数

	int w, h, c;					//?可能就特指输入图片的长,宽,通道

	int inputs;						//一张输入图片的元素个数,默认等于net->h * net->w * net->c
	int outputs;					//一张图片对应的输出元素个数,对于卷积层,可以根据输入及kernel相关参数,算出outputs

	int max_crop;					//?
	int min_crop;					//?
	int center;						//?
	float angle;					//?
	float aspect;					//?
	float exposure;					//?
	float saturation;				//?
	float hue;						//?
	
	


	//调整学习率相关
	learning_rate_policy policy;    //调整学习率的方法
	float gamma;					//具体看parse_net_option(),todo:这里有个问题,如果把这一行挪到learning_rate下面,就会编译不过,报的是没有gamma这个成员,但是很奇怪,明明有的
	float learning_rate;			//初始学习率parse_net_options()中赋值
	
	float scale;					//具体看parse_net_option()
	float power;					//?
	int time_steps;					//应该是一轮batch里跳着读的步长
	int step;						//具体看parse_net_option()
	float *scales;					//具体看parse_net_option()
	int   *steps;					//具体看parse_net_option()
	int num_steps;					//?
	int burn_in;					//具体看parse_net_option()

	int max_batches;				//?最大batch次数?用于终止训练吗?


}network;

//初始化网络,注意这里的n不包括通用参数层,就是那个[net]
network make_network(int n);
#endif // !NETWORK_H

