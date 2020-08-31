#ifndef NETWORK_H
#define NETWORK_H
#include "layer.h"

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
	float learning_rate;

	float *cost;					//损失
	int gpu_index;					//所用gpu的卡号

	int time_steps;					//应该是一轮batch里跳着读的步长
	int notruth;					//?

	int adam;						//?我目前理解为是否开adam优化
	float B1;						//?adam优化相关的参数
	float B2;						//?adam优化相关的参数
	float eps;						//?adam优化相关的参数

	int w, h, c;					//?可能就特指输入图片的长,宽,通道

}network;

//初始化网络,注意这里的n不包括通用参数层,就是那个[net]
network make_network(int n);
#endif // !NETWORK_H

