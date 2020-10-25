#include "connected_layer.h"
#include <stdlib.h>
#include <stdio.h>

#include "utils.h"
connected_layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize)
{
	int i = 0;
	connected_layer l = { 0 };
	l.type = CONNECTED;

	l.inputs = inputs;
	l.outputs = outputs;
	l.batch = batch;
	l.batch_normalize = batch_normalize;

	//全连接层的输出图宽高都为1,就是很多层通道的1x1的窗口
	l.h = 1;
	l.w = 1;
	//通道就是单张输入图输入元素个数
	l.c = inputs;
	
	// 全连接层的输出图高为1,宽也为1
	l.out_h = 1;                                
	l.out_w = 1;
	// 全连接层输出图片的通道数等于一张输入图片对应的输出元素个数
	l.out_c = outputs;                          

	//全连接层的所有输出(整个batch)
	l.output = calloc(batch * outputs, sizeof(float));
	//全连接层的局部残差
	l.delta = calloc(batch * outputs, sizeof(float));

	// 由下面forward_connected_layer()函数中调用的gemm()可以看出，l.weight_updates应该理解为outputs行，inputs列
	l.weight_updates = calloc(inputs * outputs, sizeof(float));
	// 全连接层偏置更新值个数就等于一张输入图片的输出元素个数
	l.bias_updates = calloc(outputs, sizeof(float));

	// 由下面forward_connected_layer()函数中调用的gemm()可以看出，l.weight应该理解为outputs行，inputs列
	// 全连接层就是全图的所有元素映射到结果图上的一点(与inputs个参数相乘),然后结果图有outputs个元素,也就是说参数总共有outputs * inputs个
	l.weights = calloc(outputs * inputs, sizeof(float));
	// 全连接层偏置个数就等于一张输入图片的输出元素个数
	l.biases = calloc(outputs, sizeof(float));

	//初始化参数
	float scale = sqrt(2. / inputs);
	for (i = 0; i < outputs*inputs; ++i) {
		l.weights[i] = scale*rand_uniform(-1, 1);
	}

	// 初始化所有偏置值为0
	for (i = 0; i < outputs; ++i) {
		l.biases[i] = 0;
	}

	//TODO:暂时不知道这个作用
	if (batch_normalize) {
		l.scales = calloc(outputs, sizeof(float));
		l.scale_updates = calloc(outputs, sizeof(float));
		for (i = 0; i < outputs; ++i) {
			l.scales[i] = 1;
		}

		l.mean = calloc(outputs, sizeof(float));
		l.mean_delta = calloc(outputs, sizeof(float));
		l.variance = calloc(outputs, sizeof(float));
		l.variance_delta = calloc(outputs, sizeof(float));

		l.rolling_mean = calloc(outputs, sizeof(float));
		l.rolling_variance = calloc(outputs, sizeof(float));

		l.x = calloc(batch*outputs, sizeof(float));
		l.x_norm = calloc(batch*outputs, sizeof(float));
	}

	l.activation = activation;
	fprintf(stderr, "connected                            %4d  ->  %4d\n", inputs, outputs);
	return l;
}
