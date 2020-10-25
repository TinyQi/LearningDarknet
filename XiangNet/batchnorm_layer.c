#include "batchnorm_layer.h"
#include <stdlib.h>
#include <stdio.h>
#include "blas.h"
layer make_batchnorm_layer(int batch, int w, int h, int c)
{
	fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w, h, c);
	layer l = { 0 };
	l.type = BATCHNORM;
	l.batch = batch;
	l.h = l.out_h = h;
	l.w = l.out_w = w;
	l.c = l.out_c = c;
	l.output = calloc(h * w * c * batch, sizeof(float));
	l.delta = calloc(h * w * c * batch, sizeof(float));
	l.inputs = w*h*c;
	l.outputs = l.inputs;

	//下面scale和bias是以通道数来申请内存的,说明batch norm是一个batch里,所有特征图相同通道的分布一起做归一化
	//缩放参数,可学习参数,为了弥补归一化后激活函数非线性这个特性的缺失,特此增加scale和bias这两个参数来增加一部分非线性的能力
	l.scales = calloc(c, sizeof(float));
	l.scale_updates = calloc(c, sizeof(float));
	//偏移参数
	l.biases = calloc(c, sizeof(float));
	l.bias_updates = calloc(c, sizeof(float));
	int i;
	//所有缩放参数都初始化为1
	for (i = 0; i < c; ++i) {
		l.scales[i] = 1;
	}
	
	l.mean = calloc(c, sizeof(float));
	l.variance = calloc(c, sizeof(float));

	l.rolling_mean = calloc(c, sizeof(float));
	l.rolling_variance = calloc(c, sizeof(float));

	l.forward = forward_batchnorm_layer;
	l.backward = backward_batchnorm_layer;

	return l;
}



void forward_batchnorm_layer(layer l, network net)
{
	if (l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
	if (l.type == CONNECTED) {
		l.out_c = l.outputs;
		l.out_h = l.out_w = 1;
	}
	copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
	if (net.train) {
		mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
		variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

		//以下4行代码就是实现,指数加权移动平均,目的就是为了在传播过程中,给权重到前几层的参数,这样比较有关联性
		//这里有个问题啊,这rolling_mean初始化为0,乘以这个0.99不还是0嘛?得知道在哪赋值
		//解答:仔细看下面的axpy_cpu函数,里面给l.rolling_mean赋值了
		// rolling_mean和rolling_variance在预测过程(推理过程)才用到
		scal_cpu(l.out_c, .99, l.rolling_mean, 1);
		axpy_cpu(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
		scal_cpu(l.out_c, .99, l.rolling_variance, 1);
		axpy_cpu(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);

		//具体实现归一化算法,就是每个值减去均值,再除以标准差(方差开根号,为了避免标准差为0出现除以0的错误情况,特此加了一个很小的值做容错)
		normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);
		//todo:x_norm的作用?猜测是训练过程中需要保存归一化后的输出,反向传播的时候可以反推回激活函数,但这个不就是l.output,反向传播过程中l.output这个参数不是也存在嘛
		copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
	}
	else {
		normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
	}

	//归一化后,进行缩放和偏移,由于归一化使激活值的分布缺失了部分的非线性属性,所以这里利用缩放scale和偏移bias两个可以学习的参数增加其非线性的属性
	scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
	add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}

void backward_batchnorm_layer(layer l, network net)
{
}
