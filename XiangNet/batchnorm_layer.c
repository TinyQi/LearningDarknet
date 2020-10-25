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

	//����scale��bias����ͨ�����������ڴ��,˵��batch norm��һ��batch��,��������ͼ��ͬͨ���ķֲ�һ������һ��
	//���Ų���,��ѧϰ����,Ϊ���ֲ���һ���󼤻��������������Ե�ȱʧ,�ش�����scale��bias����������������һ���ַ����Ե�����
	l.scales = calloc(c, sizeof(float));
	l.scale_updates = calloc(c, sizeof(float));
	//ƫ�Ʋ���
	l.biases = calloc(c, sizeof(float));
	l.bias_updates = calloc(c, sizeof(float));
	int i;
	//�������Ų�������ʼ��Ϊ1
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

		//����4�д������ʵ��,ָ����Ȩ�ƶ�ƽ��,Ŀ�ľ���Ϊ���ڴ���������,��Ȩ�ص�ǰ����Ĳ���,�����Ƚ��й�����
		//�����и����Ⱑ,��rolling_mean��ʼ��Ϊ0,�������0.99������0��?��֪�����ĸ�ֵ
		//���:��ϸ�������axpy_cpu����,�����l.rolling_mean��ֵ��
		// rolling_mean��rolling_variance��Ԥ�����(�������)���õ�
		scal_cpu(l.out_c, .99, l.rolling_mean, 1);
		axpy_cpu(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
		scal_cpu(l.out_c, .99, l.rolling_variance, 1);
		axpy_cpu(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);

		//����ʵ�ֹ�һ���㷨,����ÿ��ֵ��ȥ��ֵ,�ٳ��Ա�׼��(�������,Ϊ�˱����׼��Ϊ0���ֳ���0�Ĵ������,�ش˼���һ����С��ֵ���ݴ�)
		normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);
		//todo:x_norm������?�²���ѵ����������Ҫ�����һ��������,���򴫲���ʱ����Է��ƻؼ����,�����������l.output,���򴫲�������l.output�����������Ҳ������
		copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
	}
	else {
		normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
	}

	//��һ����,�������ź�ƫ��,���ڹ�һ��ʹ����ֵ�ķֲ�ȱʧ�˲��ֵķ���������,����������������scale��ƫ��bias��������ѧϰ�Ĳ�������������Ե�����
	scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
	add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}

void backward_batchnorm_layer(layer l, network net)
{
}
