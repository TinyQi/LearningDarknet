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

	//ȫ���Ӳ�����ͼ��߶�Ϊ1,���Ǻܶ��ͨ����1x1�Ĵ���
	l.h = 1;
	l.w = 1;
	//ͨ�����ǵ�������ͼ����Ԫ�ظ���
	l.c = inputs;
	
	// ȫ���Ӳ�����ͼ��Ϊ1,��ҲΪ1
	l.out_h = 1;                                
	l.out_w = 1;
	// ȫ���Ӳ����ͼƬ��ͨ��������һ������ͼƬ��Ӧ�����Ԫ�ظ���
	l.out_c = outputs;                          

	//ȫ���Ӳ���������(����batch)
	l.output = calloc(batch * outputs, sizeof(float));
	//ȫ���Ӳ�ľֲ��в�
	l.delta = calloc(batch * outputs, sizeof(float));

	// ������forward_connected_layer()�����е��õ�gemm()���Կ�����l.weight_updatesӦ�����Ϊoutputs�У�inputs��
	l.weight_updates = calloc(inputs * outputs, sizeof(float));
	// ȫ���Ӳ�ƫ�ø���ֵ�����͵���һ������ͼƬ�����Ԫ�ظ���
	l.bias_updates = calloc(outputs, sizeof(float));

	// ������forward_connected_layer()�����е��õ�gemm()���Կ�����l.weightӦ�����Ϊoutputs�У�inputs��
	// ȫ���Ӳ����ȫͼ������Ԫ��ӳ�䵽���ͼ�ϵ�һ��(��inputs���������),Ȼ����ͼ��outputs��Ԫ��,Ҳ����˵�����ܹ���outputs * inputs��
	l.weights = calloc(outputs * inputs, sizeof(float));
	// ȫ���Ӳ�ƫ�ø����͵���һ������ͼƬ�����Ԫ�ظ���
	l.biases = calloc(outputs, sizeof(float));

	//��ʼ������
	float scale = sqrt(2. / inputs);
	for (i = 0; i < outputs*inputs; ++i) {
		l.weights[i] = scale*rand_uniform(-1, 1);
	}

	// ��ʼ������ƫ��ֵΪ0
	for (i = 0; i < outputs; ++i) {
		l.biases[i] = 0;
	}

	//TODO:��ʱ��֪���������
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
