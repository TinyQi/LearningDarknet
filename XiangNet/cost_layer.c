#include "cost_layer.h"
#include <stdlib.h>
#include <stdio.h>

/*
** ֧������cost functions
** L1    �����ľ���ֵ֮��
** SSE   ��L2������ƽ���ͣ����Բ鿴blas.c�е�l2_cpu()������û�г���1/2����Ϊ���󲿷��������,
ȫ��Ӧ����the sum of squares due to error���ο��ģ�http://blog.sina.com.cn/s/blog_628033fa0100kjjy.html��
** MASKED Ŀǰֻ������darknet9000.cfg��ʹ��
** SMOOTH
*/
COST_TYPE get_cost_type(char *s)
{
	if (strcmp(s, "sse") == 0) return SSE;
	if (strcmp(s, "masked") == 0) return MASKED;
	if (strcmp(s, "smooth") == 0) return SMOOTH;
	if (strcmp(s, "L1") == 0) return L1;
	fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
	return SSE;
}

char *get_cost_string(COST_TYPE a)
{
	switch (a) {
	case SSE:
		return "sse";
	case MASKED:
		return "masked";
	case SMOOTH:
		return "smooth";
	case L1:
		return "L1";
	}
	return "sse";
}

cost_layer make_cost_layer(int batch, int inputs, COST_TYPE type, float scale)
{
	fprintf(stderr, "cost                                           %4d\n", inputs);
	cost_layer l = { 0 };
	l.type = COST;

	l.scale = scale;
	l.batch = batch;
	l.inputs = inputs;
	l.outputs = inputs;
	l.cost_type = type;
	l.delta = calloc(inputs*batch, sizeof(float));
	l.output = calloc(inputs*batch, sizeof(float));
	l.cost = calloc(1, sizeof(float));

	l.forward = forward_cost_layer;
	l.backward = backward_cost_layer;

	return l;
}


void forward_cost_layer(const cost_layer l, network net)
{

}



void backward_cost_layer(const cost_layer l, network net)
{

}