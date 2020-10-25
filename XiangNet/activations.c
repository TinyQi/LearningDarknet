#include "activations.h"
#include <stdio.h>

//���ݼ������ö����,�������ַ���
char *get_activation_string(ACTIVATION a)
{
	switch (a) {
	case LOGISTIC:
		return "logistic";
	case LOGGY:
		return "loggy";
	case RELU:
		return "relu";
	case ELU:
		return "elu";
	case RELIE:
		return "relie";
	case RAMP:
		return "ramp";
	case LINEAR:
		return "linear";
	case TANH:
		return "tanh";
	case PLSE:
		return "plse";
	case LEAKY:
		return "leaky";
	case STAIR:
		return "stair";
	case HARDTAN:
		return "hardtan";
	case LHTAN:
		return "lhtan";
	default:
		break;
	}
	return "relu";
}

//�����ַ�������Ӧ�ļ����ö������
ACTIVATION get_activation(char *s)
{
	if (strcmp(s, "logistic") == 0) return LOGISTIC;
	if (strcmp(s, "loggy") == 0) return LOGGY;
	if (strcmp(s, "relu") == 0) return RELU;
	if (strcmp(s, "elu") == 0) return ELU;
	if (strcmp(s, "relie") == 0) return RELIE;
	if (strcmp(s, "plse") == 0) return PLSE;
	if (strcmp(s, "hardtan") == 0) return HARDTAN;
	if (strcmp(s, "lhtan") == 0) return LHTAN;
	if (strcmp(s, "linear") == 0) return LINEAR;
	if (strcmp(s, "ramp") == 0) return RAMP;
	if (strcmp(s, "leaky") == 0) return LEAKY;
	if (strcmp(s, "tanh") == 0) return TANH;
	if (strcmp(s, "stair") == 0) return STAIR;
	fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
	return RELU;
}

//todo
/*
** ���ݲ�ͬ�ļ�������ͣ����ò�ͬ�ļ��������������Ԫ��x
** ���룺 x    �������Ԫ�أ�������
**       a    ���������
*/
float activate(float x, ACTIVATION a)
{
	/*switch (a) {
	case LINEAR:
		return linear_activate(x);
	case LOGISTIC:
		return logistic_activate(x);
	case LOGGY:
		return loggy_activate(x);
	case RELU:
		return relu_activate(x);
	case ELU:
		return elu_activate(x);
	case RELIE:
		return relie_activate(x);
	case RAMP:
		return ramp_activate(x);
	case LEAKY:
		return leaky_activate(x);
	case TANH:
		return tanh_activate(x);
	case PLSE:
		return plse_activate(x);
	case STAIR:
		return stair_activate(x);
	case HARDTAN:
		return hardtan_activate(x);
	case LHTAN:
		return lhtan_activate(x);
	}*/
	return 0;
}

/**
** �ü������������x�е�ÿһ��Ԫ��
** ���룺 x    ����������飬һ��Ϊ�����ÿ����Ԫ�ļ�Ȩ����Wx+b���ڱ�������Ҳ�൱������������ز�������
**       n    x�к��ж��ٸ�Ԫ��
**       a    ���������
** ˵�����ú������������x�е�Ԫ�أ�ע����������ú���һ������ÿһ�������ǰ�򴫲������У�����forward_connected_layer()�ȣ�
**      �������һ�����ú����������Ϊÿһ����������
*/
void activate_array(float *x, const int n, const ACTIVATION a)
{
	int i;
	// �������x�е�Ԫ��
	for (i = 0; i < n; ++i) {
		// ���ݲ�ͬ�ļ�������ͣ����ò�ͬ�ļ��������
		x[i] = activate(x[i], a);
	}
}



/*
** ���ݲ�ͬ�ļ������ȡ��������ݶȣ�������
** ���룺 x    ��������յ�����ֵ
**       a    ��������ͣ������ļ�������ͼ�activations.h��ö������ACTIVATION�Ķ���
** ����� �������������x�ĵ���ֵ
*/
float gradient(float x, ACTIVATION a)
{
	// ���·ֱ���ȡ���ּ����������ĵ���ֵ���������������ȡ�������ڲ�ע��
	//switch (a) {
	//case LINEAR:
	//	return linear_gradient(x);
	//case LOGISTIC:
	//	return logistic_gradient(x);
	//case LOGGY:
	//	return loggy_gradient(x);
	//case RELU:
	//	return relu_gradient(x);
	//case ELU:
	//	return elu_gradient(x);
	//case RELIE:
	//	return relie_gradient(x);
	//case RAMP:
	//	return ramp_gradient(x);
	//case LEAKY:
	//	return leaky_gradient(x);
	//case TANH:
	//	return tanh_gradient(x);
	//case PLSE:
	//	return plse_gradient(x);
	//case STAIR:
	//	return stair_gradient(x);
	//case HARDTAN:
	//	return hardtan_gradient(x);
	//case LHTAN:
	//	return lhtan_gradient(x);
	//}
	return 0;
}

/*
** ���㼤����Լ�Ȩ����ĵ�����������delta���õ���ǰ�����յ�delta�����ж�ͼ��
** ���룺 x    ��ǰ������������ά��Ϊl.batch * l.out_c * l.out_w * l.out_h��
**       n    l.output��ά�ȣ���Ϊl.batch * l.out_c * l.out_w * l.out_h����������batch�ģ�
**       ACTIVATION    ���������
**       delta     ��ǰ�����ж�ͼ���뵱ǰ�����xά��һ����
** ˵��1�� �ú������������˼�������ڼ�Ȩ����ĵ����������õ���������֮ǰ��ɴ󲿷ּ�������ж�ͼdelta����ӦԪ����ˣ�����˵��øĺ���֮�󣬽��õ��ò����յ����ж�ͼ
** ˵��2�� ����ֱ���������ֵ�󼤻����������ĵ���ֵ����Ϊ����������ʹ�õľ��󲿷ּ���������������ĵ���ֵ����������Ϊ���ֵ�ĺ������ʽ��
�������Sigmoid�����������f(x)�����䵼��ֵΪf(x)'=f(x)*(1-f(x)),����������y=f(x)����ôf(x)'=y*(1-y)��ֻ��Ҫ���ֵy�Ϳ����ˣ�����Ҫ����x��ֵ��
����ʱ��ȷ��darknet����û��ʹ������ļ�����������ڱ���Ҫ����ֵ���ܹ��������ֵ����activiation.c�ļ��У��м����������ʱû������Ҳû�����ϲ鵽����
** ˵��3�� ����l.delta�ĳ�ֵ����������ע�⵽�ڿ�ĳһ����������ʱ�򣬱��������е�backward_convolutional_layer()������û�з����ڴ�֮ǰ��l.delta����ֵ����䣬
**        ֻ����callocΪ�䶯̬�������ڴ棬������l.delta������Ԫ�ص�ֵ��Ϊ0,��ô����ʹ��*=������õ���ֵ����Ϊ0���ǵģ����ֻ��ĳһ�㣬����˵ĳһ���͵Ĳ㣬��ȷ������ɻ�
**        ���������������кܶ��ģ����ж������ͣ�һ����˵�������Ծ����Ϊ���һ�㣬������COST����REGIONΪ���һ�㣬��Щ���У����l.delta����ֵ��������l.delta���ɺ�
**        ��ǰ��㴫���ģ���ˣ����������е�ĳһ��ʱ��l.delta��ֵ��������Ϊ0.
*/
void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
	int i;
	for (i = 0; i < n; ++i) {
		delta[i] *= gradient(x[i], a);
	}
}