#include "blas.h"

#include <float.h>

void fill_cpu(int N, float ALPHA, float * X, int INCX)
{
	int i = 0;
	int iter_end = N / INCX;
	for (i = 0; i < iter_end; i++)
	{
		X[i * INCX] = ALPHA;
	}
}


/*
** ����֯�ļ�����������x��ƽ��ֵ�������mean��һ��ʸ�����������x�Ƕ���3ͨ����ͼƬ����ômean��ά�Ⱦ�Ϊͨ����3
** ��Ҳ��ÿ������ͼƬ��õ�3������ͼ��,Ϊ���㣬���ǳ�������ͨ���ֱ�Ϊ��һ���ڶ�������ͨ��������ÿ��ѵ������Ķ���һ��batch��ͼƬ��
** ������ջ����batch����ͨ����ͼƬ��mean�еĵ�һ��Ԫ�ؾ��ǵ�һ��ͨ����ȫ��batch���������ͼ����Ԫ�ص�ƽ��ֵ����������
** ����������Ҫ�ô�֮һӦ�þ���ʵ��batch normalization�ĵ�һ���ˣ�
** ���룺
**       x         �����������ݣ�����l.output���������Ԫ�ظ���Ϊl.batch*l.outputs
**       batch     һ��batch�а�����ͼƬ��������l.batch
**       filters   �ò���������˲���������Ҳ���ò��������ͼƬ��ͨ����������Ծ��������˵�����Ǻ˵ĸ����ˣ�
**       spatial   �ò�������ÿ���������ͼ�ĳߴ磬Ҳ������l.out_w*l.out_h
**       mean      ��õ�ƽ��ֵ��ά��Ϊfilters��Ҳ��ÿ���˲�����Ӧ��һ����ֵ��ÿ���˲����ᴦ������ͼƬ��
** ˵���� �ú����ľ�����ÿ��Բο���batchnorm_layer.c�е�forward_batchnorm_layer()����
** ˵��2��mean_cpu()������һ���������ѧ���㺯��������֯�ļ���x��ĳЩ���ݵľ�ֵ��x�ľ���洢�ṹ�Ӿ������������
**       ��дע��ʱ����Ҫ�ο���batchnorm_layer.c�е�forward_batchnorm_layer()�����Ըú����ĵ��ã�
**       �����Щע�;ͼ�����һЩ���庬�壬�����Щ�������������⣬������Ҫ��ס������һ��һ�����ѧ���㺯����
**       ��ͬ�ط����øú��������в�ͬ�ĺ��塣
** ˵��3����ֵ����Щ���ݵľ�ֵ��x�а������ڶ����ݣ�mean�е�ÿ��Ԫ�ؾ�����Ӧx����Щ���ݵ�ƽ��ֵ�أ�
**       �˴����ǽ��batchnorm_layer.c�е�forward_batchnorm_layer()�����ĵ��������ͣ�
**       ���е�xΪl.output����l.batch�У�ÿ����l.out_c*l.out_w*l.out_h��Ԫ�أ�ÿһ���ֿ��Էֳ�
**       l.out_c�У�l.out_w*l.out_h�У���ôl.mean�е�ÿһ��Ԫ�أ���ĳһ��ͨ��������batch�������ƽ��ֵ
**       ���������㣬��3���ˣ���ô���ͨ����3����ÿ������ͼƬ�������3������ͼ���������ÿ�����ͼƬ��3ͨ���ģ�
**       ��ÿ������batch=64��ͼƬ����ô�������64��3ͨ����ͼƬ����mean�е�ÿ��Ԫ�ؾ���ĳ��ͨ��������64��ͼƬ
**       ����Ԫ�ص�ƽ��ֵ�������1��ͨ���ϣ�����64��ͼƬ����ƽ��ֵ��
** ˵��4����ȫ���Ӳ��ǰ�򴫲������У�sptial=1����Ϊȫ���Ӳ��������Կ�����1*1������ͼ
*/
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
	// scale�����ֵ�еķ�ĸ��
	float scale = 1. / (batch * spatial);
	int i, j, k;
	// ���ѭ������Ϊfilters��Ҳ��mean��ά�ȣ�ÿ��ѭ�����õ�һ��ƽ��ֵ
	for (i = 0; i < filters; ++i) {
		mean[i] = 0;
		// �в�ѭ������Ϊbatch��Ҳ������ÿ������ͼƬ��Ӧ��ĳһͨ���ϵ����
		for (j = 0; j < batch; ++j) {
			// �ڲ�ѭ��������һ���������ͼ����������ֵ
			for (k = 0; k < spatial; ++k) {
				// �������������ע�ͣ������ƫ���Ǻ���Ȼ��
				int index = j*filters*spatial + i*spatial + k;
				mean[i] += x[index];
			}
		}
		// ���Ըþ�ֵ���漰Ԫ�ص��ܸ������õ�ƽ��ֵ
		mean[i] *= scale;
	}
}

/*
** ��������x��ÿ��Ԫ�صķ�����µĹ��̺������mean_cpu���ƣ�����׸����
** ����������Ҫ�ô�֮һӦ�þ���batch normalization�ĵڶ����ˣ�
** ���룺
**       x         �����������ݣ�����l.output���������Ԫ�ظ���Ϊl.batch*l.outputs
**       batch     һ��batch�а�����ͼƬ��������l.batch
**       filters   �ò���������˲���������Ҳ���ò��������ͼƬ��ͨ����������Ծ��������˵�����Ǻ˵ĸ����ˣ�
**       spatial   �ò�������ÿ���������ͼ�ĳߴ磬Ҳ������l.out_w*l.out_h
**       mean      ��õ�ƽ��ֵ��ά��Ϊfilters��Ҳ��ÿ���˲�����Ӧ��һ����ֵ��ÿ���˲����ᴦ������ͼƬ��
*/
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
	// Ϊʲô���㷽���ĸҪ��ȥ1�أ��ο�����ɣ�https://www.zhihu.com/question/20983193
	// ��ʵ�ϣ���ͳ��ѧ�У��������õķ�����㹫ʽ�����÷�ĸ��1,��ʱ��Ϊ�������ݵķ����ǻ��ھ�ֵ����̶���������ģ�
	// ������n�����ݵ��������ھ�ֵ�̶�������£���������ɶ�Ϊn-1��ֻҪn-1�����ݹ̶�����n�������ɾ�ֵ�Ƴ���
	float scale = 1. / (batch * spatial - 1);
	int i, j, k;
	for (i = 0; i < filters; ++i) {
		variance[i] = 0;
		for (j = 0; j < batch; ++j) {
			for (k = 0; k < spatial; ++k) {
				int index = j*filters*spatial + i*spatial + k;
				// ÿ��Ԫ�ؼ�ȥ��ֵ��ƽ��
				variance[i] += pow((x[index] - mean[i]), 2);
			}
		}
		variance[i] *= scale;
	}
}

void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
	int b, f, i;
	for (b = 0; b < batch; ++b) {
		for (f = 0; f < filters; ++f) {
			for (i = 0; i < spatial; ++i) {
				int index = b*filters*spatial + f*spatial + i;
				x[index] = (x[index] - mean[f]) / (sqrt(variance[f]) + .000001f);
			}
		}
	}
}


/*
**  ������X�е����ݸ��Ƶ�Y�У�ֵ���ƣ�������ָ�븴�ƣ���֮��X��Y֮�����޹�����
**  ���룺 N       X�а�������ЧԪ�ظ���
**        X       ����ʼ����float����ָ��
**        INCX    ��������������������X�з���INCX�ı�����Ž��г�ʼ����ֵ����
*/
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
	int i;
	for (i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}


/*
** axpy�����Դ�����һ�ֻ������������y= alpha*x + y����������x,yΪʸ����alphaΪʵ��ϵ��
** ���Բο���https://www.youtube.com/watch?v=PQ1Q85JGgZg
** ���룺  N       X�а�������ЧԪ�ظ���
**        ALPHA   ϵ��alpha
**        X       ���������ʸ��X
**        INCX    ��������������������X�з���INCX�ı�����Ų�������
**        Y       ���������ʸ����Ҳ�൱�������
*/
//axpy_cpu(size, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
	int i;
	//ע��������+=��*����ֵ(����),�൱�ڼ�ȥһС���ֵ���
	for (i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
	int i;
	for (i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}


/*
** ���룺 input   һ������ͼƬ���ݣ����������softmax_cpu()ע�ͣ���ͬ��
**       n       һ�����������к��е�Ԫ�ظ���n=l.inputs/l.groups
**       temp    �¶Ȳ���������softmax���¶Ȳ�������������һ��softmax with temperature��Ӧ�û��кܶ��
**       stride  ���
**       output  ��һ������ͼƬ���ݶ�Ӧ�������Ҳ��l.output������һ�������Ӧ��ĳһ���֣�
** ˵����������ʵ�ֵľ��Ǳ�׼��softmax��������Ψһ�е�仯�ľ�������ָ������֮ǰ����ÿ������Ԫ�ؼ�ȥ�˸�������Ԫ���е����ֵ����������ֵ�ȶ��ԣ�
**      ���ڴˣ����Բο����ͣ�http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/��
**      ��ƪ����д�Ĳ��������л��ᵽ��softmax-loss���˴�û��ʵ�֣��˴�ʵ�ֵ�Ҳ���������ᵽ��softmax��������softmax-loss�ֿ�ʵ���ˣ���
*/
void softmax(float *input, int n, float temp, int stride, float *output)
{
	int i;
	float sum = 0;
	// ����ʼ���ֵΪfloat�е���Сֵ-FLT_MAX��������float.h�У�
	float largest = -FLT_MAX;
	// Ѱ�������е����ֵ������ΪʲôҪ�ҳ����ֵ����Ϊ����ֵ�����ϵ��ȶ�����ϸ�����http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
	// ��ƪ����д�Ĳ��������ڽӽ�β����ʱ���ᵽ��ΪʲôҪ��ȥ�����е����ֵ��
	for (i = 0; i < n; ++i) {
		if (input[i*stride] > largest) largest = input[i*stride];
	}
	for (i = 0; i < n; ++i) {
		// �ڽ���ָ������֮�䣬�����沩����˵�����ȼ�ȥ���ֵ����Ȼ�¶Ȳ���ҲҪ����
		float e = exp(input[i*stride] / temp - largest / temp);
		sum += e;                       // ���
		output[i*stride] = e;           // ����ÿһ������Ľ����������Ӧ�������
	}
	// ���һ������һ��ת��Ϊ���ʣ�����softmax������ԭ�͡���������������������output��
	for (i = 0; i < n; ++i) {
		output[i*stride] /= sum;
	}
}


/**
* @brief ������input����softmax����õ����output
* @param input    softmax�������������ݣ���������batch�ģ�����net.input����һ��������
* @param n        һ�����������к��е�Ԫ�ظ���n=l.inputs/l.groups
* @param batch    һ��batch�������е�ͼƬ����������net.batch��
* @param batch_offset    һ������ͼƬ���е�Ԫ�ظ�������ֵ����l.inputs�����Խ���batch_offset��Ŀ����Ҫ�����ò�����input������������Ƭ��λ��
* @param groups   һ������ͼƬ��Ԫ�ر��ֳ��˼��飬ֵΪl.groups����������������ļ�ָ�������δָ������Ĭ��Ϊ1��,���������ʱ��û������ô�ã�
*                 �󲿷ֵ�����ֵ��Ϊ1,Ҳ���൱��û���������
* @param group_offset    ֵ����n����ƫ�ƣ���ÿ������ͼƬԪ������������ƫ�ƣ�
* @param stride  ��ȣ��������������axpy_cpu()�����е�INCX������һ��ע�ⲻͬ�ھ�����е�l.stride�����������ָ����stride�����ÿ������
*                �����г�ȡԪ�أ������ȡ��������Ϊstride����������Ԫ�أ�������������Ԫ�أ�ʵ��û���õ���stride=1ʱ����Ȼ���൱��û�����������
*                �����������ݶ��õ��ˣ����������softmax_layer���У��൱��û�ã���Ϊ��forward_softmax_layer()�У����øú���ʱ��stride�Ѿ�
*                ��д��Ϊ1,�����ܸģ���֪������û�������ط�ʹ�������������
* @param temp     softmax���¶Ȳ���l.temperature������softmax���¶Ȳ�������������һ��softmax with temperature��Ӧ�û��кܶ��
* @param output   ��softmax����֮��õ������l.output�������ʣ�����input������ͬ��Ԫ�ظ�������make_softmax_layer()������ʵ�ɴ�Ҳ��֪��
*                stride��ֵ��ȻΪ1,��Ȼoutput��Ԫ�ظ����϶�����input��Ԫ�ظ��������Զ���softmax��˵���о�����stride��û�б�Ҫ�ģ��е�����ì�ܵ���˼��
* @note ����ע����Ե���softmax_layer�����в�ͬ�ط����ñ��������ڵ��ô�������ϸע�ͣ������ע�ͳ������µ����ʵ�λ����������һ�¹�ϵ������input
*        �а���batch������ͼƬ���������ݣ�����һ��ͼƬ����inputs��Ԫ�أ�һ��ͼƬ��Ԫ���ֳַ���groups�飬ÿ��Ԫ�ظ���Ϊn=l.inputs/l.groups
*/

//softmax_cpu   (net.input + index, l.classes, l.batch*l.n, l.inputs / l.n,   l.w*l.h,    1,                l.w*l.h,    1,          l.output + index);
void softmax_cpu(float *input,      int n,     int batch,   int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
	int g, b;
	// ����batch�е�ÿ��ͼƬ(����ÿ����Ƭ��ϸ��ͬһ�������Ϊһ��)
	for (b = 0; b < batch; ++b) {
		// ÿ��ͼƬ�ְ��������һ��һ�����,һ���൱��
		for (g = 0; g < groups; ++g) {
			//������������������
			//input:λ��ָ�������ά����,Ҳ����C1����
			// n:�����м���,���������е�2,����C1,C2
			// batch:һ��batch����������ﻹϸ�ֵ�ÿһ�������,�������о��� "-##-" �ָ�����һ��
			// batch_offset:һ��batch����,��������l.input,�������ﻹ��������batch��ϸ����,Ҳ����l.input/l.n
			// groups:�����ж��ٸ�����
			// group_offset:ÿ�ε����Ĳ���
			// stride:softmax����һ���������������͵�exp,���ڴ洢������޶�,��Ҫһ����l.w*l.h���ܶ�,Ҳ����һ���������������Ĵ�С
			// temp:�¶Ȳ���,�����������˵����
			// output:������,�������������һ�����ڴ�ṹ
			//xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C1C1C1C1C1C2C2C2C2C2C2-##-xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C2C1C2C1C2C1C2C1C2C1C2
			// ѭ�������ܽ�����˵����:ÿ��batch(ϸ�ֵ������)��ÿһ�����񵥶�����softmax,batch����ͼƬ֮��ĵ���,groups��������֮��ĵ���,
			// n����ͬ��������,ÿһ��ĵ���
			softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
		}
	}
}