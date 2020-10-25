#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "img2col.h"
#include "gemm.h"
/*
**  ��������ͼ��ĸ߶�(h)�����߲�0�ĸ���(pad)������˳ߴ�(size)�Լ����(stride)�������������ͼ�ĸ߶�
**  ���룺l    ����㣬�����þ��������в�����ʵ������û�б�Ҫ��������l����Ϊֻ��Ҫ�����е��ĸ���������
**  �����int���ͣ����ͼ��ĸ߶�
**  ˵�������������ʵ��Ӧ�ÿ��Խ�һ������һ�£���Ȼ�������ֻ���������������ʱ����һ�Σ�֮��Ͳ������ˣ�����ôӰ�����ܣ�
**       ����������lʵ�ڲ��ף�l�Ƚϴ󣬰�ֵ���ݸ��ƹ��̱Ƚ��߳�����Ҫô��ֻ�����õ����ĸ�������Ҫô����l��ָ�룬
**       ���Ҳ���Ҫ����ֵ�ˣ�ֱ���ں����ڲ�Ϊl.out_h��ֵ
*/
int convolutional_out_height(convolutional_layer l)
{
	// pad��ÿ�߲�0�ĸ�������˳���2
	// ��stride=1��pad=size/2������������������ȡ����ʱ������߶Ⱦ͵�������߶ȣ�same���ԣ���
	// ��stride=1,pad=0ʱ��Ϊvalid����
	// ��stride������1ʱ������߶Ⱥ�С������߶ȣ��ߴ�һ������С��
	// ���㹫ʽ�Ƶ���������߶�Ϊx����ͼ��߶�Ϊh+2*pad�����أ�����߶�Ϊx������x-1�ξ������λ��
	// ��ռ��(x-1)*stride+size�����أ����ܻ�ʣ��res�����أ���resһ��С��stride�����򻹿�������λһ�Σ���
	// �����(x-1)*stride+size+res=h+2*pad��->x=(h+2*pad-size)/stride+1-res/stride����Ϊres<stride��
	// ��������������˵��ֵΪ0,���ǵõ����յ�����߶�Ϊx=(h+2*pad-size)/stride+1
	return (l.h + 2 * l.pad - l.size) / l.stride + 1;
}

/*
**  ��������ͼ��Ŀ��(w)�����߲�0�ĸ���(pad)������˳ߴ�(size)�Լ����(stride)�������������ͼ�Ŀ��
**  ����һ������convolutional_out_height()���ƣ�����׸��
*/
int convolutional_out_width(convolutional_layer l)
{
	return (l.w + 2 * l.pad - l.size) / l.stride + 1;
}


static size_t get_workspace_size(layer l)
{
	return (size_t)l.out_h*l.out_w*l.size*l.size*l.c * sizeof(float);
}

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
{
	convolutional_layer l = { 0 };

	l.type = CONVOLUTIONAL;//������

	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.binary;
	l.xnor;
	l.batch = batch;
	l.stride = stride;
	l.batch_normalize = batch_normalize;

	// �þ�����ܵ�Ȩ��Ԫ�أ������Ԫ�أ�����=����ͼ��ͨ����*����˸���*����˳ߴ�
	// ����Ϊһ�������Ҫ����������ͼƬ������ͨ���ϣ�����˵��һ������ˣ�ʵ�ʺ��еľ���˲���������Ҫ��������ͼƬ��ͨ������
	l.weights = calloc(c*n*size*size, sizeof(float));

	l.weight_updates = calloc(c*n*size*size, sizeof(float));

	// bias����Wx+b�е�b�������weights����W�����ж��ٸ�����ˣ����ж��ٸ�b����W�ĸ���һһ��Ӧ��ÿ��W��Ԫ�ظ���Ϊc*size*size��
	l.biases = calloc(n, sizeof(float));
	l.bias_updates = calloc(n, sizeof(float));

	//n*c*size*size,����˸��� x �����ͨ�� x ����˿�� x ����˸߶�
	l.nweights = c*n*size*size;
	//һ������˶�Ӧһ��ƫ��
	l.nbiases = n;

	//��ʼ��Ȩ��ϵ��
	float scale = sqrt(2. / (size*size*c));
	int i = 0;
	//����Ĳ���������,�����Ԥѵ��Ȩ��,�������ʼ���Ĳ����ͻᱻ����,Ҳ����˵����Ĳ�����û����,Ӧ�ü�һ���жϵ�,�����Ԥѵ��Ȩ�ؾͲ���������ĳ�ʼ������
	for (i = 0; i < c * n * size * size; ++i) l.weights[i] = scale * rand_normal();


	// ���ݸò�����ͼ��ĳߴ硢����˳ߴ��Լ���ȼ����������ͼ�Ŀ�Ⱥ͸߶�
	// ����ߴ�Ԥ����˳ߴ�������
	int out_w = convolutional_out_width(l);
	int out_h = convolutional_out_height(l);
	l.out_w = out_w;
	l.out_h = out_h;
	l.out_c = n;

	// ��Ӧÿ������ͼƬ�������������ͼ����Ԫ�ظ�����ÿ������ͼƬ��õ�nҲ��l.out_c������ͼ��
	l.outputs = l.out_h * l.out_w * l.out_c;    
	// ÿ������ͼƬ������Ԫ�ظ���
	l.inputs = l.w * l.h * l.c;

	// l.outputΪ�ò����е����������mini-batch��������ͼƬ�������
	l.output = calloc(l.batch*l.outputs, sizeof(float));
	l.delta = calloc(l.batch*l.outputs, sizeof(float));

	// ���������ָ�뺯������Ӧ���ּ��㣺ǰ�򣬷��򣬸���
	// ������C++,����ʵ�ָ�����麯��
	l.forward = forward_convolutional_layer;
	l.backward = backward_convolutional_layer;
	l.update = update_convolutional_layer;


	//TODO:������Щ���������õ�,�������
	//--------------------------------------------------
	if (binary) {
		l.binary_weights = calloc(c*n*size*size, sizeof(float));
		l.cweights = calloc(c*n*size*size, sizeof(char));
		l.scales = calloc(n, sizeof(float));
	}
	if (xnor) {
		l.binary_weights = calloc(c*n*size*size, sizeof(float));
		l.binary_input = calloc(l.inputs*l.batch, sizeof(float));
	}

	if (batch_normalize) {
		l.scales = calloc(n, sizeof(float));
		l.scale_updates = calloc(n, sizeof(float));
		for (i = 0; i < n; ++i) {
			l.scales[i] = 1;
		}

		l.mean = calloc(n, sizeof(float));
		l.variance = calloc(n, sizeof(float));

		l.mean_delta = calloc(n, sizeof(float));
		l.variance_delta = calloc(n, sizeof(float));

		l.rolling_mean = calloc(n, sizeof(float));
		l.rolling_variance = calloc(n, sizeof(float));
		l.x = calloc(l.batch*l.outputs, sizeof(float));
		l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
	}
	if (adam) {
		l.adam = 1;
		l.m = calloc(c*n*size*size, sizeof(float));
		l.v = calloc(c*n*size*size, sizeof(float));
		l.bias_m = calloc(n, sizeof(float));
		l.scale_m = calloc(n, sizeof(float));
		l.bias_v = calloc(n, sizeof(float));
		l.scale_v = calloc(n, sizeof(float));
	}
	//--------------------------------------------------
	l.workspace_size = get_workspace_size(l);
	l.activation = activation;

	fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

	return l;
}
//todo
void add_bias(float *output, float *biases, int batch, int n, int size)
{
	int i, j, b;
	for (b = 0; b < batch; ++b) {
		for (i = 0; i < n; ++i) {
			for (j = 0; j < size; ++j) {
				output[(b*n + i)*size + j] += biases[i];
			}
		}
	}
}
//todo
void scale_bias(float *output, float *scales, int batch, int n, int size)
{
	int i, j, b;
	for (b = 0; b < batch; ++b) {
		for (i = 0; i < n; ++i) {
			for (j = 0; j < size; ++j) {
				output[(b*n + i)*size + j] *= scales[i];
			}
		}
	}
}
void forward_convolutional_layer(const convolutional_layer l, network net)
{
	int out_h = l.out_h;
	int out_w = l.out_w;

	int i = 0;
	
	//��ʼ����ǰ�������������Ԫ��Ϊ0
	fill_cpu(l.outputs, 0, l.output, 1);
	
	// �Ƿ���ж�ֵ���������������Ӧ��ֻ�е�һ�������ʹ�ðɣ���Ϊ����ֱ�Ӷ�net.input�������������Ǵ���ģ���Ϊ��forward_network()���У�
	// ÿ����һ�㶼�Ὣnet.input = l.output������һ������뱻����Ϊ��ǰ��������
	if (l.xnor) {
		//todo:����δ֪
	/*	binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
		swap_binary(&l);
		binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
		net.input = l.binary_input;*/
	}
	
	int m = l.n;							//���������,Ҳ�������ͨ����,A������(rows),C������(rows)
	int k = l.size*l.size*l.c;				//ÿ��������ж��ٸ�Ԫ��,A������(cols),B������(rows)
	int n = out_h * out_w;					//�ò�ÿ������ͼ��Ԫ�ظ���,һ��ͼ����m(���ͨ��)������ͼ,B������(cols),C������(cols)

	float *a = l.weights;					//���о����(����Ȩ���ļ�)
	float *b = net.workspace;				//�����λ���Ǳ�ʾ������ͼ���������к��ͼ������,(����������ų�һ��)
	float *c = l.output;					//�洢����ͼƬ����������ͼ

	float alpha = 1;
	float beta = 1;

	//ÿ��ѭ�����һ��batch�е�һ��ͼ
	//C=A*B,A�Ǿ����,B������ͼ,C������ͼ
	for (i = 0; i < l.batch; i++)
	{
		//�Ȱ�������������ͼƬ��һ����,Ȼ���ٰ��д洢,���ٺ����������
		img2col(net.input, b, l.w, l.h, l.c, l.size, l.stride, l.pad);

		//���о������,�������ʱ��,ֻ��Ҫ��׼a,b,c���Ե�������,�Լ�ת�ú�������м���
		gemm(0, 0, m, n, k, alpha, a, k, b, n, beta, c, n);

		//ָ��ָ����һ��ͼ,ѭ���������batch��ͼ
		c += (n*m);
		//net.input += l.inputs;//��ʵ�����Ҳͦ�õ�
		net.input += l.w * l.h * l.c;
	}

	//todo:����δ֪
	// ����Ҫ�淶����BN�ڷ����Լ��������֮ǰ��ɣ�
	if (l.batch_normalize) {
		forward_batchnorm_layer(l, net);
	}
	else {
		add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
	}

	//���ü������ÿ�����Ԫ���ټ����Ա�
	activate_array(l.output, m*n*l.batch, l.activation);
	
	//todo:����δ֪
	//if (l.binary || l.xnor) swap_binary(&l);
}

void update_convolutional_layer(convolutional_layer l, int batch, float learning_rate, float momentum, float decay)
{
	int size = l.size*l.size*l.c*l.n;
	axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.n, momentum, l.bias_updates, 1);

	if (l.scales) {
		axpy_cpu(l.n, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
		scal_cpu(l.n, momentum, l.scale_updates, 1);
	}
	
	//����һ������Ȩ��˥��,��Ȩ��˥����(-decay * batch * weights)���ӵ�Ȩ�ظ���ֵ����
	axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
	//y = alpha * x + y
	//l.weights=l.weights - learning_rate/batch * l.weight_updates(����)
	//���Ȩ�صĸ���
	axpy_cpu(size, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(size, momentum, l.weight_updates, 1);


	//����L2����(Ȩ��˥��)
	//Ȩ��˥������L2����,��ֹȨ�ع���,����ģ�͹����
	// ����:ΪʲôȨ�ع���ᵼ�¹����?
	// 1.Ȩ��С,��ζ�Ÿ��Ӷȵ�,���Ӷȵ���ζ�ŷ���������,����Ȩ�ش�ͷ�������,�͹����
	// 2. ����ѧ�ĽǶ�����,����ϵ�ʱ��,���߶��Ǻ�Ť�����ҵ�,����֧���ں̵ܶ�������������𵴵�����,��ϵ����Ȼ�ܴ�,Ҳ���ǹ���ϵ�����
}

/*
** ����aΪ�׵�ַ�˺�n��Ԫ����ӣ������ܺ�
*/
float sum_array(float *a, int n)
{
	int i;
	float sum = 0;
	for (i = 0; i < n; ++i) sum += a[i];
	return sum;
}

/*
** ����ÿ������˵�ƫ�ø���ֵ����νƫ�ø���ֵ������bias = bias - alpha * bias_update�е�bias_update
** ���룺 bias_updates     ��ǰ������ƫ�õĸ���ֵ��ά��Ϊl.n������ǰ�����˵ĸ�����
**       delta            ��ǰ������ж�ͼ����l.delta��
**       batch            һ��batch���е�ͼƬ��������l.batch��
**       n                ��ǰ�����˸�������l.h��
**       k                ��ǰ����������ͼ�ߴ磨��l.out_w*l.out_h��
** ԭ����ǰ������ж�ͼl.delta�������Լ�Ȩ����ĵ�����Ҳ����ƫ�ø���ֵ��ֻ������ÿl.out_w*l.out_h��Ԫ�ض���Ӧͬһ��
**      ƫ�ã������Ҫ������������õ��ĺ;��������Ե�ǰ���ƫ�õĵ�����l.delta��ά��Ϊl.batch*l.n*l.out_h*l.out_w,
**      �����ɹ���l.batch�У�ÿ����l.n*l.out_h*l.out_w�У�����һ�����ֿ���������l.n��l.out_h*l.out_w�У���ÿһС�о�
**      ��Ӧͬһ�������Ҳ��ͬһ��ƫ�ã�
*/
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
	int i, b;
	// ����batch��ÿ������ͼƬ
	// ע�⣬����ƫ�ø���ֵ����������ͼƬ���ܺͣ�����ͼƬ�޷Ǿ����ظ�һ��ͼƬ�Ĳ�������ͼ��ɣ���
	// ��֮��һ������˶�Ӧһ��ƫ�ø���ֵ����ƫ�ø���ֵ����batch����������ͼƬ�ۻ���ƫ�ø���ֵ��
	// ��ÿ��ͼƬҲ��Ҫ����ƫ�ø���ֵ��ͣ���Ϊÿ���������ÿ��ͼƬ���λ�����˾�����㣬�ⶼ��ƫ�ø���ֵ�й��ף��Եõ�ÿ��ͼƬ����ƫ�ø���ֵ��
	for (b = 0; b < batch; ++b) {
		// ��͵�һ������ͼƬ����ƫ�ø���ֵ
		for (i = 0; i < n; ++i) {
			bias_updates[i] += sum_array(delta + size*(i + b*n), size);
		}
	}
}


/*
�����ص�
1. ��ʽ���򴫵ݵ��ǲв�,Ҳ���ǿ�����l.delta,��������ʧ������ʽ�����ÿһ���������󵼵���,
   �ص�������ʧ��������(�ھ����ʵ�ʾ���:�в�xȨ��(ע��:���Ȩ���ǻ�δ���µ�Ȩ��),��Ϊ��ʱ�����������)
2. ��ÿһ���Ȩ�ظ���ֵ���Ǹ��ݴ�����ǰ��Ĳв��Ȩ�ص�ƫ������ѧϰ�ʵõ���(ƫ�ø���ֵͬ��),
   �ص����ڲв��Ȩ����(�ھ����ʵ�ʾ���:�в�x����,��Ϊ��ʱ�������Ȩ��)
3. ���ھ��������м����,������һ�㴫�صĲв�,���ø��ݼ������������,������ʵ�ľ������Ĳв�,�ſ���ȥ��weights��bias��ƫ��
4. ���ݲв������Ȩ�ظ���ֵ��ƫ�ø���ֵ��,��Ҫ�������в���ǰ��(����Ƿ��򴫲��ĺ���˼��),
   ������Ҫ�Ѳв��������,Ҳ���ǶԳ��Ե�ǰȨ��(δ����),ͬ����һ����ܵ���Ҳ���ȿ��Ǽ��������
*/
void backward_convolutional_layer(convolutional_layer l, network net)
{
	int i = 0;
	int m = l.n;	//����˸���

	//����˵�Ԫ�ظ���(һ������˿������Ϊ�ж��ͨ����ͼƬ,n������������ͼƬ�ж��ٸ�����)
	int n = l.size * l.size * l.c;
	//ÿ���������ͼ��Ԫ�ظ���
	int k = l.out_w * l.out_h;

	// ��һ�㴫�صĲв� ���� ��ǰ���������ݼ����������ĵ���
	// ����֮��,�в���ܾ�����������������
	//  m * k * l.batch: �в��ܹ��ж��ٸ�Ԫ��(�ж��ٸ�����˾��ж���������ͼ)
	gradient_array(l.output, m * k * l.batch, l.activation, l.delta);
	
	if (l.batch_normalize)
	{
		backward_batchnorm_layer(l, net);
	}
	else
	{
		backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
	}

	// 1. ���㵱ǰ���Ȩ�ظ���ֵ(�в� �� ������������)
	// 2. ����ǰ��Ĳв�,������ǰ��ľ������(�в� �� Ȩ��(δ���µ�))
	for (i = 0; i < l.batch; i++)
	{
		//�в�,����ѭ��,����ÿ��batch��Ĳв�
		float *a = l.delta + i * m * k;

		float *b = net.workspace;
		float *c = l.weight_updates;

		//��ǰ�������������,���в�����������,�����òв� ? ����
		float *im = net.input + i * l.c * l.h * l.w;

		//������,���ų���,���þ�������,���� "�в� ? ����"������
		img2col(im, b, l.w, l.h, l.c, l.size, l.stride, l.pad);

		//ִ�� "�в� ? ����"������
		//a: (l.out_c) * (l.out_h*l*out_w),����forward�������Ƴ�
		//b: (l.c * l.size * l.size) * (l.out_h * l.out_w)
		//��Ҫת��b ,������� "�в� ? ����"������
		// 
		//!!!ע��:������в�?���벢���Ǿ������,����ÿһ���в�Ԫ�ض�����ǰ�����ʱ��"Ȩ�����������"��Ȩ����ƫ��
		// ����һ��4*4������ͼ, kernel��2*2, ǰ�򴫲������ͼ�� 3*3
		// ��ô�����ʱ��в�ͼҲ��3*3��,���Ǿ��Բв�ͼ�е����Ͻ�(0,0)���Ԫ�ؾ���.
		// (0,0)���Ԫ����ǰ������ʱ��,չ�������ʽ�Ļ�,����
		// --------------------���滭ͼ���----------------------
		// a,b,c,d					
		// e,f,g,h      w1,w2		o1,o2,o3		delta1,delta2,delta3
		// i,j,k,l		w3,w4		o4,o5,o6		delta4,delta5,delta6
		// m,n,o,p					o7,o8,o9		delta7,delta8,delta9
		// ����			kernel		���			�в�
		// 
		// ���յľ������ϸ��Ϊ:
		// a*w1 + b*w2 + e*w3 + f*w4 = o1 (1)
		// b*w1 + c*w2 + f*w3 + g*w4 = o2 (2)
		// c*w1 + d*w2 + g*w3 + h*w4 = o3 (3)
		// e*w1 + f*w2 + i*w3 + j*w4 = o4 (4)
		// ......
		// ��������
		// ����ǰ�򴫲���ʱ��,��w1��ص�����9��,Ȼ���򴫲���ʱ��,���еĲв����Ȩ��w1��,Ҳ����9��
		// ��ǰ��Ĳв��w1ƫ���� = delta1 * a + delta2 * b + ..... + delta8 * j + delta9 * k  ?
		// ����3��Ȩ��Ҳ��ͬ��
		// 
		// �����ٽ��;�������ʵ�������в��Ȩ�ص�ƫ��
		// ����ͼ���ݾ���Ĺ���,���ĳ��������״
		// a,b,c,e,f....j,k
		// b,c,d,f,g....k,l
		// e,f,g,i,j....n,o  (��״ 4 * 9)
		// f,g,h,j,k....o,p
		// 
		// �в�ͼ��������״�洢��
		// delta1,delta2,delta3,delta4,delta5,delta6,delta7,delta8,delta9   (��״1 * 9)
		// ������������й��,�ٽ����ƫ����ʵ�ʹ�ʽ(������?��),����Ҫ������ͼ������ͼת�ú�����в�ͼ���о���˷�
		// ���� ���������״ (1 * 4) ,��Ӧ�ò��������4��Ȩ�ص�ƫ��
		// ע��:��������û�п���ͨ��,����ͨ���޷Ǿ�������ͼ�����ž���׷�Ӽ���,Ȼ��в�ͼ����(1*9),��Ȼ�������������������һ�����
		// ������������˫ͨ���������,a(�в�ͼ):(1 * 9),b'(����ͼ):(8 * 9), c = a * b' ,c����״��(1*8),��Ӧ���2ͨ���ľ���˵�8��Ȩ�ص�ƫ��
		gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

		//ִ�� "�в� ? Ȩ��(δ����)"������,ʹ�òв�ݹ��������
		if (net.delta)
		{
			a = l.weights;

			b = l.delta + i * m * k;

			c = net.workspace;

			//a: (l.n) * (l.c * l.size * l.size) �ȼ��� m * n
			//b: (l.out_c) * (l.out_h*l*out_w),�ȼ��� m * k ,l.out_c = l.n
			//��Ҫת��a,������� "�в� ? Ȩ��(δ����)"������
			//���Ǿ���������������Ͳв� ? Ȩ�� �����ʵ�ֵ�
			//������ľ������ϸ�ֿ��Կ���,�������һ����9�����ϸ�ֹ�ʽ,ÿ����ʽ4�������,���Բв����������ƫ������36��
			//����ʵ�����������ƫ���Ĺ��̾��󻯾���: 
			// a'(Ȩ��,a��1 * 4,Ϊ�˾������,��a����ת��) * b(�в�, 1 * 9) = c(���򴫵ݹ��������µĲв�ͼ,4 * 9)
			//����36���ǻ��ھ���������?��ϸ�ֹ���ֵ�,�������滹��Ҫ���¼�������Ĳв�ͼc,
			// �Ӿ������ŵ���ʽ���ų�ͼ��(�ظ�������Ϊ�˽�ʡ�ռ�,��ʱ�任�ռ�)
			gemm(1, 0, n, k, m, 1, a, n, b, k, 1, c, k);

			//����ͼ�Ӿ������ŵ���ʽ���ͼ������з�ʽ
			col2im_cpu(net.workspace, l.c, l.h, l.w, l.size, l.stride, l.pad, net.delta + i*l.c*l.h*l.w);
		}
	}
}