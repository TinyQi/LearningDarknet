#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "img2col.h"
#include "gemm.h"
/*
**  根据输入图像的高度(h)，两边补0的个数(pad)，卷积核尺寸(size)以及跨度(stride)计算输出的特征图的高度
**  输入：l    卷积层，包含该卷积层的所有参数，实际这里没有必要输入整个l，因为只需要到其中的四个参数而已
**  输出：int类型，输出图像的高度
**  说明：这个函数的实现应该可以进一步改善一下，虽然这个函数只是在最初构建网络时调用一次，之后就不调用了，不怎么影响性能，
**       但输入整个l实在不妥（l比较大，按值传递复制过程比较冗长），要么就只输入用到的四个参数，要么传入l的指针，
**       并且不需要返回值了，直接在函数内部为l.out_h赋值
*/
int convolutional_out_height(convolutional_layer l)
{
	// pad是每边补0的个数，因此乘以2
	// 当stride=1，pad=size/2（整数除法，会往下取整）时，输出高度就等于输入高度（same策略）；
	// 当stride=1,pad=0时，为valid策略
	// 当stride不等于1时，输出高度恒小于输入高度（尺寸一定会缩小）
	// 计算公式推导：设输出高度为x，总图像高度为h+2*pad个像素，输出高度为x，则共有x-1次卷积核移位，
	// 共占有(x-1)*stride+size个像素，可能还剩余res个像素，且res一定小于stride（否则还可以再移位一次），
	// 因此有(x-1)*stride+size+res=h+2*pad，->x=(h+2*pad-size)/stride+1-res/stride，因为res<stride，
	// 对于整数除法来说，值为0,于是得到最终的输出高度为x=(h+2*pad-size)/stride+1
	return (l.h + 2 * l.pad - l.size) / l.stride + 1;
}

/*
**  根据输入图像的宽度(w)，两边补0的个数(pad)，卷积核尺寸(size)以及跨度(stride)计算输出的特征图的宽度
**  与上一个函数convolutional_out_height()类似，不再赘述
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

	l.type = CONVOLUTIONAL;//层属性

	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.binary;
	l.xnor;
	l.batch = batch;
	l.stride = stride;
	l.batch_normalize = batch_normalize;

	// 该卷积层总的权重元素（卷积核元素）个数=输入图像通道数*卷积核个数*卷积核尺寸
	// （因为一个卷积核要作用在输入图片的所有通道上，所以说是一个卷积核，实际含有的卷积核参数个数需要乘以输入图片的通道数）
	l.weights = calloc(c*n*size*size, sizeof(float));

	l.weight_updates = calloc(c*n*size*size, sizeof(float));

	// bias就是Wx+b中的b（上面的weights就是W），有多少个卷积核，就有多少个b（与W的个数一一对应，每个W的元素个数为c*size*size）
	l.biases = calloc(n, sizeof(float));
	l.bias_updates = calloc(n, sizeof(float));

	//n*c*size*size,卷积核个数 x 卷积核通道 x 卷积核宽度 x 卷积核高度
	l.nweights = c*n*size*size;
	//一个卷积核对应一个偏置
	l.nbiases = n;

	//初始化权重系数
	float scale = sqrt(2. / (size*size*c));
	int i = 0;
	//这里的操作不合理,如果有预训练权重,那这里初始化的参数就会被覆盖,也就是说这里的操作就没意义,应该加一层判断的,如果有预训练权重就不进行下面的初始化环节
	for (i = 0; i < c * n * size * size; ++i) l.weights[i] = scale * rand_normal();


	// 根据该层输入图像的尺寸、卷积核尺寸以及跨度计算输出特征图的宽度和高度
	// 输入尺寸预卷积核尺寸决定输出
	int out_w = convolutional_out_width(l);
	int out_h = convolutional_out_height(l);
	l.out_w = out_w;
	l.out_h = out_h;
	l.out_c = n;

	// 对应每张输入图片的所有输出特征图的总元素个数（每张输入图片会得到n也即l.out_c张特征图）
	l.outputs = l.out_h * l.out_w * l.out_c;    
	// 每张输入图片的像素元素个数
	l.inputs = l.w * l.h * l.c;

	// l.output为该层所有的输出（包括mini-batch所有输入图片的输出）
	l.output = calloc(l.batch*l.outputs, sizeof(float));
	l.delta = calloc(l.batch*l.outputs, sizeof(float));

	// 卷积层三种指针函数，对应三种计算：前向，反向，更新
	// 类似于C++,子类实现父类的虚函数
	l.forward = forward_convolutional_layer;
	l.backward = backward_convolutional_layer;
	l.update = update_convolutional_layer;


	//TODO:以下这些参数干嘛用的,还不清楚
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
	
	//初始化当前卷积层的所有输出元素为0
	fill_cpu(l.outputs, 0, l.output, 1);
	
	// 是否进行二值化操作（这个操作应该只有第一个卷积层使用吧？因为下面直接对net.input操作，这个理解是错误的，因为在forward_network()含中，
	// 每进行一层都会将net.input = l.output，即下一层的输入被设置为当前层的输出）
	if (l.xnor) {
		//todo:作用未知
	/*	binarize_weights(l.weights, l.n, l.c*l.size*l.size, l.binary_weights);
		swap_binary(&l);
		binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
		net.input = l.binary_input;*/
	}
	
	int m = l.n;							//卷积核数量,也就是输出通道数,A的行数(rows),C的行数(rows)
	int k = l.size*l.size*l.c;				//每个卷积核有多少个元素,A的列数(cols),B的行数(rows)
	int n = out_h * out_w;					//该层每个特征图的元素个数,一张图会有m(输出通道)个特征图,B的列数(cols),C的列数(cols)

	float *a = l.weights;					//所有卷积核(就是权重文件)
	float *b = net.workspace;				//在这个位置是表示对输入图像重新排列后的图像数据,(按卷积规则排成一行)
	float *c = l.output;					//存储所有图片的所有特征图

	float alpha = 1;
	float beta = 1;

	//每次循环卷积一个batch中的一张图
	//C=A*B,A是卷积核,B是输入图,C是特征图
	for (i = 0; i < l.batch; i++)
	{
		//先按卷积规则把输入图片扩一下容,然后再按行存储,加速后续卷积运算
		img2col(net.input, b, l.w, l.h, l.c, l.size, l.stride, l.pad);

		//进行卷积运算,填参数的时候,只需要认准a,b,c各自的行与列,以及转置后的行与列即可
		gemm(0, 0, m, n, k, alpha, a, k, b, n, beta, c, n);

		//指针指向下一张图,循环卷积整个batch的图
		c += (n*m);
		//net.input += l.inputs;//其实用这个也挺好的
		net.input += l.w * l.h * l.c;
	}

	//todo:作用未知
	// 如需要规范化（BN在非线性激活函数处理之前完成）
	if (l.batch_normalize) {
		forward_batchnorm_layer(l, net);
	}
	else {
		add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
	}

	//利用激活函数对每个输出元素再计算以便
	activate_array(l.output, m*n*l.batch, l.activation);
	
	//todo:作用未知
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
	
	//在这一步进行权重衰变,把权重衰减项(-decay * batch * weights)叠加到权重更新值上面
	axpy_cpu(size, -decay*batch, l.weights, 1, l.weight_updates, 1);
	//y = alpha * x + y
	//l.weights=l.weights - learning_rate/batch * l.weight_updates(导数)
	//完成权重的更新
	axpy_cpu(size, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(size, momentum, l.weight_updates, 1);


	//关于L2正则化(权重衰减)
	//权重衰减就是L2正则化,防止权重过大,导致模型过拟合
	// 问题:为什么权重过大会导致过拟合?
	// 1.权重小,意味着复杂度低,复杂度低意味着泛化能力好,所以权重大就泛化不好,就过拟合
	// 2. 从数学的角度来讲,过拟合的时候,曲线都是很扭曲剧烈的,而能支持在很短的区间产生剧烈震荡的曲线,其系数毕然很大,也就是过拟合的特征
}

/*
** 将以a为首地址此后n个元素相加，返回总和
*/
float sum_array(float *a, int n)
{
	int i;
	float sum = 0;
	for (i = 0; i < n; ++i) sum += a[i];
	return sum;
}

/*
** 计算每个卷积核的偏置更新值，所谓偏置更新值，就是bias = bias - alpha * bias_update中的bias_update
** 输入： bias_updates     当前层所有偏置的更新值，维度为l.n（即当前层卷积核的个数）
**       delta            当前层的敏感度图（即l.delta）
**       batch            一个batch含有的图片张数（即l.batch）
**       n                当前层卷积核个数（即l.h）
**       k                当前层输入特征图尺寸（即l.out_w*l.out_h）
** 原理：当前层的敏感度图l.delta是误差函数对加权输入的导数，也就是偏置更新值，只是其中每l.out_w*l.out_h个元素都对应同一个
**      偏置，因此需要将其加起来，得到的和就是误差函数对当前层各偏置的导数（l.delta的维度为l.batch*l.n*l.out_h*l.out_w,
**      可理解成共有l.batch行，每行有l.n*l.out_h*l.out_w列，而这一大行又可以理解成有l.n，l.out_h*l.out_w列，这每一小行就
**      对应同一个卷积核也即同一个偏置）
*/
void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
	int i, b;
	// 遍历batch中每张输入图片
	// 注意，最后的偏置更新值是所有输入图片的总和（多张图片无非就是重复一张图片的操作，求和即可）。
	// 总之：一个卷积核对应一个偏置更新值，该偏置更新值等于batch中所有输入图片累积的偏置更新值，
	// 而每张图片也需要进行偏置更新值求和（因为每个卷积核在每张图片多个位置做了卷积运算，这都对偏置更新值有贡献）以得到每张图片的总偏置更新值。
	for (b = 0; b < batch; ++b) {
		// 求和得一张输入图片的总偏置更新值
		for (i = 0; i < n; ++i) {
			bias_updates[i] += sum_array(delta + size*(i + b*n), size);
		}
	}
}


/*
归纳重点
1. 链式法则传递的是残差,也就是框架里的l.delta,它是由损失根据链式法则对每一层的输入的求导得来,
   重点在于损失对输入求导(在卷积层实质就是:残差x权重(注意:这个权重是还未更新的权重),因为这时候变量是输入)
2. 而每一层的权重更新值又是根据传到当前层的残差对权重的偏导乘以学习率得到的(偏置更新值同理),
   重点在于残差对权重求导(在卷积层实质就是:残差x输入,因为这时候变量是权重)
3. 由于卷积层最后还有激活函数,所以下一层传回的残差,还得根据激活函数对输入求导,才是真实的卷积运算的残差,才可以去求weights和bias的偏导
4. 根据残差计算完权重更新值和偏置更新值后,需要继续将残差往前传(这就是反向传播的核心思想),
   就是需要把残差对输入求导,也就是对乘以当前权重(未更新),同理上一层接受到后也会先考虑激活函数的求导
*/
void backward_convolutional_layer(convolutional_layer l, network net)
{
	int i = 0;
	int m = l.n;	//卷积核个数

	//卷积核的元素个数(一个卷积核可以理解为有多个通道的图片,n在这里就是这个图片有多少个像素)
	int n = l.size * l.size * l.c;
	//每张输出特征图的元素个数
	int k = l.out_w * l.out_h;

	// 下一层传回的残差 乘以 当前层的输出根据激活函数对输入的导数
	// 这样之后,残差才能经过激活函数传到卷积处
	//  m * k * l.batch: 残差总共有多少个元素(有多少个卷积核就有多少张特征图)
	gradient_array(l.output, m * k * l.batch, l.activation, l.delta);
	
	if (l.batch_normalize)
	{
		backward_batchnorm_layer(l, net);
	}
	else
	{
		backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
	}

	// 1. 计算当前层的权重更新值(残差 × 卷积运算的输入)
	// 2. 将当前层的残差,传过当前层的卷积运算(残差 × 权重(未更新的))
	for (i = 0; i < l.batch; i++)
	{
		//残差,根据循环,计算每个batch里的残差
		float *a = l.delta + i * m * k;

		float *b = net.workspace;
		float *c = l.weight_updates;

		//当前层卷积运算的输入,将残差传导经过卷积层,就是用残差 ? 输入
		float *im = net.input + i * l.c * l.h * l.w;

		//将输入,重排成列,利用矩阵运算,加速 "残差 ? 输入"的运算
		img2col(im, b, l.w, l.h, l.c, l.size, l.stride, l.pad);

		//执行 "残差 ? 输入"的运算
		//a: (l.out_c) * (l.out_h*l*out_w),可以forward函数里推出
		//b: (l.c * l.size * l.size) * (l.out_h * l.out_w)
		//需要转置b ,才能完成 "残差 ? 输入"的运算
		// 
		//!!!注意:在这里残差?输入并不是卷积运算,而是每一个残差元素对所有前向计算时的"权重输入相乘项"的权重求偏导
		// 比如一个4*4的输入图, kernel是2*2, 前向传播的输出图是 3*3
		// 那么反向的时候残差图也是3*3的,我们就以残差图中的左上角(0,0)这个元素举例.
		// (0,0)这个元素在前向计算的时候,展开卷积公式的话,就是
		// --------------------下面画图表达----------------------
		// a,b,c,d					
		// e,f,g,h      w1,w2		o1,o2,o3		delta1,delta2,delta3
		// i,j,k,l		w3,w4		o4,o5,o6		delta4,delta5,delta6
		// m,n,o,p					o7,o8,o9		delta7,delta8,delta9
		// 输入			kernel		输出			残差
		// 
		// 最终的卷积运算细分为:
		// a*w1 + b*w2 + e*w3 + f*w4 = o1 (1)
		// b*w1 + c*w2 + f*w3 + g*w4 = o2 (2)
		// c*w1 + d*w2 + g*w3 + h*w4 = o3 (3)
		// e*w1 + f*w2 + i*w3 + j*w4 = o4 (4)
		// ......
		// 依次类推
		// 所以前向传播的时候,与w1相关的项有9项,然后反向传播的时候,所有的残差项对权重w1求导,也会有9项
		// 当前层的残差对w1偏导的 = delta1 * a + delta2 * b + ..... + delta8 * j + delta9 * k  ?
		// 其他3个权重也是同理
		// 
		// 下面再解释矩阵运算实现如何求残差对权重的偏导
		// 输入图根据卷积的规则,重拍成下面的形状
		// a,b,c,e,f....j,k
		// b,c,d,f,g....k,l
		// e,f,g,i,j....n,o  (形状 4 * 9)
		// f,g,h,j,k....o,p
		// 
		// 残差图是以下形状存储的
		// delta1,delta2,delta3,delta4,delta5,delta6,delta7,delta8,delta9   (形状1 * 9)
		// 矩阵相乘行列有规矩,再结合求偏导的实际公式(看上面?行),就需要把输入图的重拍图转置后再与残差图进行矩阵乘法
		// 最终 结果矩阵形状 (1 * 4) ,对应该层求出来的4个权重的偏导
		// 注意:以上例子没有考虑通道,考虑通道无非就是输入图的重排矩阵追加几行,然后残差图还是(1*9),当然如果多个卷积核又是另外一种情况
		// 比如上面例子双通道的情况下,a(残差图):(1 * 9),b'(输入图):(8 * 9), c = a * b' ,c的形状是(1*8),对应这个2通道的卷积核的8个权重的偏导
		gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

		//执行 "残差 ? 权重(未更新)"的运算,使得残差传递过卷积运算
		if (net.delta)
		{
			a = l.weights;

			b = l.delta + i * m * k;

			c = net.workspace;

			//a: (l.n) * (l.c * l.size * l.size) 等价于 m * n
			//b: (l.out_c) * (l.out_h*l*out_w),等价于 m * k ,l.out_c = l.n
			//需要转置a,才能完成 "残差 ? 权重(未更新)"的运算
			//还是举上面的例子来解释残差 ? 权重 是如何实现的
			//从上面的卷积运算细分可以看出,卷积运算一共有9个卷积细分公式,每个公式4个相乘项,所以残差对输入求怕偏导会有36项
			//所以实际上求输入的偏导的过程矩阵化就是: 
			// a'(权重,a是1 * 4,为了矩阵计算,将a进行转置) * b(残差, 1 * 9) = c(反向传递过卷积层的新的残差图,4 * 9)
			//但是36项是基于卷积运算矩阵?的细分规则分的,所以下面还需要把新计算出来的残差图c,
			// 从矩阵重排的形式重排成图像(重复计算是为了节省空间,用时间换空间)
			gemm(1, 0, n, k, m, 1, a, n, b, k, 1, c, k);

			//输入图从矩阵重排的形式变回图像的排列方式
			col2im_cpu(net.workspace, l.c, l.h, l.w, l.size, l.stride, l.pad, net.delta + i*l.c*l.h*l.w);
		}
	}
}