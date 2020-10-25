#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
	typedef layer convolutional_layer;

	/*
	**
	**
	**  输入：batch    每个batch含有的图片数
	**      h               图片高度（行数）
	**      w               图片宽度（列数）
	c               输入图片通道数
	n               卷积核个数
	size            卷积核尺寸
	stride          跨度
	padding         四周补0长度
	activation      激活函数类别
	batch_normalize 是否进行BN(规范化)
	binary          是否对权重进行二值化
	xnor            是否对权重以及输入进行二值化
	adam            使用adam优化方式
	*/
	convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);


	void forward_convolutional_layer(const convolutional_layer l, network net);

	void backward_convolutional_layer(convolutional_layer l, network net);

	void update_convolutional_layer(convolutional_layer l, int batch, float learning_rate, float momentum, float decay);
#ifdef __cplusplus
}
#endif