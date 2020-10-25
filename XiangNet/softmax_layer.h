#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
	typedef layer softmax_layer;

	/*
	** 在darknet中，softmax_layer一般作为网络的倒数第二层（因为darknet中把cost也算作一层，
	** 一般cost_layer作为最后一层），可以参见vgg以及alexnet的网络配置文件（vgg-16.cfg,alexnet.cfg）。
	** softmax_layer本身也没有训练参数，所以比较简单，只是darknet中的实现似乎引入了一些不太常见的东西，导致有些地方理解上比较费劲。
	** softmax_layer构建函数
	** 输入： batch
	**       intputs
	**       groups
	** 注意：此处的softmax_layer层，单张图片的输入元素个数l.inputs等于输出元素个数l.outputs（总输入元素与总输出元素个数也将相同），
	*/
	softmax_layer make_softmax_layer(int batch, int inputs, int groups);

#ifdef __cplusplus
}
#endif