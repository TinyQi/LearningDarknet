#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
	typedef layer softmax_layer;

	/*
	** ��darknet�У�softmax_layerһ����Ϊ����ĵ����ڶ��㣨��Ϊdarknet�а�costҲ����һ�㣬
	** һ��cost_layer��Ϊ���һ�㣩�����Բμ�vgg�Լ�alexnet�����������ļ���vgg-16.cfg,alexnet.cfg����
	** softmax_layer����Ҳû��ѵ�����������ԱȽϼ򵥣�ֻ��darknet�е�ʵ���ƺ�������һЩ��̫�����Ķ�����������Щ�ط�����ϱȽϷѾ���
	** softmax_layer��������
	** ���룺 batch
	**       intputs
	**       groups
	** ע�⣺�˴���softmax_layer�㣬����ͼƬ������Ԫ�ظ���l.inputs�������Ԫ�ظ���l.outputs��������Ԫ���������Ԫ�ظ���Ҳ����ͬ����
	*/
	softmax_layer make_softmax_layer(int batch, int inputs, int groups);

#ifdef __cplusplus
}
#endif