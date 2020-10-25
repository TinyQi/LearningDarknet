#include "activations.h"
#include "layer.h"
#include "network.h"


typedef layer connected_layer;

#ifdef __cplusplus
extern "C" {
#endif
	/*
	** 构建全连接层
	** 输入： batch             该层输入中一个batch所含有的图片张数，等于net.batch
	**       inputs            全连接层每张输入图片的元素个数
	**       outputs           全连接层输出元素个数（由网络配置文件指定，如果未指定，默认值为1,在parse_connected()中赋值）
	**       activation        激活函数类型
	**       batch_normalize   是否进行BN
	** 返回： 全连接层l
	*/
	connected_layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize);

#ifdef __cplusplus
}
#endif