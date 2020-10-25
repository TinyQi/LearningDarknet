#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
	typedef layer maxpool_layer;

	/*
	** 构建最大池化层
	** 输入： batch     该层输入中一个batch所含有的图片张数，等于net.batch
	**       h,w,c     该层输入图片的高度（行），宽度（列）与通道数
	**       size      池化核尺寸
	**       stride    跨度
	**       padding   四周补0长度
	** 返回： 最大池化层l
	** 说明：最大池化层与卷积层比较类似，所以有较多的变量可以类比卷积层参数，比如池化核，池化核尺寸，跨度，补0长度等等
	*/
	maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);

#ifdef __cplusplus
}
#endif