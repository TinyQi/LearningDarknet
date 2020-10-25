#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
	typedef layer maxpool_layer;

	/*
	** �������ػ���
	** ���룺 batch     �ò�������һ��batch�����е�ͼƬ����������net.batch
	**       h,w,c     �ò�����ͼƬ�ĸ߶ȣ��У�����ȣ��У���ͨ����
	**       size      �ػ��˳ߴ�
	**       stride    ���
	**       padding   ���ܲ�0����
	** ���أ� ���ػ���l
	** ˵�������ػ���������Ƚ����ƣ������н϶�ı���������Ⱦ�������������ػ��ˣ��ػ��˳ߴ磬��ȣ���0���ȵȵ�
	*/
	maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding);

#ifdef __cplusplus
}
#endif