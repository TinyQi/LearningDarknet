#include "activations.h"
#include "layer.h"
#include "network.h"


typedef layer connected_layer;

#ifdef __cplusplus
extern "C" {
#endif
	/*
	** ����ȫ���Ӳ�
	** ���룺 batch             �ò�������һ��batch�����е�ͼƬ����������net.batch
	**       inputs            ȫ���Ӳ�ÿ������ͼƬ��Ԫ�ظ���
	**       outputs           ȫ���Ӳ����Ԫ�ظ����������������ļ�ָ�������δָ����Ĭ��ֵΪ1,��parse_connected()�и�ֵ��
	**       activation        ���������
	**       batch_normalize   �Ƿ����BN
	** ���أ� ȫ���Ӳ�l
	*/
	connected_layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize);

#ifdef __cplusplus
}
#endif