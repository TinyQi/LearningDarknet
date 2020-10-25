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
	**  ���룺batch    ÿ��batch���е�ͼƬ��
	**      h               ͼƬ�߶ȣ�������
	**      w               ͼƬ��ȣ�������
	c               ����ͼƬͨ����
	n               ����˸���
	size            ����˳ߴ�
	stride          ���
	padding         ���ܲ�0����
	activation      ��������
	batch_normalize �Ƿ����BN(�淶��)
	binary          �Ƿ��Ȩ�ؽ��ж�ֵ��
	xnor            �Ƿ��Ȩ���Լ�������ж�ֵ��
	adam            ʹ��adam�Ż���ʽ
	*/
	convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);


	void forward_convolutional_layer(const convolutional_layer l, network net);

	void backward_convolutional_layer(convolutional_layer l, network net);

	void update_convolutional_layer(convolutional_layer l, int batch, float learning_rate, float momentum, float decay);
#ifdef __cplusplus
}
#endif