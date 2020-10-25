#include "region_layer.h"
#include "activations.h"
#include "utils.h"
#include "box.h"
#include "blas.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
//yolo2����ʧ��


layer make_region_layer(int batch, int h, int w, int n, int classes, int coords)
{
	layer l = { 0 };
	l.type = REGION;

	/// �����ڶ��������ο�layer.h�е�ע��
	l.n = n;                                                ///< һ��cell��������Ԥ����ٸ����ο�box��
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = n*(classes + coords + 1);                         ///< region_layer�����ͨ����
	l.out_w = l.w;                                          ///< region_layer������������ߴ�һ�£�ͨ����Ҳһ����Ҳ������һ�㲢���ı��������ݵ�ά��
	l.out_h = l.h;
	l.out_c = l.c;
	l.classes = classes;                                    ///< �������������ѵ�����ݼ�����ӵ�е��������������
	l.coords = coords;                                      ///< ��λһ����������Ĳ���������һ��ֵΪ4,�����������ĵ�����x,y�Լ�����w,h��
	l.cost = calloc(1, sizeof(float));                      ///< Ŀ�꺯��ֵ��Ϊ�����ȸ�����ָ��
	l.biases = calloc(n * 2, sizeof(float));
	l.bias_updates = calloc(n * 2, sizeof(float));
	l.outputs = h*w*n*(classes + coords + 1);               ///< һ��ѵ��ͼƬ����region_layer���õ������Ԫ�ظ���������������*ÿ������Ԥ��ľ��ο���*ÿ�����ο�Ĳ���������
	l.inputs = l.outputs;                                   ///< һ��ѵ��ͼƬ���뵽reigon_layer���Ԫ�ظ�����ע����һ��ͼƬ������region_layer������������Ԫ�ظ�����ȣ�
	/**
	* ÿ��ͼƬ���е���ʵ���ο�����ĸ�����30��ʾһ��ͼƬ�������30��ground truth���ο�ÿ����ʵ���ο���
	* 5������������x,y,w,h�ĸ���λ�������Լ��������,ע��30��darknet������д���ģ�ʵ����ÿ��ͼƬ����
	* ��û��30����ʵ���ο�Ҳ��û����ô���������Ϊ�˱���һ���ԣ����ǻ�������ô��Ĵ洢�ռ䣬ֻ�����е�
	* ֵδ�ն���.
	*/
	l.truths = 30 * (5);
	l.delta = calloc(batch*l.outputs, sizeof(float));       ///< l.delta,l.input,l.output���������Ĵ�С��һ����
	/**
	* region_layer�����ά��Ϊl.out_w*l.out_h�����������ά�ȣ����ͨ����Ϊl.out_c����������ͨ������
	* ��ͨ��������n*(classes+coords+1)����region_layer�����l.output�е��״洢��ʲô�أ��洢��
	* ��������grid cell����Ԥ����ο�box����������Ϣ����Yolo���ľ�֪����Yolo���ģ�����ս�ͼƬ
	* ���ֳ���S*S��������Ϊ7*7��������ÿ��������Ԥ��B����������B=2�����ο����һ������ľ�����Щ
	* ������������������Ԥ����ο���Ϣ��Ŀ����ģ���У������þ��ο�����ʾ����λ��⵽�����壬ÿ�����ο���
	* �����˾��ο�λ��Ϣx,y,w,h��������������Ŷ���Ϣc���Լ����ڸ���ĸ��ʣ������20�࣬��ô���о��ο�
	* ������������������20��ĸ��ʣ���ע���ˣ������ʵ���������е������в�ͬ�����Ȳ�����Ȼ���ܲ�ͬ������
	* ����������������ÿ������Ԥ��2��box��Ҳ�п��ܸ��ࣩ����Ϊ�ؼ����ǣ����ά�ȵļ��㷽ʽ��ͬ���������ᵽ
	* ���һ�������ά��Ϊһ��S_w*S_c*(B*5+C)��tensor����������������S*S��������д��S_w��S_c�ǿ��ǵ�
	* ���񻮷�ά�Ȳ�һ��S_w=S_c=S������ò�������õĶ���S_w=S_c�ģ�����7*7,13*13����֮���׾Ϳ����ˣ���
	* ʵ���ϣ������е㲻ͬ�������ά��Ӧ��ΪS_w*S_c*B*(5+C),CΪ�����Ŀ�����繲��20�ࣻ5����Ϊ��4����λ
	* ��Ϣ�����һ�����Ŷ���Ϣc������5��������Ҳ��ÿ�����ο򶼰���һ�����ڸ���ĸ��ʣ����������о��ο���
	* һ�����ڸ���ĸ��ʣ������Դ�l.outputs�ļ��㷽ʽ�п��������Զ�Ӧ�ϣ�l.out_w = S_w, l.out_c = S_c,
	* l.out_c = B*(5+C)����֪��������״洢ʲô֮�󣬽�����Ҫ��������ô�洢�ģ��Ͼ��������һ����ά������
	* ��ʵ��������һ��һά�������洢�ģ���ϸ��ע�Ϳ��Բο�����forward_region_layer()�Լ�entry_index()
	* ������ע�ͣ���������������ֻ��ǱȽ��������ģ�Ӧ�ý���ͼ��˵����
	*/
	l.output = calloc(batch*l.outputs, sizeof(float));
	int i;
	for (i = 0; i < n * 2; ++i) {
		l.biases[i] = .5;
	}

	l.forward = forward_region_layer;
	l.backward = backward_region_layer;

	fprintf(stderr, "detection\n");
	srand(0);

	return l;
}


/**
* @brief ����ĳ�����ο���ĳ��������l.output�е�������һ�����ο������x,y,w,h,c,C1,C2...,Cn��Ϣ��
*        ǰ�ĸ����ڶ�λ�������Ϊ���ο�����������Ŷ���Ϣc�������ο��д�������ĸ���Ϊ��󣬶�C1��Cn
*        Ϊ���ο���������������ֱ�������n������ĸ��ʡ������������ȡ�þ��ο��׸���λ��ϢҲ��xֵ��
*        l.output����������ȡ�þ��ο����Ŷ���Ϣc��l.output�е���������ȡ�þ��ο�����������ʵ��׸�
*        ����Ҳ��C1ֵ�������������ǻ�ȡ���ο��ĸ�������������ȡ�����������entry��ֵ����Щ��
*        forward_region_layer()�����ж����õ�������l.output�Ĵ洢��ʽ����entry=0ʱ�����ǻ�ȡ���ο�
*        x������l.output�е���������entry=4ʱ�����ǻ�ȡ���ο����Ŷ���Ϣc��l.output�е���������
*        entry=5ʱ�����ǻ�ȡ���ο��׸���������C1��l.output�е�������������Բο�forward_region_layer()
*        �е��ñ�����ʱ��ע��.
* @param l ��ǰregion_layer
* @param batch ��ǰ��Ƭ������batch�еĵڼ��ţ���Ϊl.output�а�������batch�����������Ҫ��λĳ��ѵ��ͼƬ
*              ������ڶ������е�ĳ�����ο򣬵�Ȼ��Ҫ�ò���.
* @param location ���������˵ʵ�����о�������߲����������������������ȡn��loc��ֵ�����n���Ǳ�ʾ������
*                 �ĵڼ���Ԥ����ο򣨱���ÿ������Ԥ��5�����ο���ônȡֵ��Χ���Ǵ�0~4����loc����ĳ��
*                 ͨ���ϵ�Ԫ��ƫ�ƣ�region_layer�����ͨ����Ϊl.out_c = (classes + coords + 1)����
*                 ����˵����û��˵���ף��ⶼ��l.output�Ĵ洢�ṹ��أ���������ϸע���Լ�����˵������֮��
*                 �鿴һ�µ��ñ������ĸ�����orward_region_layer()��֪���ˣ�����ֱ������n��j*l.w+i�ģ�
*                 û�б�Ҫ����location�������������¼���һ��n��loc.
* @param entry �����ƫ��ϵ���������������������Ҫ����l.output�Ĵ洢�ṹ�ˣ���������ϸע���Լ�����˵��.
* @details l.output��������Ĵ洢�����Լ��洢��ʽ�Ѿ��ڶ���ط�˵���ˣ��ٶ�����ֶ�����ͼ��˵�����˴���
*          ��Ҫ���¼��䣬��Ϊ����Ĳο�ͼ��˵����l.output�д洢������batch��ѵ�������ÿ��ѵ��ͼƬ�������
*          l.out_w*l.out_h������ÿ�������Ԥ��l.n�����ο�ÿ�����ο���l.classes+l.coords+1��������
*          �����һ������ͨ����Ϊl.n*(l.classes+l.coords+1)�����������������������ά�����Ǹ�ʲô���ӵġ�
*          չ��һά����洢ʱ��l.output�������ȷֳ�batch����Σ�ÿ����δ洢��һ��ѵ��ͼƬ�������������һ��ϸ�֣�
*          ȡ���е�һ��η������ô���д洢�˵�һ��ѵ��ͼƬ�����������Ԥ��ľ��ο���Ϣ��ÿ������Ԥ����l.n�����ο�
*          �洢ʱ��l.n�����ο��Ƿֿ��洢�ģ�Ҳ�����ȴ洢���������еĵ�һ�����ο򣬶���洢���������еĵڶ������ο�
*          �������ƣ����ÿ��������Ԥ��5�����ο�����Լ�������һ��ηֳ�5���жΡ�����ϸ�֣�5���ж���ȡ��
*          һ���ж�������������ж��а��У���l.out_w*l.out_h�����񣬰��д洢�����δ洢������ѵ��ͼƬ�������������
*          �ĵ�һ�����ο���Ϣ��Ҫע����ǣ�����жδ洢��˳�򲢲��ǰ��������洢ÿ�����ο��������Ϣ��
*          �����ȴ洢���о��ο��x�����������е�y,Ȼ�������е�w,����h��c�����ĵĸ�������Ҳ�ǲ�ֽ��д洢��
*          ������һ���Ӵ洢��һ�����ο�������ĸ��ʣ������ȴ洢��������������һ��ĸ��ʣ��ٴ洢�����ڶ���ĸ��ʣ�
*          ������˵��һ�ж����ȴ洢��l.out_w*l.out_h��x��Ȼ����l.out_w*l.out_c��y��������ȥ��
*          �����l.out_w*l.out_h��C1�����ڵ�һ��ĸ��ʣ���C1��ʾ���������ƣ���l.out_w*l.outh��C2,...,
*          l.out_w*l.out_c*Cn�����蹲��n�ࣩ�����Կ��Լ������жηֳɼ���С�Σ�����Ϊx,y,w,h,c,C1,C2,...Cn
*          С�Σ�ÿС�εĳ��ȶ�Ϊl.out_w*l.out_c.
*          ���ڻع����������������������batch���Ǵ�ε�ƫ�������ӵڼ�����ο�ʼ����Ӧ�ǵڼ���ѵ��ͼƬ����
*          ��location����õ���n�����жε�ƫ�������ӵڼ����жο�ʼ����Ӧ�ǵڼ������ο򣩣�
*          entry����С�ε�ƫ�������Ӽ���С�ο�ʼ����Ӧ���������ֲ�����x,c����C1������loc�������Ķ�λ��
*          ǰ��ȷ���õڼ�����еĵڼ��ж��еĵڼ�С�ε��׵�ַ��loc���ǴӸ��׵�ַ������loc��Ԫ�أ��õ����ն�λ
*          ĳ�����������x��c��C1��������ֵ������l.output�д洢������������ʾ���������ֻ����һ��ѵ��ͼƬ�������
*          ���batchֻ��Ϊ0��������l.out_w=l.out_h=2,l.classes=2����
*          xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2-#-xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2��
*          n=0��λ��-#-��ߵ��׵�ַ����ʾÿ������Ԥ��ĵ�һ�����ο򣩣�n=1��λ��-#-�ұߵ��׵�ַ����ʾÿ������Ԥ��ĵڶ������ο�
*          entry=0,loc=0��ȡ����x���������һ�ȡ���ǵ�һ��xҲ��l.out_w*l.out_h�������е�һ�������е�һ�����ο�x������������
*          entry=4,loc=1��ȡ����c���������һ�ȡ���ǵڶ���cҲ��l.out_w*l.out_h�������еڶ��������е�һ�����ο�c������������
*          entry=5,loc=2��ȡ����C1���������һ�ȡ���ǵ�����C1Ҳ��l.out_w*l.out_h�������е����������е�һ�����ο�C1������������
*          ���Ҫ��ȡ��һ�������е�һ�����ο�w�����������أ�����Ѿ���ȡ����xֵ����������Ȼ��x����������3*l.out_w*l.out_h���ɻ�ȡ����
*          ������delta_region_box()������������
*          ���Ҫ��ȡ�����������е�һ�����ο�C2�����������أ�����Ѿ���ȡ����C1ֵ����������Ȼ��C1����������l.out_w*l.out_h���ɻ�ȡ����
*          ������delta_region_class()�����е�������
*          ���Ͽ�֪��entry=0ʱ,��ƫ��0��С�Σ��ǻ�ȡx��������entry=4,�ǻ�ȡ���Ŷ���Ϣc��������entry=5���ǻ�ȡC1������.
*          l.output�Ĵ洢��ʽ���¾������������˾���˵���Ѿ�������ˣ������ӻ�Ч���վ�����ͼ��˵����
*/
int entry_index(layer l, int batch, int location, int entry)
{
	//location:��ÿ������һ��Ԥ����˳�����������ڼ���Ԥ��򣬱���2*2��������ͼ��4������ÿ������2��Ԥ���
	//         ��ô����Ԥ�������˳�����tl1,tr1,bl1,br1,tl2,tr2,bl2,br2,���location=5,��ô����ָtl2���Ԥ���
	//entry:����һ���������ȡ��һ�����������ܹ���x,y,w,h,c,C1,C2����7������entry����ľ��ǵڼ��࣬
	//      entry=0����ָx����׵�ַ��entry=3����ָw����׵�ַ

	//n:����ÿ�������Ӧ�ĵڼ���Ԥ����ο�
	int n = location / (l.w * l.h);
	//loc:ÿһ��С���������������loc=2,���������������ǵ����������еľ��ο���Ӧ��Ԥ����������
	//    ��x��˵������xxxxyyyywwwwhhhhccccC1C1C1C1C2C2C2C2�е�xxxx�ĵ�����x

	int loc = location % (l.w*l.h);
	return batch*l.outputs + n*l.w*l.h*(l.coords + l.classes + 1) + entry*l.w*l.h + loc;
}

/** ��ȡĳ�����ο��4����λ��Ϣ����������ľ��ο�������l.output�л�ȡ�þ��ο�Ķ�λ��Ϣx,y,w,h��.
* @param x region_layer���������l.output����������batchԤ��õ��ľ��ο���Ϣ
* @param biases
* @param n
* @param index ���ο���׵�ַ�����������ο��д洢���׸�����x��l.output�е�������
* @param i �ڼ��У�region_layerά��Ϊl.out_w*l.out_c��ͨ����Ϊ��
* @param j
* @param w
* @param h
* @param stride����
*/
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
	//�������,��Ԥ��ֵ����ɹ�һ����������Ϣ
	// �����x�Ѿ�����softmax����������
	box b;
	b.x = (i + x[index + 0 * stride]) / w;
	b.y = (j + x[index + 1 * stride]) / h;

	//�����еĽ��빫ʽ:exp(tw)*pw:�����tw��������Ԥ���ֵ,�����pw���������Ŀ��Ԥ��ֵ,���������biases[2 * n]
	b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
	b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
	return b;
}

//todo
float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
	box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
	float iou = box_iou(pred, truth);

	//�������,
	// ����һ����������Ϣ�����yolo2������:tx,ty�ǻ���cell���Ͻǵ�ƫ����,ȡֵ[0,1],
	//							tw,thҲ��yolo2��������뷽ʽ,
	//							���������:label.w = pw * exp(tw),����,label.�ǹ�һ������,pw��������wֵ,tw������Ԥ��ֵ
	float tx = (truth.x*w - i);// ������"label.x = (i + tx)/w"�����ʽ����tx
	float ty = (truth.y*h - j);
	float tw = log(truth.w*w / biases[2 * n]); // ����ı��뷽ʽ�ǽ�����̵ĵ���,����"label.w = pw * exp(tw)"�����ʽ����tw
	float th = log(truth.h*h / biases[2 * n + 1]);

	//����Ĵ����Ѿ�����ǩ�����yolo�Ľ����ʽ,Ȼ���ٸ�����Ԥ�������ʧ����
	//��ʧ���Ǳ�ǩֵ(�����)��ȥԤ��ֵ
	delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
	delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);

	delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
	delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);

	//ע��,�����stride�������Ҫ���ṹ��Ϣ���ڴ�ṹ
	// xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C1C1C1C1C1C2C2C2C2C2C2-##-xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C2C1C2C1C2C1C2C1C2C1C2
	// ��,�������l.output�ﱣ���Ԥ����Ϣ,��   1.ÿ��Ԫ������һ��;2.ÿһ���ȼ������������һ��;3.Ȼ������һ��batch
	// ���������strideӦ����l.w*l.h,��Ϊһ���ȼ���������� "l.w*l.h" ��
	return iou;
}

void delta_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, int stride, float *avg_cat)
{
	int i, n;
	if (hier) {
		//������yolo9000�õ�һЩת���ַ�
		/*float pred = 1;
		while (class >= 0) {
			pred *= output[index + stride*class];
			int g = hier->group[class];
			int offset = hier->group_offset[g];
			for (i = 0; i < hier->group_size[g]; ++i) {
				delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
			}
			delta[index + stride*class] = scale * (1 - output[index + stride*class]);

			class = hier->parent[class];
		}
		*avg_cat += pred;*/
	}
	else {
		for (n = 0; n < classes; ++n) {
			//��ʧ����,
			// �����ǰ��n�������Ǳ�ǩ(class)������,��ôĿ�����1,������ʧ����1 - ��ǰ���͸���,
			// �����ǰn�����Ͳ��Ǳ�ǩ(class)������,��ôĿ�����0,������ʧ���� 0 - ��ǰ���͸���,
			delta[index + stride*n] = scale * (((n == class) ? 1 : 0) - output[index + stride*n]);
			if (n == class) *avg_cat += output[index + stride*n];
		}
	}
}


/**
* @param l
* @param net
* @details ��������ε�����entry_index()��������ʹ�õĲ���������ͬ�����������һ��������ͨ�����һ��������
*          ����ȷ����region_layer���l.output�����ݴ洢��ʽ��Ϊ�������������豾���������l.w = 2, l.h= 3,
*          l.n = 2, l.classes = 2, l.coords = 4, l.c = l.n * (l.coords + l.classes + 1) = 21,
*          l.output�д洢�����о��ο����Ϣ������ÿ�����ο����4����λ��Ϣ����x,y,w,h��һ�����Ŷȣ�confidience��
*          ����c���Լ��������ĸ���C1,C2�������У������ֻ���������l.classes=2������ôһ������ͼƬ���ջ���
*          l.w*l.h*l.n�����ο�l.w*l.h��Ϊ����ͼ�񻮷ֲ�����ĸ�����ÿ������Ԥ��l.n�����ο򣩣���ô
*          l.output�д洢��Ԫ�ظ�������l.w*l.h*l.n*(l.coords + 1 + l.classes)����ЩԪ��ȫ�������һά����
*          ����ʽ�洢��l.output�У��洢��˳��Ϊ��
*          xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C1C1C1C1C1C2C2C2C2C2C2-##-xxxxxx-yyyyyy-wwwwww-hhhhhh-cccccc-C1C2C1C2C1C2C1C2C1C2C1C2
*          ����˵�����£�-##-�����ֳ����Σ����ҷֱ��Ǵ�����������ĵ�1��box�͵�2��box����Ϊl.n=2����ʾÿ������Ԥ������box����
*          �ܹ���l.w*l.h�������Ҵ洢ʱ�������������x,y,w,h,c��Ϣ�۵�һ����ƴ�����������xxxxxx��������Ϣ����l.w*l.h=6����
*          ��Ϊÿ����l.classes��������𣬶���Ҳ�Ǻ�xywhһ����ÿһ�඼���д洢���ȴ洢l.w*l.h=6��C1�࣬����洢6��C2�࣬
*         ��Ϊ�����ע�Ϳ��Ժ����е����ע�ͣ�ע�ⲻ��C1C2C1C2C1C2C1C2C1C2C1C2��ģʽ�����ǽ����е����𿪷ֱ��д洢����
* @details ���ŶȲ���c��ʾ���Ǹþ��ο��ڴ�������ĸ��ʣ���C1��C2�ֱ��ʾ���ο��ڴ�������ʱ��������1������2�ĸ��ʣ�
*          ���c*C1���þ��ο��ڴ�������1�ĸ��ʣ�c*C2���þ��ο��ڴ�������2�ĸ���
*/
void forward_region_layer(const layer l, network net)
{
	// ��ʧ�ļ���,��Ԥ��2����������Ϊ��
	// yolo2��������������Ԥ���,Ԥ���ά����:4άλ����Ϣ(xywh),1ά���Ŷ�,2ά�����Ϣ,����yolo2���õ�ÿ��������5�������Ļ�,
	// ��ÿ���������(4+1+2)*5=35ά,Ҳ����35��ͨ��,ÿ�������������ά�ȶ����漰����ʧ�ļ���,�������ʧ���㿴����
	// 
	// ������ʧ�Ǿ���ĳ���������˵,����һ��ǰ��ѵ����˵�޷Ǿ�������ѭ������batch,w,h������,����һ��batch����������

	// �����ָ���Ԥ������ĺͲ�����Ԥ�������,
	// ����Ԥ������������:��gd��iou��ߵ��������Ǹ���Ԥ���gd�������
	// ������Ԥ������������:���˸���Ԥ������֮������������
	// 
	// ����Ԥ���������������ʧ����:
	// 1.Ԥ����xywh��gd��xywh����ʧ����,����mse��ʧ
	// 2.���Ŷȵ���ʧ����,
	//		a.���û�д�rescore�������,����Ԥ������ŶȺ�1֮���mse��ʧ;
	//  	b.�������rescore�������,��ô������Ԥ�����gd��iouΪĿ��,����Ԥ������Ŷ������iou֮���mse��ʧ
	// 3.������ʧ����,����Ԥ����������gd�������mse��ʧ,�������2�����ά��[0.5,0.88],gd�����ά��[0,1],�����Ӧ����mse
	// ������Ԥ���������������ʧ����
	// 4.������ѵ����ͼƬ����û�дﵽ12800֮ǰ(����ֵ),����Ԥ�����������xywh��mse��ʧ(������λ���ǹ̶��������ļ��������)
	//		a.��һ����Ŀ�ľ�����������ѧϰ������λ����Ϣ,��������
	// 5.����ѵ��ͼƬ��������12800֮��Ͳ��ټ���xywh����ʧ��,���Ƿ�����������Ŷ���ʧ
	//		a.������Ԥ�����Ŷȳ���0.6(����ֵ),�򲻼������Ŷ���ʧ
	//		b.������Ԥ�����ŶȲ�����0.6,��������Ŷȵ�mse��ʧ,���Ŷ�Ŀ����0,��Ϊ��ʱ���ж���������ǲ����������,����һ�������Ŷ�,��ô������ʧ

	int i, j, b, t, n;
	//��ʧ��������������
	memcpy(l.output, l.inputs, l.batch * l.outputs * sizeof(float));
	//����������ͼ��Ԫ�ض���ʼ��Ϊ0
	memset(l.delta, 0, l.batch * l.outputs * sizeof(float));
	
	//����ѭ�����Ƕ�ÿ��Ԥ���Ľ�����н���
	// xy�ǻ���cell�����ĵ��ƫ����,����logistic�����,Ҳ����sigmoid������,���޶���[0,1]
	for (b = 0; b < l.batch; ++b) {
		// ע��region_layer���е�l.n������ÿ��cell grid��������Ԥ��ľ��ο���������Ǿ�����о���˵ĸ�����
		for (n = 0; n < l.n; ++n) {
			// ��ȡ ĳһ�ж��׸�x�ĵ�ַ
			int index = entry_index(l, b, n*l.w*l.h, 0);
			// ע��ڶ���������2*l.w*l.h��Ҳ���Ǵ�index+l.output����ʼ����֮��2*l.w*l.h��Ԫ�ؽ���logistic���������Ҳ���Ƕ�
			// һ���ж������е�x,y����logistic������������û�и����׵��ǣ�Ϊʲô��x,y���м�������������w,h�أ�
			// ���:�������sigmoid�������ƫ�����޶���[0,1],��ô����ʹ��������ƫ����,
			// ��Ԥ���Ŀ������Ŀ�����ͼƬ���κεط�,���޶��Ļ�,���ƫ�������ܻ�ܴ�,���յ���ģ��������
			// ����Ϊ����Ԥ�������ĵ�����cell��,�Ͷ�ƫ������sigmoid��������
			// ��Ԥ����w��h�ǿ��Դ���1��,����ܴ�Ŀ�,�����˵�ǰ������
			activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
			
			// ������һ�����˴��ǻ�ȡһ���ж����׸����Ŷ���Ϣcֵ�ĵ�ַ������Ը��ж������е�cֵ�����ж��ڹ���l.w*l.h��cֵ������logistic���������
			// ���Ŷ���[0,1]����,��logistic�����(sigmoid����)����
			index = entry_index(l, b, n*l.w*l.h, 4);
			activate_array(l.output + index, l.w*l.h, LOGISTIC);
		}
	}

	if (l.softmax_tree) {
		//
	}
	else if (l.softmax)
	{
		int index = entry_index(l, 0, 0, 5);

		//������class�������softmax,Ҳ�������������͵ĺ�Ϊ1 ,ÿһ�����͵�expֵ���Գ�����������exp���ܺ�
		softmax_cpu(net.input + index, l.classes, l.batch*l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, 1, l.output + index);
	}

	//����Ƿ�ѵ������,��ֱ���˳�,��Ϊ����Ҫ������ʧ��
	if (!net.train)return;

	//avg��ͷ�ı�������Ϊ�˴�ӡ��ƽ������
	float avg_iou = 0;	
	float recall = 0;	//�ٻ���
	float avg_cat = 0;
	float avg_obj = 0;
	float avg_anyobj = 0;
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;

	//���Ĳ�ѭ�����Ǳ���ÿ�������ÿ�������
	//ÿ������򶼽������Ŷȵ���ʧ����
	//���ҳ��ڵ���(12800��֮ǰ),�������ÿ��Ԥ����������֮���xywh����ʧ
	for ( b = 0; b < l.batch; b++)
	{

		//1.-----------------
		//���㸺���������Ŷ���ʧ
		//���������������Ԥ�����������������ʧ
		//����
		for ( j = 0; j < l.h; j++)
		{
			for ( i = 0; i < l.w; i++)
			{
				for ( n = 0; n < l.n; n++)//����ÿ����������������
				{
					//��ȡ��ǰ������xywh���������
					int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);

					//Ԥ���
					box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, l.w*l.h);

					//�ӱ�ǩ���ҳ��뵱ǰԤ���iou���Ŀ�
					float best_iou = 0;
					//���30���ڲ�������,yolo2�����֧��30����ǩ��
					for (t = 0; t < 30; t++)
					{
						//t * 5����Ϊһ����ǩ��5��Ԫ��,xywh��class
						//b * l.truths ����Ϊnet.truth������һ��batch��ͼƬ,��l.truths��ָһ��ͼƬӵ�е����Ǳ�ǩ����
						//net.truth����ڴ�ṹӦ������ǰ������,��ʹ��һ��ͼƬ��ֻ��һ����,���Ƕ�������30������ڴ�ռ�
						box truth = float_to_box(net.truth + t * 5 + b * l.truths, 1);

						//�������һ���յı�ǩ,֤����ͼƬ�ı�ǩ�Ѿ�����ͷ��,������,��仰����֤��"net.truth����ڴ�ṹӦ������ǰ������"��仰
						if (!truth.x) break;

						//����Ԥ�������ʵ��ǩ��iou
						float iou = box_iou(pred, truth);
						
						if (iou > best_iou)
						{
							best_iou = iou;
						}
					}

					int obj_index = entry_index(l, b, n * l.w * l.h + j * l.w + i, 4);
					avg_anyobj += l.output[obj_index];//�������

					l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
					if (best_iou > l.thresh)
					{
						l.delta[obj_index] = 0;
					}
					
					//���ڵ���,������������Ԥ���������������ʧ,�Դ������������
					if (*net.seen < 12800)
					{
						box truth = { 0 };
						truth.x = (i + 0.5) / l.w;
						truth.y = (j + 0.5) / l.h;
						//���l.biases��w��h��ֵ,��make_region_layer()���캯�������Ĭ�ϸ�ֵΪ0.5
						//����parser.c�ļ��еĽ���region��ĺ�������ֶ�l.biases���и�ֵ,��ֵ�����ݾ��������ļ�����anchor�������ֵ
						//Ҳ����˵,���ܹ��캯������Ĭ�ϸ�ֵΪ0.5,����ֻҪ�����ļ���ָ����anchor�Ĳ���,�����������������ʧ
						truth.w = l.biases[2 * n] / l.w;
						truth.h = l.biases[2 * n + 1] / l.h;
						
						//todo:scale�Ǹ����??? Ӧ�þ���һ��Ȩ��,ƽ���λ�ò�������ʧ��������ʧ�������ռ��
						delta_region_box(truth, l.output, l.biases, n, obj_index, i, j, l.w, l.h, l.delta, 0.01, l.w*l.h);
					}
				}
			}
		}

		//2.-----------------
		//���ݱ�ǩ���㸺����ñ�ǩ��Ԥ���ĸ�������ʧ:xywh,���Ŷ�,class
		for (t = 0; t < 30; t++)
		{
			//��net.truth��ȡ��һ����ǩ
			box truth = float_to_box(net.truth + t * 5 + b * l.truths, 1);

			if (!truth.x)break;
			float best_iou = 0;
			int best_n = 0;
			//���ݱ�ǩ��������Ϣ,���iou����ԭ��,�ҵ�����Ԥ�������ǩ��Ԥ���
			i = truth.x * l.w;
			j = truth.y * l.h;

			//��truth�Ƶ�ͼƬ���Ͻ�,��Ϊ��������iou��ʱ��,���Ԥ���ͱ�ǩ������ϽǶ��Ƶ�(0,0)ȥ����iou
			box shift_truth = truth;
			shift_truth.x = 0;
			shift_truth.y = 0;
			for ( n = 0; n < l.n; n++)
			{
				int box_index = entry_index(l, b, j * l.w + i + l.w * l.h * n, 0);
				box pred = get_region_box(l.output, l.biases, n, box_index, i, j, l.w, l.h, 1);
				
				//�Ƿ�ʹ�������ļ����anchor��Ϣ
				if (l.bias_match)
				{
					pred.w = l.biases[2 * n] / l.w;
					pred.h = l.biases[2 * n + 1] / l.h;
				}

				pred.x = 0;
				pred.y = 0;

				float iou = box_iou(pred, shift_truth);
				if (best_iou < iou)
				{
					best_iou = iou;
					best_n = n;//��¼�˵�ǰcell����ѵ�Ԥ���,�������ǩ�������ʧ
				}
			}

			//����һ����֪�������ĸ�Ԥ���Ҫ����ǰ��ǩ������ʧ
			//����֪����Ԥ�����l.output�е��ڴ������Ͷ�λ���˾����Ԥ���
			int box_index = entry_index(l, b, j*l.w + i + best_n * l.w*l.h, 0);

			//����������ʧ
			//
			float iou = delta_region_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, l.delta, l.coord_scale *  (2 - truth.w*truth.h), l.w*l.h);
			//ע��:�������Ȩ�ص�����:l.coord_scale *  (2 - truth.w*truth.h):
			//						  l.coord_scale = 1,
			//						  (2 - truth.w*truth.h):�����������岻֪��,��������Ԥ���Խ��,����ʧԽС
			//���ڴ�ӡ,iou����0.5֤�������ҵ��˸ñ�ǩ
			if (iou > .5) recall += 1;
			avg_iou += iou;
			

			//�������Ŷ���ʧ,��ʵ�����������"box_index"�����Ϻ���4λ,�͵������Ŷȵ�λ��
			int obj_index = entry_index(l, b, j*l.w + i + best_n * l.w*l.h, 4);
			avg_obj += l.output[obj_index];//���ڼ�¼ƽ�������Ŷ�
			l.delta[obj_index] = l.object_scale * (1 - l.output[obj_index]);
			if (l.rescore)//����������������,�����µķ�ʽ�������Ŷ���ʧ
			{
				l.delta[obj_index] = l.object_scale * (iou - l.output[obj_index]);
			}

			//��ǰ��ǩ������
			int class = net.truth[t * 5 + b*l.truths + 4];//��4���ǵ�5����������,ǰ�ĸ���xywh

			//����class��ʧ
			if (l.map) class = l.map[class];//����Ǹ�yolo9000�йص�,��ʱ�ò���
			int class_index = entry_index(l, b, j*l.w + i + best_n * l.w*l.h, 5);//��ʵ�����������"box_index"�����Ϻ���5λ,�͵�������λ��
			delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat);
			++count;
			++class_count;
		}
	}

	//��������l.cost�����ʧ������ֵ�Ѿ�����Ҫ��,���򴫲����ش�����l.delta���ֵ,���������ÿһ��Ԫ�ص���ʧ
	//��network.c����ļ����void backward_network(network net)������,
	// ����ôһ�����:net.delta = prev.delta;
	// �������ǰ�ǰһ������ľֲ���ʧָ�븳��net.deltaָ��
	// ��region��ķ��򴫲������ֻ��õ�ǰregion���delta���Ƹ�net.delta,���൱�ڸ��Ƹ�ǰһ��������l.delta,�Դ˴ﵽ���򴫲��е���ʽ����

	//��Ϊyolo2��ʧ�����������������ƽ����,������ֻ�Ǽ�����ÿһ���ֵ,����������Ҫƽ����͵ó����յ�l.cost
	(*l.cost) = pow(mag_array(l.delta,l.outputs * l.batch), 2);//�����е�С���,mag_array����,���������ƽ���Ϳ����ŵ�,���Ǻ��������Ѿ�����ƽ������
	printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, count);

}

void backward_region_layer(const layer l, network net)
{
	//��֪��Ϊʲôgithub�ϵ�Դ�����仰ע����,ע���˵Ļ�,��ʧ��ô����ȥ?����ʱ��ע��
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}
