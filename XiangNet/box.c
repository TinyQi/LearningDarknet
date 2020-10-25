#include "box.h"

/** �Ӵ洢���ο���Ϣ�Ĵ������У���ȡĳһ�����ο�λ��Ϣ�����أ�f�������������ĳһ��ʼ��ַ��.
* @param f �洢�˾��ο���Ϣ��ÿ�����ο���5������ֵ���˺�������ȡ����4�����ڶ�λ�Ĳ���x,y,w,h����������������ţ�
* @param stride ��ȣ���������Խȡֵ
* @return b ���ο���Ϣ
*/
box float_to_box(float *f, int stride)
{
	/// f�д洢ÿһ�����ο���Ϣ��˳��Ϊ: x, y, w, h, class_index���������ȡǰ�ĸ���
	/// Ҳ�����ο�Ķ�λ��Ϣ�����һ�������������Ϣ���ڴ˴���ȡ
	box b;
	b.x = f[0];
	b.y = f[1 * stride];
	b.w = f[2 * stride];
	b.h = f[3 * stride];
	return b;
}


/** �����������ο��ཻ���־��ε�ĳһ�ߵı߳����ӵ���������������ཻ���־��εĸߣ�Ҳ�����ǿ�.
* @param x1 ��һ�����ο��x���꣨����y���꣬�ӵ�������������������ཻ���־��εĿ����������x����)
* @param w1 ��һ�����ο�Ŀ������Ҫ�����ཻ���־��εĸߣ���Ϊy���꣬���淲��˵x����ģ�������Ϊy���꣬��Ȼ����Ӧ���Ϊ��)
* @param x2 �ڶ������ο��x����
* @param w2 �ڶ������ο�Ŀ�
* @details ��ֽ�ϻ�һ���������Σ��Լ���һ����μ��㽻��������ͺ��������Ĵ����ˣ����ȼ�����������ߵ�x���꣬�Ƚϴ�С��
*          ȡ����ߣ���Ϊleft����������������ұߵ�x���꣬ȡ��С�ߣ���Ϊright��right-left�����ཻ���־��εĿ�
* @return �������ο��ཻ���־��εĿ���߸�
*/
float overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

/** �������ο��󽻣������������ο�a,b�ཻ���ֵ����.
* @return ��������a,b�ཻ���ֵ����
* @note ���������β��ཻ��ʱ�򣬷��ص�ֵΪ0����ʱ����õ���w,h��С��0,w,h�ǰ�������overlap()�����ķ�ʽ����õ��ģ�
*       ��ֽ�ϱȻ�һ�¾�֪��Ϊʲô��С��0�ˣ�
*/
float box_intersection(box a, box b)
{
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

/** �������ο��󲢣������������ο�a,b�󲢵����.
* @return ��������a,b��֮��������������a���������b�������ȥ�ཻ���ֵ������
*/
float box_union(box a, box b)
{
	float i = box_intersection(a, b);
	float u = a.w*a.h + b.w*b.h - i;
	return u;
}


/** ����IoUֵ.
* @details IoUֵ����Ŀ���⾫ȷ�ȵ�һ������ָ�꣬ȫ����intersection over union����������ľ��ǽ��Ȳ�ֵ��
*          �����ϵ���˼��ֱ�ӣ��������������ཻ���ֵ����������������֮�����������������������ָ��ʱ��
*          ����Ϊģ�ͼ�⵽�ľ��ο���GroundTruth��ǵľ��ο�֮��Ľ��Ȳ�ֵ�����ɷ�ӳ��⵽�ľ��ο���GroundTruth֮����ص��ȣ���
*          ���������ο���ȫ�ص�ʱ��ֵΪ1����ȫ���ཻʱ��ֵΪ0��
*/
float box_iou(box a, box b)
{
	return box_intersection(a, b) / box_union(a, b);
}