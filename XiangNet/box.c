#include "box.h"

/** 从存储矩形框信息的大数组中，提取某一个矩形框定位信息并返回（f是这个大数组中某一起始地址）.
* @param f 存储了矩形框信息（每个矩形框有5个参数值，此函数仅提取其中4个用于定位的参数x,y,w,h，不包含物体类别编号）
* @param stride 跨度，按倍数跳越取值
* @return b 矩形框信息
*/
box float_to_box(float *f, int stride)
{
	/// f中存储每一个矩形框信息的顺序为: x, y, w, h, class_index，这里仅提取前四个，
	/// 也即矩形框的定位信息，最后一个物体类别编号信息不在此处提取
	box b;
	b.x = f[0];
	b.y = f[1 * stride];
	b.w = f[2 * stride];
	b.h = f[3 * stride];
	return b;
}


/** 计算两个矩形框相交部分矩形的某一边的边长（视调用情况，可能是相交部分矩形的高，也可能是宽）.
* @param x1 第一个矩形框的x坐标（或者y坐标，视调用情况，如果计算的是相交部分矩形的宽，则输入的是x坐标)
* @param w1 第一个矩形框的宽（而如果要计算相交部分矩形的高，则为y坐标，下面凡是说x坐标的，都可能为y坐标，当然，对应宽变为高)
* @param x2 第二个矩形框的x坐标
* @param w2 第二个矩形框的宽
* @details 在纸上画一下两个矩形，自己想一下如何计算交集的面积就很清楚下面的代码了：首先计算两个框左边的x坐标，比较大小，
*          取其大者，记为left；而后计算两个框右边的x坐标，取其小者，记为right，right-left即得相交部分矩形的宽。
* @return 两个矩形框相交部分矩形的宽或者高
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

/** 两个矩形框求交：计算两个矩形框a,b相交部分的面积.
* @return 两个矩形a,b相交部分的面积
* @note 当两个矩形不相交的时候，返回的值为0（此时计算得到的w,h将小于0,w,h是按照上面overlap()函数的方式计算得到的，
*       在纸上比划一下就知道为什么会小于0了）
*/
float box_intersection(box a, box b)
{
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);
	if (w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

/** 两个矩形框求并：计算两个矩形框a,b求并的面积.
* @return 两个矩形a,b求并之后的总面积（就是a的面积加上b的面积减去相交部分的面积）
*/
float box_union(box a, box b)
{
	float i = box_intersection(a, b);
	float u = a.w*a.h + b.w*b.h - i;
	return u;
}


/** 计算IoU值.
* @details IoU值，是目标检测精确度的一个评判指标，全称是intersection over union，翻译成中文就是交比并值，
*          字面上的意思很直接，就是两个矩形相交部分的面积比两个矩形求并之后的总面积，用来做检测评判指标时，
*          含义为模型检测到的矩形框与GroundTruth标记的矩形框之间的交比并值（即可反映检测到的矩形框与GroundTruth之间的重叠度），
*          当两个矩形框完全重叠时，值为1；完全不相交时，值为0。
*/
float box_iou(box a, box b)
{
	return box_intersection(a, b) / box_union(a, b);
}