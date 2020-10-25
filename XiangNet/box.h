#ifndef BOX_H
#define BOX_H

//矩形框,归一化坐标
typedef struct box
{
	float x, y, w, h;
}box;

box float_to_box(float *f, int stride);

float box_iou(box a, box b);



#endif // !BOX_H
