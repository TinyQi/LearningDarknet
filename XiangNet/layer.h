#ifndef LAYER_H
#define LAYER_H

#include "activations.h"

//前向声明,只是为了typedf struct layer ,这样之后在layer结构体里声明layer变量就不需要家sruuct关键词
struct layer;
typedef struct layer layer;

struct network;
typedef struct network network;

typedef enum LAYER_TYPE
{

}LAYER_TYPE;

typedef enum COST_TYPE
{

}COST_TYPE;


typedef struct layer
{
	LAYER_TYPE type;
	COST_TYPE cost_type;
	ACTIVATION activations;

	//成员函数,用指针函数实现
	void(*forward)(struct layer, network);
	void(*backward)(struct layer, network);
	void(*update)(struct layer, int a,float b,float c,float d);


	int classes;
	float jitter;

}layer;





















#endif // LAYER_H


