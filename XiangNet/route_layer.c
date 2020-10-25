#include "route_layer.h"

//连接层,把不同层相互重叠在一起
//例如输入层1：26*26*256   输入层2：26*26*128  则route layer输出为：26*26*（256+128）
route_layer make_route_layer(int batch, int n, int * input_layers, int * input_size)
{
	route_layer l = { 0 };
	return l;
}
