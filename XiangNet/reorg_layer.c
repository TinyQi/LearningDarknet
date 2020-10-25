#include "reorg_layer.h"

//结构重组层,比如将矮胖的 26X26X256 变成 高瘦的 13X13X1024
layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra)
{
	layer l = { 0 };
	return l;
}
