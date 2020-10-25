#ifndef REORG_LAYER_H
#define REORG_LAYER_H

#include "layer.h"
#include "network.h"

layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra);

#endif