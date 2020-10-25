#ifndef ROUTE_LAYER_H
#define ROUTE_LAYER_H
#include "network.h"
#include "layer.h"


typedef layer route_layer;

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_size);

#endif