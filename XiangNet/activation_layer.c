#include "activation_layer.h"
#include <stdlib.h>
#include <stdio.h>
layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
{
	layer l = { 0 };
	l.type = ACTIVE;

	l.inputs = inputs;
	l.outputs = inputs;
	l.batch = batch;
	
	//todo:yolo2源码里这里是float*,我觉得不用啊
	l.output = calloc(batch * inputs, sizeof(float));
	l.delta = calloc(batch * inputs, sizeof(float));

	l.forward = forward_activation_layer;
	l.backward = backward_activation_layer;

	l.activation = activation;
	fprintf(stderr, "Activation Layer: %d inputs\n", inputs);
	
	return l;
}


void forward_activation_layer(layer l, network net)
{

}


void backward_activation_layer(layer l, network net)
{

}
