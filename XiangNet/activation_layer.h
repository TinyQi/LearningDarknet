#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef layer activation_layer;

	//TODO:×¢ÊÍ
	layer make_activation_layer(int batch, int inputs, ACTIVATION activation);

	void forward_activation_layer(layer l, network net);
	void backward_activation_layer(layer l, network net);
#ifdef __cplusplus
}
#endif