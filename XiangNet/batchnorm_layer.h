#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif

	//TODO:×¢ÊÍ
	layer make_batchnorm_layer(int batch, int w, int h, int c);
	void forward_batchnorm_layer(layer l, network net);
	void backward_batchnorm_layer(layer l, network net);
#ifdef __cplusplus
}
#endif
