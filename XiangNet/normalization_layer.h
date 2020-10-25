#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
	typedef layer normalization_layer;

	//TODO:×¢ÊÍ
	layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa);

#ifdef __cplusplus
}
#endif