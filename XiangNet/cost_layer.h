#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
	typedef layer cost_layer;

	COST_TYPE get_cost_type(char *s);

	char *get_cost_string(COST_TYPE a);

	cost_layer make_cost_layer(int batch, int inputs, COST_TYPE type, float scale);

	void forward_cost_layer(const cost_layer l, network net);
	void backward_cost_layer(const cost_layer l, network net);
#ifdef __cplusplus
}
#endif