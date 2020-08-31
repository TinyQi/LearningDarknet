#include "network.h"
#include <stdlib.h>
network make_network(int n)
{
	network net = { 0 };
	net.n = n;
	net.layers = calloc(net.n, sizeof(layer));
	net.seen = calloc(1, sizeof(int));
	net.cost = calloc(1, sizeof(float));

	return net;
}
