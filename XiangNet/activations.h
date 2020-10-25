#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

/*
**  ���еļ�������ö���ࣩ
*/
typedef enum {
	LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
}ACTIVATION;

ACTIVATION get_activation(char *s);

void activate_array(float *x, const int n, const ACTIVATION a);

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);
#endif // !ACTIVATIONS_H



