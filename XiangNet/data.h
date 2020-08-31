#ifndef DATA_H
#define DATA_H

#include "matrix.h"
#include "box.h"
#include "list.h"
typedef struct data {
	int w, h;
	matrix X;
	matrix Y;

	//todo
	int shallow;
	int *num_boxes;
	box **boxes;
}data;



list *get_path(char *filename);







#endif // !DATA_H

