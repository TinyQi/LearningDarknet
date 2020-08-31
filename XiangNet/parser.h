#ifndef PARSER_H
#define PARSER_H

#include "network.h"
#include "list.h"

//用于保存每一层的配置文件
typedef struct section
{
	char *type;
	list *options;
}section;

network parse_network_cfg(char *filename);


list *read_cfg(char *filename);


void load_weights(network *net, char *filename);
#endif