#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "parser.h"
#include "list.h"
#include "utils.h"
#include "option_list.h"
#include "cuda.h"

//网络与各个层之间的承接变量、中间变量
typedef struct size_param
{
	int batch;
	int inputs;
	int h, w, c;
	int index;
	int time_steps;
	network net;
}size_param;

//解析网络通用参数,其实就是与option中各个节点进行比对,如果在配置文件里有这个参数,那么就读进来,否则就用默认值
void parse_net_option(list *options, network *net)
{
	//这里读进来的batch是人为定义的一轮batch的数量,下面会把这里的batch 除以subdivisions,来作为真实的batch
	net->batch = option_find_int(options, "batch", 1);
	net->learning_rate = option_find_float(options, "learning_rate", 0.001);
	net->momentum = option_find_float(options, "momentum", 0.9);
	net->decay = option_find_float(options, "decay", 0.0001);
	int subdivs = option_find_int(options, "subdivisions", 1);
	net->time_steps = option_find_int(options, "time_steps", 1);
	net->notruth = option_find_int(options, "notruth", 0);

	//这里计算得到网络训练时真实的batch数量
	net->batch /= subdivs;
	net->batch *= net->time_steps;
	net->subdivisions = subdivs;


	net->adam = option_find_int(options, "adam", 0);
	if (net->adam)
	{
		net->B1 = option_find_float(options, "B1", 0.9);
		net->B2 = option_find_float(options, "B2", 0.999);
		net->eps = option_find_float(options, "eps", 0.00000001);
	}

	net->h = option_find_int(options, "height", 0);
	net->w = option_find_int(options, "width", 0);
	net->c = option_find_int(options, "channels", 0);
}

int is_network(section *s)
{
	return (strcmp(s->type, "[net]") == 0) || 
		(strcmp(s->type, "[network]") == 0);
}

network parse_network_cfg(char * filename)
{
	list *sections = read_cfg(filename);
	printf_cfg_list(sections);

	//读取配置文件第一部分的节点,里面都是网络的通用参数
	node *n = sections->front;
	if (!n)
	{
		error("config file has no section");
	}
	network net = make_network(n - 1);
	
	//所用显卡号
	//暂时还不同gpu
	net.gpu_index = gpu_index;

	size_param param;
	
	//提取当前层的变量
	section *s = (section*)n->val;
	list *option = s->options;
	if (!is_network(s)) error("first section must be [net] or [network]");
	
	parse_net_option(option, &net);
	


	return net;
}

list * read_cfg(char * filename)
{
	FILE *file = fopen(filename,"r");
	if (file == 0)
		file_error(filename);
	
	char *line;
	int nu = 0;

	list *sections = make_list();

	section *current = 0;

	while ((line = fgetl(file)) != 0)
	{
		nu++;
		strip(line);
		switch (line[0])
		{
		case '[' :
			//中括号开头,不是网络基础信息层,就是具体的某一层
			//所以就要申请内存,存放这一层所有的配置到一个 section里,以便放到 list sections里
			current = malloc(sizeof(section));
			list_insert(sections, current);

			current->options = make_list();
			current->type = line;
			break;
		case '\0':
		case '#' :
		case ';':
			free(line);
			break;
		default:
			if (!read_option(line,current->options))
			{
				fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
				free(line);
			}
			break;
		}
		
	}

	fclose(file);
	return sections;
}

void load_weights(network * net, char * filename)
{
	//todo
}
