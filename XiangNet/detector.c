#include <stdlib.h>
#include <stdio.h>


#include "option_list.h"
#include "utils.h"
#include "network.h"
#include "parser.h"
#include "data.h"

//目前先实现cpu上的计算
void train_detector(char *datacfg, char *cfg, char *weightfile, int *gpus, int ngpus, int clear)
{
	list *options = read_data_cfg(datacfg);
	
	char *train_image = option_find_str(options, "train", "data/train.list");
	char *backup_directory = option_find_str(options, "backup", "/backup/");

	printf_data_list(options);

	//C语言基础:结构体也只是把一堆变量按顺序排在内存中.地址是连续的
	network *nets = calloc(ngpus, sizeof(network));
	
	//printf("nets：OX%p\n", nets);



	int i = 0;
	for ( i = 0; i < ngpus; i++)
	{
		//解析配置并初始化各个层
		nets[i] = parse_network_cfg(cfg);
		if (weightfile)
		{
			//如果有预训练或者训练到一半的权重文件,则可以载入接下去训练
			load_weights(&nets[i], weightfile);
		}

		//todo::clear是什么意思
		if (clear)
		{
			//todo
			*nets[i].seen = 0;
		}

		//todo
		nets[i].learning_rate *= ngpus;
	}

	network net = nets[0];

	//todo:这个变量好像没用
	int imgs = net.batch * net.subdivisions *ngpus;
	printf("Learning rate: %g, Momentum: %g, Decay: %g", net.learning_rate, net.momentum, net.decay);

	//train:训练图片,buffer:缓存,类似于平时用的temp
	data train, buffer;

	//获取网络最后一层
	layer l = net.layers[net.n - 1];
	int classes = l.classes;
	float jitter = l.jitter;

	list *plist = get_path(train_image);

	char **path = list_to_array(plist);


	int a = 0;
}