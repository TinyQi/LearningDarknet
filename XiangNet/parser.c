#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#include "parser.h"
#include "list.h"
#include "utils.h"
#include "option_list.h"
#include "cuda.h"

//�����������֮��ĳнӱ������м����
typedef struct size_param
{
	int batch;
	int inputs;
	int h, w, c;
	int index;
	int time_steps;
	network net;
}size_param;

learning_rate_policy get_policy(char *s)
{
	if (strcmp(s, "random") == 0)      return RANDOM;
	if (strcmp(s, "poly") == 0)        return POLY;
	if (strcmp(s, "constant") == 0)    return CONSTANT;
	if (strcmp(s, "step") == 0)        return STEP;
	if (strcmp(s, "exp") == 0)         return EXP;
	if (strcmp(s, "sigmoid") == 0)     return SIG;
	if (strcmp(s, "steps") == 0)       return STEPS;

	fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
	return CONSTANT;
}

//��������ͨ�ò���,��ʵ������option�и����ڵ���бȶ�,����������ļ������������,��ô�Ͷ�����,�������Ĭ��ֵ
void parse_net_option(list *options, network *net)
{
	//�����������batch����Ϊ�����һ��batch������,�����������batch ����subdivisions,����Ϊ��ʵ��batch
	net->batch = option_find_int(options, "batch", 1);
	net->learning_rate = option_find_float(options, "learning_rate", 0.001);
	net->momentum = option_find_float(options, "momentum", 0.9);
	net->decay = option_find_float(options, "decay", 0.0001);
	int subdivs = option_find_int(options, "subdivisions", 1);
	net->time_steps = option_find_int(options, "time_steps", 1);
	net->notruth = option_find_int(options, "notruth", 0);

	//�������õ�����ѵ��ʱ��ʵ��batch����
	net->batch /= subdivs;
	net->batch *= net->time_steps;
	net->subdivisions = subdivs;

	//todo:
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

	// һ������ͼƬ��Ԫ�ظ�����������������ļ�û��ָ������Ĭ��ֵΪnet->h * net->w * net->c
	net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
	net->max_crop = option_find_int_quiet(options, "max_crop", net->w * 2);
	net->min_crop = option_find_int_quiet(options, "min_crop", net->w);
	net->center = option_find_int_quiet(options, "center", 0);

	net->angle = option_find_float_quiet(options, "angle", 0);
	net->aspect = option_find_float_quiet(options, "aspect", 1);
	net->saturation = option_find_float_quiet(options, "saturation", 1);
	net->exposure = option_find_float_quiet(options, "exposure", 1);
	net->hue = option_find_float_quiet(options, "hue", 0);

	//�����ص�����ͼƬ��Ϣ��Ϊ0��,ֱ���˳�����
	if (!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

	//����ע�����Ҵ�ĳ�������ļ��︴�Ƶ�,��������������Ϊ��Ҫ�Ĳ���������
	//learning_rate = 0.001                 //��ʼѧϰ��
	//burn_in = 1000                        //��ѧϰ��batch������û����burn_in��ֵ,��һ�׶δ�0(����0)��ʼ����,��burn_in�ε�ʱ��,rate�������Ԥ���learning_rate
	//max_batches = 500200                  //���ѧϰmax_batches��
	//policy = steps                        //����ѧϰ�ʵķ���,�кܶ෽������CONSTANT, STEP, EXP, POLY��STEPS, SIG, RANDOM,yolo2�õ���steps
	//steps = 400000, 450000                //steps�ĵ������Ծ���ѧϰbatch��������һ��ֵ�͵���һ��,Ȼ��steps�����������������ÿ���������е�һ��ֵ�͵���һ��
	//scales = .1, .1                       //����steps����,steps�ĵ�һ�ε����õ������ű���,�����������ĵ�һ��ֵ

	//��ȡ����ѧϰ�ʵķ���ö���ַ���
	char *policy_s = option_find_str(options, "policy", "constant");
	net->policy = get_policy(policy_s);
	net->burn_in = option_find_int_quiet(options, "burn_in", 0);
	net->power = option_find_float_quiet(options, "power", 4);

	//������Ǹ��ݵ���ѧϰ�ʵķ���,�����Ի��ػ����ѧϰ�ʵ��������
	//���ڲ�ͬ�ĵ���ѧϰ�ʵķ���,��ϸ���Կ�����,���������ҵ�,���ܲ��Ǻ�����:https://blog.csdn.net/u014090429/article/details/103736594
	if (net->policy == STEP)
	{
		//��Ϊֻ��������,����ֻ��Ҫһ�����,ѵ��batc�����ﵽstep,����scale������ѧϰ����������
		net->step = option_find_int(options, "step", 1);
		net->scale = option_find_float(options, "scale", 1);
	}
	else if (net->policy == STEPS)
	{
		//STEPS����steps��scales��������ͬʱ����
		char *l = option_find(options, "steps");
		char *p = option_find(options, "scales");
		if (!l || !p) error("STEPS policy must have steps and scales in cfg file");

		int len = strlen(l);
		int n = 1;
		int i;
		//ȷ����Ҫ��������
		for (i = 0; i < len; ++i) {
			if (l[i] == ',') ++n;
		}
		int *steps = calloc(n, sizeof(int));
		float *scales = calloc(n, sizeof(float));
		for (i = 0; i < n; ++i) {
			//atoi,atof: ���ַ���ת��Ϊint��float,�����;�з������ַ��Ļ�,�ͻ�ֹͣ,������ ',' ����,����û��
			int step = atoi(l);
			float scale = atof(p);

			//strchr�������ܽ���:�����ڲ��� str ��ָ����ַ�����������һ�γ����ַ���һ���޷����ַ�����λ�á�
			//��������ǽ�l,p������ָ��ָ����һ��','���ź���һλ��λ��,Ҳ���Ǹ��� ',' ���ֱ�ȡֵ
			l = strchr(l, ',') + 1;
			p = strchr(p, ',') + 1;
			
			steps[i] = step;
			scales[i] = scale;
		}
		//��ֵ������ĳ�Ա
		net->scales = scales;
		net->steps = steps;
		net->num_steps = n;
	}
	else if (net->policy == EXP)
	{
		net->gamma = option_find_float(options, "gamma", 1);
	}
	else if (net->policy == SIG)
	{
		net->gamma = option_find_float(options, "gamma", 1);
		net->step = option_find_int(options, "step", 1);
	}
	else if (net->policy == POLY|| net->policy == RANDOM)
	{
		//todo:Դ�벻������,Ŀǰ��֪��Ϊɶ������,�����ٿ�
	}
	net->max_batches = option_find_int(options, "max_batches", 0);

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

	//��ȡ�����ļ���һ���ֵĽڵ�,���涼�������ͨ�ò���
	node *n = sections->front;
	if (!n)
	{
		error("config file has no section");
	}
	network net = make_network(n - 1);
	
	//�����Կ���
	//��ʱ����ͬgpu
	net.gpu_index = gpu_index;

	size_param param;
	
	//��ȡ��ǰ��ı���
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
			//�����ſ�ͷ,�������������Ϣ��,���Ǿ����ĳһ��
			//���Ծ�Ҫ�����ڴ�,�����һ�����е����õ�һ�� section��,�Ա�ŵ� list sections��
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
