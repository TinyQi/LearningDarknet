#pragma once
#include "list.h"

//键值对类
//内含两个字符数组指针
typedef struct kvp
{
	char *key;
	char *val;
	int used;
}kvp;

//读取cfg配置文件到list中
list *read_data_cfg(char *filename);

int read_option(char *s, list *option);

void option_insert(list *l, char *key, char *val);

char *option_find(list *l, char *key);

char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
float option_find_float(list *l, char *key, float def);

char *option_find_str_quiet(list *l, char *key, char *def);
int option_find_int_quiet(list *l, char *key, int def);
float option_find_float_quiet(list *l, char *key, float def);

void option_unused(list *l);


void printf_data_list(list *l);

void printf_cfg_list(list * l);
