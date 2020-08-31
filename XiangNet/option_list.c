#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "option_list.h"
#include "utils.h"
#include "parser.h"

list * read_data_cfg(char * filename)
{
	//C风格的文件处理,以只读模式打开文件
	FILE *file = fopen(filename, "r");
	if (file == 0)
	{
		file_error(filename);
	}

	list *option = make_list();
	int nu = 0;
	char *line;

	//每次读入文件流中的一行
	//这个循环怎么循环的?因为在fgetl这个函数里,file这个流在函数执行完之后会指向下一行的位置
	while ((line = fgetl(file)) != 0)
	{
		//fget()
		//解释:但是需要注意的是，如果输入的字符串长度没有超过 n–1，那么系统会将最后输入的换行符 '\n' 保存进来
		//也就是说,如果我们预设读取512长度的字符,然后就会有两种情况,第一种就是这一行没有超过512,另一种就是超过512
		// 我们需要区分这两种情况以做动态扩展
		// 如何区分?
		// 首先不管超了还是没超,都会在最后一个元素插入'\0'元素,但是如果没有超过512就会把换行符'\n'保存进来,所以就判断字符数组最后一个元素(不包含最后的'\0'空字符)是不是'\n'就能判断是否已经读完一行
		nu++;
		//去空白字符
		strip(line);
		
		//根据每一行的第一个元素来判断是否是有效行,因为有一些行是需要忽略的,比如注释之类的
		//惭愧,C语言基础需要加强,switch后面的case不像if,当某一个case判断成功后,就会依次执行下面的case语句,直到遇到break
		//在这里的输出语句可以看出
		//如果遇到空格符 '\0' ,则会输出11111111 22222222 33333333,然后break
		//如果遇到'#' ,则只会输出22222222 33333333,然后break
		//所以这里的意思就是遇到'\0', '#' , ';' ,这三个字符都会释放当前line指针的内存,并且退出(break)switch
		switch (line[0])
		{
			//遇到下面3个字符在首位则跳过
			case '\0':
				printf("11111111\n");
			case '#':
				printf("22222222\n");
			case ';':
				printf("33333333\n");
				free(line);//遇到分号直接退出循环
				break;
			default:
				//默认情况是读取配置
				if (!read_option(line, option))
				{
					//输出错误信息,包括行数和读到的这一行的信息
					fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
					free(line);
				}
				break;
		}
	
	
	
	}
	//关闭文件流
	fclose(file);
	return option;
}

//读取s中的命令,并插入到链表中(比如: kernel = 3)
int read_option(char * s, list * option)
{
	size_t len = strlen(s);
	char *val = 0;
	size_t i = 0;
	for ( ;i < len; i++)
	{
		if (s[i] == '=')
		{
			//将等号用终止符替换,这样s指针指向的字符串就会在'='处被截断
			s[i] = '\0';
			//将'='后面的字符串赋给val,作为值
			val = &s[i + 1];
			break;
		}
	}

	//一个有效行都没找到 '=' 说明配置文件出问题了
	if (i == len - 1)
	{
		return 0;
	}
	
	char *key = s;
	
	option_insert(option, key, val);
	
	return 1;
}

//将一组值组成键值对插入list中
void option_insert(list * l, char * key, char * val)
{
	kvp *p = malloc(sizeof(kvp));

	p->key = key;
	p->val = val;
	p->used = 0;

	list_insert(l, p);
}

char * option_find(list * l, char * key)
{
	node *n = l->front;

	while (n)
	{
		kvp	*p = (kvp*)n->val;
		if (strcmp(p->key, key) == 0)
		{
			//当前键值对被查找过一次,就代表是有用的参数
			p->used = 1;
			return p->val;
		}
		n = n->next;
	}
	//没找到返回0指针,后面调用这个函数的时候需要针对返回0指针的容错处理
	return 0;
}

char * option_find_str(list * l, char * key, char * def)
{
	char *val = option_find(l,key);

	//查找成功,则返回字符串val
	if (val)return val;

	//查找失败,则使用默认值
	if (def)
		fprintf(stderr, "%s:Using default '%s'\n", key, def);
	return def;
}

int option_find_int(list * l, char * key, int def)
{
	char *val = option_find(l, key);

	//查找成功,则将字符串val转成int再返回
	if (val)return atoi(val);

	//查找失败,则使用默认值
	if (def)
		fprintf(stderr, "%s:Using default '%d'\n", key, def);
	return def;
}

float option_find_float(list * l, char * key, float def)
{
	char *val = option_find(l, key);

	//查找成功,则将字符串val转成float再返回
	if (val)return atof(val);

	//查找失败,则使用默认值
	if (def)
		fprintf(stderr, "%s:Using default '%.1f'\n", key, def);
	return def;
}

//检查一遍链表中没有没使用的键值对
// 何为被使用?就是被查找(option_find)过一次
void option_unused(list * l)
{
	node *n = l->front;

	while (n)
	{
		kvp *temp = (kvp*)n->val;
		if (!temp->used)
		{
			fprintf(stderr, "Unused field:'%s = %s'\n", temp->key, temp->val);
		}
		n = n->next;
	}
}

//打印data文件链表的内容,作测试用
void printf_data_list(list * l)
{
	node *n = l->front;
	while (n)
	{
		kvp *p = (kvp*)n->val;
		printf("%s = %s\n", p->key, p->val);
		n = n->next;
	}
}

//打印cfg文件链表的内容,作测试用
void printf_cfg_list(list * l)
{
	node *n = l->front;
	while (n)
	{
		section *p_list = (section*)n->val;
		node *temp_n = p_list->options->front;
		while (temp_n)
		{
			kvp *p = (kvp*)temp_n->val;
			printf("%s = %s\n", p->key, p->val);
			temp_n = temp_n->next;
		}
		
		printf("--------------------\n");
		n = n->next;
	}
}