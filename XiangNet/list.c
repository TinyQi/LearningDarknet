#include "list.h"
#include <stdlib.h>
list * make_list()
{
	list *l = malloc(sizeof(list));
	l->back = 0;
	l->front = 0;
	l->size = 0;
	return l;
}
int list_find(list * l, void * val)
{
	return 0;
}

//在链表插入元素
// 我真的是习惯使用驼峰命名,为了后续与源码对照,还是按着源码的习惯来吧,囧
void list_insert(list * l, void * val)
{
	node *new_node = malloc(sizeof(node));
	new_node->val = val;
	new_node->next = 0;
	//如果list本身就是空的
	if (!l->back)
	{
		l->front = new_node;
		new_node->prev = 0;

	}
	else
	{
		new_node->prev = l->back;
		l->back->next = new_node;
	}
	l->back = new_node;
	l->size++;
}

void free_node(node *n)
{
	//free之后,就无法访问n了,所以这里用变量提前接住n->next
	node *next;
	while (n)
	{
		next = n->next;

		free(n);
		n = next;
	}
}

void free_list(list * l)
{
	free_node(l->front);
	free(l);
}

//将list中每一个node中的val取出来,存到二位数组里
// 这里的二维数组,第一维就是每一行都是路径,第二维就是有多少这样的行,如果最后强转为char ** ,就可以理解成vector<string>
void ** list_to_array(list *l)
{
	void **a = calloc(l->size, sizeof(void*));
	
	int count = 0;

	node *n = l->front;
	while (n)
	{
		a[count] = n->val;
		count++;
		n = n->next;
	}

	return a;
}




