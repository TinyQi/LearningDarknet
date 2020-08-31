#pragma once
//列表类

//C语言里typedef就是起别名,将struct node组合在一起起一个新的名字,就叫node
// C的结构体声明的时候跟C++不一样,必须加一个struct,所以为了省略这个struct,就会视野typedef
// 我要尽量避开C++的使用习惯,多使用C的习惯
typedef struct node
{
	void *val;
	struct node *next;//下一个节点,这里还要使用struct关键字来声明,应该是因为在这个作用域内,typedef还没有起到作用,下面list结构体中的声明就可以使用别名node了
	struct node *prev;//上一个节点
}node;

typedef struct list
{
	int size;
	node *front;//第一个
	node *back;//最后一个节点
}list;

list *make_list();

int list_find(list *l, void *val);

//在链表插入元素
void list_insert(list *l,void*val);

void free_list(list *l);

void **list_to_array(list *l);
