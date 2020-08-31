#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
void file_error(char * filename)
{
	printf("Could not open file:%s\n", filename);
	exit(0);
}

void error(const char *s)
{
	perror(s);
	assert(0);
	exit(-1);
}

void malloc_error()
{
	fprintf(stderr, "Malloc error\n");
	exit(-1);
}

void strip(char *s)
{
	size_t i;
	size_t len = strlen(s);
	size_t offset = 0;
	for (i = 0; i < len; ++i) {
		char c = s[i];

		// offset为要剔除的字符数，比如offset=2，说明到此时需要剔除2个空白符，
		// 剔除完两个空白符之后，后面的要往前补上，不能留空
		if (c == ' ' || c == '\t' || c == '\n') ++offset;
		else s[i - offset] = c;   // 往前补上
	}

	// 依然在真正有效的字符数组最后紧跟一个terminating null-characteristic '\0'
	s[len - offset] = '\0';
}

//每次读取文件中的某一行
// return 
char * fgetl(FILE * fp)
{
	//eof为文件流的结尾标识符
	//如果是空文件,则直接返回空
	if (feof(fp))
	{
		return 0;
	}

	//默认一行512个字符数,初始动态分配的大小,后续如果超过这个大小,则扩容
	size_t size = 512;
	char *line = malloc(size * sizeof(char));
	
	//读取一行中size个字符,如果size大于这一行的所有字符数
	// 比如这一行是10个字符,那么line中前10个元素就是10个字符
	// 然后会在后面插入换行符'\n'(如果是eof也会插入)
	// 然后还会再插入一个字符数组的终止空字符 '\0'
	// 但是如果一行字符数超过512,那么就不会插入换行符或者eof,只会插入数组终止符'\0'
	//读取数据失败,会返回空
	if (!fgets(line, size, fp))
	{
		free(line);
		return 0;
	}

	//fgets总结:都会在最后一位插入'\0'
	//如果超过512(包含'\0',比如512位有效字母加上'\0',就是513,就满足这种情况),则倒数第二位不是换行符或者eof
	//如果没有超过512则倒数第二位不是换行符(或者eof)

	//计算字符数组的长度,注意换行符 '\n' 和文件终止符 eof 也包含在内,但是数组终止符'\0'不算在内
	size_t curr = strlen(line);

	//如果一行字符数超过512,则动态扩展数组内存
	//看上面的注释,了解fgets的输出字符数组的规则后
	// 如何区分fgets获取的line是否内存不够了呢?
	// 就看最后一个字符是不是换行符或者文件终止符,
	// 如果都不是则正面当前一行的字符数超过了512,需要扩展内存
	while ((line[curr - 1] != '\n') && !feof(fp))
	{
		//超过的时候下面条件必定成立
		if (curr == size - 1)
		{
			size *= 2;

			line = realloc(line, size * sizeof(char));

			//分配内存失败
			if (!line)
			{
				printf("%ld\n", size);
				malloc_error();
			}
		}

		size_t read_size = size - curr;
		if (read_size > INT_MAX)
		{
			read_size = INT_MAX - 1;
		}
		//这里需要传地址进去,所以用取地址符号
		fgets(&line[curr], read_size, fp);
		curr = strlen(line);
	}

	//去掉换行符
	if (line[curr - 1] == '\n')
	{
		line[curr - 1] = '\0';
	}

	return line;
}
