#include <stdlib.h>
#include <stdio.h>

#include "data.h"
#include "utils.h"


//获取图片路径文件中的每一行,保存到list中
list * get_path(char * filename)
{
	char *path;
	FILE *fp = fopen(filename, "r");
	if (!fp)
		file_error(filename);
	
	list *lines = make_list();
	
	while ((path = fgetl(fp)) != 0)
	{
		list_insert(lines, path);
	}

	fclose(fp);
	return lines;
}
