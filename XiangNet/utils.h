#pragma once
#include <stdio.h>
#include <time.h>
#include "list.h"

//读取文件错误时,直接退出程序
void file_error(char *filename);

void error(const char *s);

//申请内存失败
void malloc_error();

//去掉空白符（包括'\n','\t',' '）
void strip(char *s);

char *fgetl(FILE *fp);