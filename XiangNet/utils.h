#pragma once
#include <stdio.h>
#include <time.h>
#include "list.h"

#define TWO_PI 6.2831853071795864769252866

//读取文件错误时,直接退出程序
void file_error(char *filename);

void error(const char *s);

//申请内存失败
void malloc_error();

//去掉空白符（包括'\n','\t',' '）
void strip(char *s);

char *fgetl(FILE *fp);

//产生符合正太分布的随机数
float rand_normal();

int *read_map(char *filename);

float rand_uniform(float min, float max);

float mag_array(float *a, int n);