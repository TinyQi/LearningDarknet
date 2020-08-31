#include<stdio.h>
extern void train_detector(char *datacfg,char *cfg,char *weightfile, int *gpus, int ngpus, int clear);

int main(int argc,char **argv)
{
	//训练参数
	// 1. train或者detect(象征着这是训练模式还是检测模式)
	// 2. data文件的路径
	// 3. cfg文件的路径
	// 4. 权重文件的路径,没有可以不填
	// VS调试下,直接在代码里编辑,不在命令行输入
	char **temp_argv[3];
	temp_argv[0] = "train";
	temp_argv[1] = "C:/Users/Administrator/Desktop/darknet_myself/KD.data";
	temp_argv[2] = "C:/Users/Administrator/Desktop/darknet_myself/yolov3-tiny.cfg";
	
	//暂时不给预训练权重
	temp_argv[3] = "C:/Users/Administrator/Desktop/darknet_myself/yolov3-tiny.weights";

	int gpus[1] = { 1 };

	train_detector(temp_argv[1], temp_argv[2], temp_argv[3], gpus,1,0);

	return 0;
}