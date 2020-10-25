#include "img2col.h"

//#include <opencv2\core\core.hpp>
//#include <opencv2\imgproc\imgproc.hpp>
//#include <opencv2\highgui\highgui.hpp>

#include "gemm.h"

void showVec(int *data, int cols,int rows,int c)
{
	for (int i = 0; i < c; i++)
	{
		printf("--------------------------------\n");
		for (int y = 0; y < rows; y++)
		{
			for (int x = 0; x < cols; x++)
			{
				int val = data[(c - 1) * cols * rows + y * cols + x];
				printf("%d ", val);
			}
			printf("\n");
		}
	}
	
}
void showVec_float(float *data, int rows, int cols,  int c)
{
	for (int i = 0; i < c; i++)
	{
		printf("--------------------------------\n");
		for (int y = 0; y < rows; y++)
		{
			for (int x = 0; x < cols; x++)
			{
				float val = data[(c - 1) * cols * rows + y * cols + x];
				printf("%.2f ", val);
			}
			printf("\n");
		}
	}

}

//todo:这里还需要考虑pad

int getDataPixelValue(int *data, int width, int height, int x,int y,int c,int pad)
{
	//data中的真实数据并没有补0
	x -= pad;
	y -= pad;
	
	//做容错判断
	if (x<0 || x>=width || y<0 || y>=height)
		return 0;

	
	//所有的图片都是先每个通道按行抽出再排成一行,再把每个通道的一小行继续排成一大行
	//最终一个张量数据就是一个超级长的一维数组
	return data[width * height * c + y *width + x ];
}

void img2col(int *data, int *dataCol, int width, int height, int channels, int kernelSize, int stride, int pad)
{
	//计算输出图的宽高
	int out_w = (width + pad * 2 - kernelSize) / stride + 1;
	int out_h = (height + pad * 2 - kernelSize) / stride + 1;

	int kernel_elements = kernelSize * kernelSize * channels;

	for (int c = 0; c < kernel_elements; c++)
	{
		//下面三个值是卷积核的位置 x,y,c
		int w_offset = c % kernelSize;
		int h_offset = (c / kernelSize) % kernelSize;//还要取余是因为有多个通道
		int c_offset = c / kernelSize / kernelSize;

		for (int h = 0; h < out_h; h++)
		{
			for (int w = 0; w < out_w; w++)
			{
				//先获取原图中的像素
				// 第一步就是先找到原来的坐标
				int origin_x = w * stride + w_offset;
				int origin_y = h * stride + h_offset;

				int origin_index = (width*height)*c_offset + origin_y * width + origin_x;
				//根据坐标获取当前遍历到的像素的值
				//这里注意输入的是当前的通道数,我之前把channels这个表示该数据总共有多少个通道的值传进去,
				//导致我在写getDataPixelValue的时候发现,如果channels这个定值传入,里面计算索引的时候(width * height * c)就从一个定值开始了,肯定不对
				int pixel_val = getDataPixelValue(data, width, height, origin_x, origin_y, c_offset,pad);

				//对重拍的一维数组进行赋值
				//先获取重拍后图像数组的索引
				int data_col_index = c * out_h * out_w + h * out_w + w;

				dataCol[data_col_index] = pixel_val;

				int stop = 0;
			}
		}
	}
	
}



//void test_img2col()
//{
//	//cv::Mat test_img = cv::Mat::zeros(test_img_width_height, test_img_width_height, CV_8UC1);
//	int test_img_cols = 3;
//	int test_img_rows = 2;
//
//
//	float *data_A = (float*)calloc(test_img_cols * test_img_rows, sizeof(float));
//	float *data_B = (float*)calloc(test_img_cols * test_img_rows, sizeof(float));
//
//	//A
//	//data_A[0] = 1;
//	//data_A[1] = 2;
//	//data_A[2] = 3;
//	//data_A[3] = 4;
//	//data_A[4] = 5;
//	//data_A[5] = 6;
//
//	//A'
//	data_A[0] = 1;
//	data_A[1] = 4;
//	data_A[2] = 2;
//	data_A[3] = 5;
//	data_A[4] = 3;
//	data_A[5] = 6;
//
//	//B
//	//data_B[0] = 1;
//	//data_B[1] = 2;
//	//data_B[2] = 3;
//	//data_B[3] = 4;
//	//data_B[4] = 5;
//	//data_B[5] = 6;
//
//	//B'
//	data_B[0] = 1;
//	data_B[1] = 3;
//	data_B[2] = 5;
//	data_B[3] = 2;
//	data_B[4] = 4;
//	data_B[5] = 6;
//
//
//	//展示data数组
//	showVec_float(data_A, 3 ,2, 1);
//	showVec_float(data_B, 2 ,3, 1);
//
//	float *data_C = (float*)calloc(test_img_rows * test_img_rows, sizeof(float));
//	
//	int alpha = 1;
//	int beta = 0;
//	
//	//A*B,测试通过
//	//gemm(false, false, test_img_rows, test_img_rows, test_img_cols, data_A, data_B, data_C, test_img_cols, test_img_rows, test_img_rows, alpha,beta);
//	//showVec_float(data_C, test_img_rows, test_img_rows, 1);
//
//	//A*B',测试通过
//	//gemm(false, true, 2, 2, 3, data_A, data_B, data_C, 3, 3, 2, alpha, beta);
//	//showVec_float(data_C, test_img_rows, test_img_rows, 1);
//
//	//A'*B,测试通过
//	//gemm(true, false, 2, 2, 3, data_A, data_B, data_C, 2, 2, 2, alpha, beta);
//	//showVec_float(data_C, test_img_rows, test_img_rows, 1);
//
//	//A'*B'
//	gemm(1, 1, 2, 2, 3, data_A, data_B, data_C, 2, 3, 2, alpha, beta);
//	showVec_float(data_C, test_img_rows, test_img_rows, 1);
//
//	int asdasdasd = 0;
//
//	//int width = test_img_cols, height = test_img_cols, channels = 1;
//	//int k_size = 3;
//	//int stride = 1;
//	//int pad = 1;
//
//	////注意,这里一般是提前准备超量的内存备用,我这里是因为知道结果是多少个int,所以测试就简单点
//	//int out_w = ((test_img_cols + 2 * pad - k_size) / stride + 1);
//	//int out_h = out_w;
//	//int img_col_length = out_w * out_h * k_size * k_size;
//	//int *data_col = (int*)calloc(img_col_length, sizeof(int));
//
//	//img2col(data_A, data_col, width, height, channels, k_size, stride, pad);
//
//	//showVec(data_col, out_w*out_h, k_size * k_size, 1);
//
//	////test_img.release();
//	free(data_A);
//	free(data_B);
//	free(data_C);
//
//	//free(data_col);
//	
//}