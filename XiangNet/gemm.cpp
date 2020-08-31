#include "gemm.h"

//C += ALPHA * A * B 
//A,B都不做转置
//todo:darknet使用 lda,ldb,ldc 这三个变量,我目前觉得用 A_C_row,B_C_col,A_col_B_row就够了啊,后续测试的时候再看
void gemmNN( int A_C_row, int B_C_col, int A_col_B_row,
	float *A, float *B, float *C, int lda, int ldb, int ldc,int alpha)
{
	//遍历A的行,C的行
	for (int i = 0; i < A_C_row; i++)
	{
		//遍历A的列,B的行
		for (int k = 0; k < A_col_B_row; k++)
		{
			//A矩阵中第i行的第k列的元素
			register float A_PART = alpha * A[i * lda + k];
			for (int j = 0; j < B_C_col; j++)
			{
				//每次迭代j,就计算C矩阵中,第i行,第j列的结果中的一部分,等到中层循环结束,就完成了A中一行与B中所有列的运算
				C[i * ldc + j] += A_PART * B[k * ldb + j];
			}
		}
	}
}

//C += ALPHA * A * B' 
//A不转置,B转置
void gemmNT(int A_C_row, int B_C_col, int A_col_B_row,
	float *A, float *B, float *C, int lda, int ldb, int ldc, int alpha)
{
	//大循环是A的行数
	for (int i = 0; i < A_C_row; i++)
	{
		// 注意看这里的中循环是B_C_col而不像gemmNN里面中循环是A_col_B_row
		// 为什么呢?
		// 其实循环的迭代方式都是外层,第一个矩阵的行,中层是第一个矩阵的列,然后内层循环是第二个矩阵的列
		// 这里顺序跟gemmNN()里面不一样是其实是因为外面传进来的时候,两个值是相反的,
		// 两个函数本质上的迭代顺序是一样的,只是由于外面传进来的时候顺序反了,所以在循环内部换回来作为纠正
		// !!!!!我这么理解可能是错误的,需要再思考
		// 目前觉得修改这个迭代的顺序目的就是为了内存连续,加快读取速度(感觉也不对)
		// 应该是就得这么实现,这是一维数组的内存结构决定的
		for (int j = 0; j < B_C_col; j++)
		{
			register float sum = 0;
			for (int k = 0; k < A_col_B_row; k++)
			{
				sum += alpha * A[lda * i + k] * B[ldb * j + k];
			}
			C[ldc * i + j] += sum;
		}
	}
}

//C += ALPHA * A' * B 
//A转置,B不转置
void gemmTN(int A_C_row, int B_C_col, int A_col_B_row,
	float *A, float *B, float *C, int lda, int ldb, int ldc, int alpha)
{
	for (int i = 0; i < A_C_row; i++)
	{
		for (int k = 0; k < A_col_B_row; k++)
		{
			register float A_APRT = alpha * A[k * lda + i];
			for (int j = 0; j < B_C_col; j++)
			{
				C[ldc * i + j] += A_APRT *B[ldb * k + j];
			}
		}
	}
}

//C += ALPHA * A' * B'
//A不转置,B转置
void gemmTT(int A_C_row, int B_C_col, int A_col_B_row,
	float *A, float *B, float *C, int lda, int ldb, int ldc, int alpha)
{
	for (int i = 0; i < A_C_row; i++)
	{
		for (int j = 0; j < B_C_col; j++)
		{
			register float sum = 0;
			for (int k = 0; k < A_col_B_row; k++)
			{
				sum += alpha * A[lda * k + i] * B[ldb * j + k];
			}
			C[ldc * i + j] += sum;
		}
	}
}

// 总的来说就是要实现 C = ALPHA * A * B + BETA * C
// A_C_row,A或者A'的行数(这个值不管你是转置还是不转置,他代表进行运算的第一个矩阵的行数),同时肯定就是C的行数,这个值对于同一个矩阵遇到转置的时候是会变的,数值变了,含义没变
// B_C_col,B或者B'的列数(这个值不管你是转置还是不转置,他代表进行运算的第而个矩阵的列数),同时肯定是C的列数,这个值对于同一个矩阵遇到转置的时候是会变的,数值变了,含义没变
// A_col_B_row,A的列或者B的行,如果其中A或者B有转置的话,也是对应的转置后的值
// lda     A的列数（不做转置）或者行数（做转置，且给的是转置后A即A'的行数）,这个值就不会变,转置前的列就是转置后的行,数值没变,含义变了
// ldb     B的列数（不做转置）或者行数（做转置，且给的是转置后B即B'的行数）,这个值就不会变,转置前的列就是转置后的行
// ldc     C的列数
// 为什么要分上面两种命名呢?
// 因为矩阵计算的for循环的规则要根据A_C_row,B_C_col等来制定,而具体进行运算的时候,又要根据真实的数据存储参数(就像lda,ldb,ldc)来准确的取到相应的数据
void gemm(bool isA_Transpose, bool isB_Transpose, int A_C_row, int B_C_col, int A_col_B_row,
	float *A, float *B, float *C, int lda, int ldb, int ldc,int alpha ,int beta)
{
	//由于 BETA * C 这个跟A和B没有关系,所以这里先提前计算了
	for (int i = 0; i < A_C_row; i++)
	{
		for (int j = 0; j < B_C_col; j++)
		{
			C[i * B_C_col + j] *= beta;
		}
	}

	if (!isA_Transpose && !isB_Transpose)
		gemmNN(A_C_row, B_C_col, A_col_B_row, A, B, C, lda, ldb, ldc, alpha);
	else if (!isA_Transpose && isB_Transpose)
		gemmNT(A_C_row, B_C_col, A_col_B_row, A, B, C, lda, ldb, ldc, alpha);
	else if (isA_Transpose && !isB_Transpose)
		gemmTN(A_C_row, B_C_col, A_col_B_row, A, B, C, lda, ldb, ldc, alpha);
	else if (isA_Transpose && isB_Transpose)
		gemmTT(A_C_row, B_C_col, A_col_B_row, A, B, C, lda, ldb, ldc, alpha);
}