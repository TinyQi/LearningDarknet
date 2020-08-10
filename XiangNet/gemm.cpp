#include "gemm.h"

//C += ALPHA * A * B 
//A,B������ת��
//todo:darknetʹ�� lda,ldb,ldc ����������,��Ŀǰ������ A_C_row,B_C_col,A_col_B_row�͹��˰�,�������Ե�ʱ���ٿ�
void gemmNN( int A_C_row, int B_C_col, int A_col_B_row,
	float *A, float *B, float *C, int lda, int ldb, int ldc,int alpha)
{
	//����A����,C����
	for (int i = 0; i < A_C_row; i++)
	{
		//����A����,B����
		for (int k = 0; k < A_col_B_row; k++)
		{
			//A�����е�i�еĵ�k�е�Ԫ��
			register float A_PART = alpha * A[i * lda + k];
			for (int j = 0; j < B_C_col; j++)
			{
				//ÿ�ε���j,�ͼ���C������,��i��,��j�еĽ���е�һ����,�ȵ��в�ѭ������,�������A��һ����B�������е�����
				C[i * ldc + j] += A_PART * B[k * ldb + j];
			}
		}
	}
}

//C += ALPHA * A * B' 
//A��ת��,Bת��
void gemmNT(int A_C_row, int B_C_col, int A_col_B_row,
	float *A, float *B, float *C, int lda, int ldb, int ldc, int alpha)
{
	//��ѭ����A������
	for (int i = 0; i < A_C_row; i++)
	{
		// ע�⿴�������ѭ����B_C_col������gemmNN������ѭ����A_col_B_row
		// Ϊʲô��?
		// ��ʵѭ���ĵ�����ʽ�������,��һ���������,�в��ǵ�һ���������,Ȼ���ڲ�ѭ���ǵڶ����������
		// ����˳���gemmNN()���治һ������ʵ����Ϊ���洫������ʱ��,����ֵ���෴��,
		// �������������ϵĵ���˳����һ����,ֻ���������洫������ʱ��˳����,������ѭ���ڲ���������Ϊ����
		// !!!!!����ô�������Ǵ����,��Ҫ��˼��
		// Ŀǰ�����޸����������˳��Ŀ�ľ���Ϊ���ڴ�����,�ӿ��ȡ�ٶ�(�о�Ҳ����)
		// Ӧ���Ǿ͵���ôʵ��,����һά������ڴ�ṹ������
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
//Aת��,B��ת��
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
//A��ת��,Bת��
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

// �ܵ���˵����Ҫʵ�� C = ALPHA * A * B + BETA * C
// A_C_row,A����A'������(���ֵ��������ת�û��ǲ�ת��,�������������ĵ�һ�����������),ͬʱ�϶�����C������,���ֵ����ͬһ����������ת�õ�ʱ���ǻ���,��ֵ����,����û��
// B_C_col,B����B'������(���ֵ��������ת�û��ǲ�ת��,�������������ĵڶ������������),ͬʱ�϶���C������,���ֵ����ͬһ����������ת�õ�ʱ���ǻ���,��ֵ����,����û��
// A_col_B_row,A���л���B����,�������A����B��ת�õĻ�,Ҳ�Ƕ�Ӧ��ת�ú��ֵ
// lda     A������������ת�ã�������������ת�ã��Ҹ�����ת�ú�A��A'��������,���ֵ�Ͳ����,ת��ǰ���о���ת�ú����,��ֵû��,�������
// ldb     B������������ת�ã�������������ת�ã��Ҹ�����ת�ú�B��B'��������,���ֵ�Ͳ����,ת��ǰ���о���ת�ú����
// ldc     C������
// ΪʲôҪ����������������?
// ��Ϊ��������forѭ���Ĺ���Ҫ����A_C_row,B_C_col�����ƶ�,��������������ʱ��,��Ҫ������ʵ�����ݴ洢����(����lda,ldb,ldc)��׼ȷ��ȡ����Ӧ������
void gemm(bool isA_Transpose, bool isB_Transpose, int A_C_row, int B_C_col, int A_col_B_row,
	float *A, float *B, float *C, int lda, int ldb, int ldc,int alpha ,int beta)
{
	//���� BETA * C �����A��Bû�й�ϵ,������������ǰ������
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