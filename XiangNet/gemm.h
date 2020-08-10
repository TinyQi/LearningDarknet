#pragma once
//�����A_C_row����B_C_col,�����A,B,C������ת�û���ûת��
// **        A,B,C   �������һά�����ʽ��
// **        ALPHA   ϵ��
// **        BETA    ϵ��
// **        lda     A������������ת�ã�������������ת�ã��Ҹ�����ת�ú�A��A'��������
// **        ldb     B������������ת�ã�������������ת�ã��Ҹ�����ת�ú�B��B'��������
// **        ldc     C������
void gemm(bool isA_Transpose, bool isB_Transpose, int A_C_row, int B_C_col, int A_col_B_row,
	float *A, float *B, float *C, int lda, int ldb, int ldc, int alpha, int beta);