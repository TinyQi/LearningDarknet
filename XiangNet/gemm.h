#pragma once
//这里的A_C_row或者B_C_col,里面的A,B,C可能是转置或者没转置
// **        A,B,C   输入矩阵（一维数组格式）
// **        ALPHA   系数
// **        BETA    系数
// **        lda     A的列数（不做转置）或者行数（做转置，且给的是转置后A即A'的行数）
// **        ldb     B的列数（不做转置）或者行数（做转置，且给的是转置后B即B'的行数）
// **        ldc     C的列数
void gemm(bool isA_Transpose, bool isB_Transpose, int A_C_row, int B_C_col, int A_col_B_row,
	float *A, float *B, float *C, int lda, int ldb, int ldc, int alpha, int beta);