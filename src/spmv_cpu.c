#include <omp.h>
#include "../include/spmv_cpu.h"

/* ------------------------------------------------------------------ */
/* CSR                                                                  */
/* ------------------------------------------------------------------ */

void spmv_cpu_sequential(const CSRMatrix *A, const double *x, double *y) {
    for (int i = 0; i < A->rows; i++) {
        double acc = 0.0;
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++)
            acc += A->values[k] * x[A->col_idx[k]];
        y[i] = acc;
    }
}

void spmv_cpu_openmp(const CSRMatrix *A, const double *x, double *y) {
    /* dynamic: linhas têm nnz variável; distribui sob demanda */
    #pragma omp parallel for schedule(dynamic) default(none) shared(A, x, y)
    for (int i = 0; i < A->rows; i++) {
        double acc = 0.0;
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++)
            acc += A->values[k] * x[A->col_idx[k]];
        y[i] = acc;
    }
}

/* ------------------------------------------------------------------ */
/* 2:4 Structured Sparse                                                */
/* ------------------------------------------------------------------ */

void spmv_cpu_sp24_sequential(const SP24Matrix *A, const double *x, double *y) {
    int groups = A->cols_padded / 4;
    for (int i = 0; i < A->rows; i++) {
        double acc = 0.0;
        for (int g = 0; g < groups; g++) {
            int vi = i * (A->cols_padded / 2) + g * 2;
            int mi = i * (A->cols_padded / 4) + g;
            int c0 = g * 4 + (A->metadata[mi] & 0x3);
            int c1 = g * 4 + ((A->metadata[mi] >> 2) & 0x3);
            acc += A->values[vi]     * x[c0];
            acc += A->values[vi + 1] * x[c1];
        }
        y[i] = acc;
    }
}

void spmv_cpu_sp24_openmp(const SP24Matrix *A, const double *x, double *y) {
    int groups = A->cols_padded / 4;
    /* static: todas as linhas têm exatamente o mesmo número de grupos */
    #pragma omp parallel for schedule(static) default(none) shared(A, x, y, groups)
    for (int i = 0; i < A->rows; i++) {
        double acc = 0.0;
        for (int g = 0; g < groups; g++) {
            int vi = i * (A->cols_padded / 2) + g * 2;
            int mi = i * (A->cols_padded / 4) + g;
            int c0 = g * 4 + (A->metadata[mi] & 0x3);
            int c1 = g * 4 + ((A->metadata[mi] >> 2) & 0x3);
            acc += A->values[vi]     * x[c0];
            acc += A->values[vi + 1] * x[c1];
        }
        y[i] = acc;
    }
}
