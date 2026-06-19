#ifndef SPMV_CPU_H
#define SPMV_CPU_H

#include "csr.h"
#include "sp24.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --- CSR --- */

/* Referência sequencial — usada como ground-truth na verificação */
void spmv_cpu_sequential(const CSRMatrix *A, const double *x, double *y);

/* Paralelo com OpenMP — linhas distribuídas entre threads */
void spmv_cpu_openmp(const CSRMatrix *A, const double *x, double *y);

/* --- 2:4 Structured Sparse --- */

void spmv_cpu_sp24_sequential(const SP24Matrix *A, const double *x, double *y);
void spmv_cpu_sp24_openmp(const SP24Matrix *A, const double *x, double *y);

#ifdef __cplusplus
}
#endif

#endif
