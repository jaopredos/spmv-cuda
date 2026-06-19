#ifndef CSR_H
#define CSR_H

#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Sparse matrix in Compressed Sparse Row (CSR) format.
 *
 *   values  : non-zero values, left-to-right, top-to-bottom
 *   col_idx : column index of each entry in values[]
 *   row_ptr : row_ptr[i]..row_ptr[i+1]-1 is the range in values[] for row i
 *             row_ptr has (rows+1) entries; row_ptr[rows] == nnz
 */
typedef struct {
    int     rows;
    int     cols;
    int     nnz;
    double *values;
    int    *col_idx;
    int    *row_ptr;
} CSRMatrix;

CSRMatrix *csr_from_matrix(const Matrix *m);
void       csr_free(CSRMatrix *csr);
void       csr_print(const CSRMatrix *csr);

/* y = A * x  (sequential CPU reference) */
void csr_spmv(const CSRMatrix *A, const double *x, double *y);

/*
 * Verifica corretude de um SpMV externo contra a referência sequencial.
 * result[] é o vetor produzido pela implementação a testar.
 * Retorna 1 se correto (dentro de tol), 0 caso contrário.
 */
int csr_verify(const CSRMatrix *A, const double *x,
               const double *result, double tol);

#ifdef __cplusplus
}
#endif

#endif
