#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/csr.h"

CSRMatrix *csr_from_matrix(const Matrix *m) {
    if (!m) return NULL;

    /* contagem de não-zeros */
    int nnz = 0;
    for (int i = 0; i < m->rows * m->cols; i++)
        if (m->data[i] != 0.0) nnz++;

    CSRMatrix *csr = malloc(sizeof(CSRMatrix));
    if (!csr) return NULL;

    csr->rows    = m->rows;
    csr->cols    = m->cols;
    csr->nnz     = nnz;
    csr->values  = malloc(nnz  * sizeof(double));
    csr->col_idx = malloc(nnz  * sizeof(int));
    csr->row_ptr = malloc((m->rows + 1) * sizeof(int));

    if (!csr->values || !csr->col_idx || !csr->row_ptr) {
        csr_free(csr);
        return NULL;
    }

    int k = 0;
    for (int i = 0; i < m->rows; i++) {
        csr->row_ptr[i] = k;
        for (int j = 0; j < m->cols; j++) {
            double v = matrix_get(m, i, j);
            if (v != 0.0) {
                csr->values[k]  = v;
                csr->col_idx[k] = j;
                k++;
            }
        }
    }
    csr->row_ptr[m->rows] = nnz;

    return csr;
}

void csr_free(CSRMatrix *csr) {
    if (!csr) return;
    free(csr->values);
    free(csr->col_idx);
    free(csr->row_ptr);
    free(csr);
}

void csr_print(const CSRMatrix *csr) {
    printf("CSR %dx%d  nnz=%d\n", csr->rows, csr->cols, csr->nnz);

    printf("row_ptr : ");
    for (int i = 0; i <= csr->rows; i++) printf("%d ", csr->row_ptr[i]);
    printf("\n");

    printf("col_idx : ");
    for (int k = 0; k < csr->nnz; k++) printf("%d ", csr->col_idx[k]);
    printf("\n");

    printf("values  : ");
    for (int k = 0; k < csr->nnz; k++) printf("%.4f ", csr->values[k]);
    printf("\n");
}

/* y = A * x — referência sequencial na CPU */
void csr_spmv(const CSRMatrix *A, const double *x, double *y) {
    for (int i = 0; i < A->rows; i++) {
        double acc = 0.0;
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++)
            acc += A->values[k] * x[A->col_idx[k]];
        y[i] = acc;
    }
}

/*
 * Compara result[] com a saída da referência sequencial.
 * Usa erro relativo por elemento; fallback para erro absoluto quando
 * o valor de referência é muito pequeno (< 1e-12).
 */
int csr_verify(const CSRMatrix *A, const double *x,
               const double *result, double tol) {
    double *ref = malloc(A->rows * sizeof(double));
    if (!ref) return 0;

    csr_spmv(A, x, ref);

    int ok = 1;
    for (int i = 0; i < A->rows; i++) {
        double diff = fabs(result[i] - ref[i]);
        double base = fabs(ref[i]);
        double err  = (base > 1e-12) ? (diff / base) : diff;

        if (err > tol) {
            fprintf(stderr,
                    "VERIFICACAO FALHOU linha %d: ref=%.10f  got=%.10f  err=%.2e\n",
                    i, ref[i], result[i], err);
            ok = 0;
        }
    }

    free(ref);
    return ok;
}
