#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../include/macko.h"

MACKOMatrix *macko_from_csr(const CSRMatrix *csr) {
    if (!csr) return NULL;

    int rows = csr->rows;

    int total_chunks = 0;
    for (int i = 0; i < rows; i++) {
        int row_nnz = csr->row_ptr[i + 1] - csr->row_ptr[i];
        total_chunks += (row_nnz + MACKO_CHUNK_SIZE - 1) / MACKO_CHUNK_SIZE;
    }

    MACKOMatrix *A = malloc(sizeof(MACKOMatrix));
    if (!A) return NULL;

    A->rows         = rows;
    A->cols         = csr->cols;
    A->nnz          = csr->nnz;
    A->total_chunks = total_chunks;
    A->valid_masks  = calloc(total_chunks, sizeof(uint8_t));
    A->col_bases    = malloc(total_chunks * sizeof(int16_t));
    A->col_deltas   = calloc((long)total_chunks * MACKO_CHUNK_SIZE, sizeof(int16_t));
    A->vals         = calloc((long)total_chunks * MACKO_CHUNK_SIZE, sizeof(double));
    A->row_ptr      = malloc(rows * sizeof(int));
    A->row_nchunks  = malloc(rows * sizeof(int));

    if (!A->valid_masks || !A->col_bases || !A->col_deltas ||
        !A->vals || !A->row_ptr || !A->row_nchunks) {
        macko_free(A);
        return NULL;
    }

    int c = 0;
    for (int i = 0; i < rows; i++) {
        int row_start = csr->row_ptr[i];
        int row_nnz   = csr->row_ptr[i + 1] - row_start;
        int nchunks   = (row_nnz + MACKO_CHUNK_SIZE - 1) / MACKO_CHUNK_SIZE;

        A->row_ptr[i]     = c;
        A->row_nchunks[i] = nchunks;

        for (int g = 0; g < nchunks; g++) {
            int base_idx = row_start + g * MACKO_CHUNK_SIZE;
            int col_base = csr->col_idx[base_idx]; /* k=0 sempre válido */

            A->col_bases[c]   = (int16_t)col_base;
            A->valid_masks[c] = 0;

            for (int k = 0; k < MACKO_CHUNK_SIZE; k++) {
                int idx  = base_idx + k;
                int slot = MACKO_SLOT(c, k);
                if (idx < row_start + row_nnz) {
                    A->col_deltas[slot] = (int16_t)(csr->col_idx[idx] - col_base);
                    A->vals[slot]       = csr->values[idx];
                    A->valid_masks[c]  |= (uint8_t)(1 << k);
                }
                /* inválido: col_deltas[slot]=0, vals[slot]=0.0 (calloc já zerou) */
            }
            c++;
        }
    }

    return A;
}

MACKOMatrix *macko_from_matrix(const Matrix *m) {
    if (!m) return NULL;

    CSRMatrix *csr = csr_from_matrix(m);
    if (!csr) return NULL;

    MACKOMatrix *A = macko_from_csr(csr);
    csr_free(csr);
    return A;
}

void macko_free(MACKOMatrix *A) {
    if (!A) return;
    free(A->valid_masks);
    free(A->col_bases);
    free(A->col_deltas);
    free(A->vals);
    free(A->row_ptr);
    free(A->row_nchunks);
    free(A);
}

void macko_print(const MACKOMatrix *A) {
    printf("MACKO %dx%d  nnz=%d  total_chunks=%d  chunk_size=%d\n",
           A->rows, A->cols, A->nnz, A->total_chunks, MACKO_CHUNK_SIZE);

    int limit = A->rows < 4 ? A->rows : 4;
    for (int i = 0; i < limit; i++) {
        printf("  row %d (chunks=%d): ", i, A->row_nchunks[i]);
        for (int g = 0; g < A->row_nchunks[i]; g++) {
            int c = A->row_ptr[i] + g;
            printf("[ ");
            for (int k = 0; k < MACKO_CHUNK_SIZE; k++) {
                if (!(A->valid_masks[c] & (1 << k)))
                    printf("pad ");
                else {
                    int col = A->col_bases[c] + A->col_deltas[MACKO_SLOT(c, k)];
                    printf("col%d=%.4f ", col, A->vals[MACKO_SLOT(c, k)]);
                }
            }
            printf("] ");
        }
        printf("\n");
    }
}

void spmv_cpu_macko_sequential(const MACKOMatrix *A, const double *x, double *y) {
    for (int i = 0; i < A->rows; i++) {
        double acc = 0.0;
        int first = A->row_ptr[i];
        int last  = first + A->row_nchunks[i];
        for (int c = first; c < last; c++) {
            uint8_t mask = A->valid_masks[c];
            int16_t base = A->col_bases[c];
            for (int k = 0; k < MACKO_CHUNK_SIZE; k++) {
                if (mask & (1 << k)) {
                    int col = (int)base + (int)A->col_deltas[MACKO_SLOT(c, k)];
                    acc += A->vals[MACKO_SLOT(c, k)] * x[col];
                }
            }
        }
        y[i] = acc;
    }
}

void spmv_cpu_macko_openmp(const MACKOMatrix *A, const double *x, double *y) {
    /* static: chunks por linha já distribuem o trabalho de forma uniforme */
    #pragma omp parallel for schedule(static) default(none) shared(A, x, y)
    for (int i = 0; i < A->rows; i++) {
        double acc = 0.0;
        int first = A->row_ptr[i];
        int last  = first + A->row_nchunks[i];
        for (int c = first; c < last; c++) {
            uint8_t mask = A->valid_masks[c];
            int16_t base = A->col_bases[c];
            for (int k = 0; k < MACKO_CHUNK_SIZE; k++) {
                if (mask & (1 << k)) {
                    int col = (int)base + (int)A->col_deltas[MACKO_SLOT(c, k)];
                    acc += A->vals[MACKO_SLOT(c, k)] * x[col];
                }
            }
        }
        y[i] = acc;
    }
}
