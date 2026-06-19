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
    A->chunks       = malloc(total_chunks * sizeof(MACKOChunk));
    A->row_ptr      = malloc(rows * sizeof(int));
    A->row_nchunks  = malloc(rows * sizeof(int));

    if (!A->chunks || !A->row_ptr || !A->row_nchunks) { macko_free(A); return NULL; }

    int c = 0;
    for (int i = 0; i < rows; i++) {
        int row_start = csr->row_ptr[i];
        int row_nnz   = csr->row_ptr[i + 1] - row_start;
        int nchunks   = (row_nnz + MACKO_CHUNK_SIZE - 1) / MACKO_CHUNK_SIZE;

        A->row_ptr[i]     = c;
        A->row_nchunks[i] = nchunks;

        for (int g = 0; g < nchunks; g++) {
            MACKOChunk *chunk = &A->chunks[c++];
            for (int k = 0; k < MACKO_CHUNK_SIZE; k++) {
                int idx = row_start + g * MACKO_CHUNK_SIZE + k;
                if (idx < row_start + row_nnz) {
                    chunk->cols[k] = csr->col_idx[idx];
                    chunk->vals[k] = csr->values[idx];
                } else {
                    chunk->cols[k] = -1;
                    chunk->vals[k] = 0.0;
                }
            }
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
    free(A->chunks);
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
            MACKOChunk *chunk = &A->chunks[A->row_ptr[i] + g];
            printf("[ ");
            for (int k = 0; k < MACKO_CHUNK_SIZE; k++) {
                if (chunk->cols[k] == -1)
                    printf("pad ");
                else
                    printf("col%d=%.4f ", chunk->cols[k], chunk->vals[k]);
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
            const MACKOChunk *chunk = &A->chunks[c];
            for (int k = 0; k < MACKO_CHUNK_SIZE; k++)
                if (chunk->cols[k] != -1)
                    acc += chunk->vals[k] * x[chunk->cols[k]];
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
            const MACKOChunk *chunk = &A->chunks[c];
            for (int k = 0; k < MACKO_CHUNK_SIZE; k++)
                if (chunk->cols[k] != -1)
                    acc += chunk->vals[k] * x[chunk->cols[k]];
        }
        y[i] = acc;
    }
}
