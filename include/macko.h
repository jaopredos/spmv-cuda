#ifndef MACKO_H
#define MACKO_H

#include <stdint.h>
#include "matrix.h"
#include "csr.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MACKO_CHUNK_SIZE 4  /* elementos por chunk; múltiplo de 4 para alinhamento */

/*
 * Chunk: CHUNK_SIZE pares (col, val) armazenados de forma entrelaçada.
 * Layout em memória: [col0|col1|...|colK|val0|val1|...|valK]
 * Colunas inválidas (padding) usam col=-1, val=0.0
 */
typedef struct {
    int    cols[MACKO_CHUNK_SIZE];
    double vals[MACKO_CHUNK_SIZE];
} MACKOChunk;

typedef struct {
    int         rows;
    int         cols;
    int         nnz;
    int         total_chunks;
    MACKOChunk *chunks;      /* array de chunks, row-major */
    int        *row_ptr;     /* row_ptr[i] = índice do primeiro chunk da linha i */
    int        *row_nchunks; /* quantos chunks tem a linha i */
} MACKOMatrix;

/* Constrói a partir de CSRMatrix (já construída) */
MACKOMatrix *macko_from_csr(const CSRMatrix *csr);

/* Constrói direto da Matrix densa (conveniência) */
MACKOMatrix *macko_from_matrix(const Matrix *m);

void   macko_free(MACKOMatrix *A);
void   macko_print(const MACKOMatrix *A);  /* debug: imprime até 4 linhas */

/* SpMV CPU sequencial */
void   spmv_cpu_macko_sequential(const MACKOMatrix *A, const double *x, double *y);

/* SpMV CPU OpenMP */
void   spmv_cpu_macko_openmp(const MACKOMatrix *A, const double *x, double *y);

#ifdef __cplusplus
}
#endif

#endif
