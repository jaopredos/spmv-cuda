#ifndef MACKO_H
#define MACKO_H

#include <stdint.h>
#include "matrix.h"
#include "csr.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MACKO_CHUNK_SIZE 4  /* elementos por chunk; múltiplo de 4 para alinhamento */

/* índice flat do slot k do chunk c nos arrays col_deltas e vals */
#define MACKO_SLOT(c, k)  ((c) * MACKO_CHUNK_SIZE + (k))

/*
 * MACKOMatrix em layout SoA (Structure of Arrays): cada campo do chunk
 * vive em seu próprio array contíguo, sem padding de alinhamento entre
 * campos de tipos diferentes (o problema que existia no layout AoS,
 * onde o `double vals[4]` forçava 5 bytes de padding por chunk).
 *
 *   valid_masks[c]       : bit k = 1 se o slot k do chunk c é válido
 *   col_bases[c]         : coluna absoluta do primeiro elemento válido do chunk c
 *   col_deltas[SLOT(c,k)]: delta[k] = col_abs[k] - col_base (irrelevante se inválido)
 *   vals[SLOT(c,k)]      : valor não-zero (0.0 em slots de padding)
 *
 * Reconstrução da coluna absoluta: col_bases[c] + col_deltas[MACKO_SLOT(c,k)]
 */
typedef struct {
    int      rows;
    int      cols;
    int      nnz;
    int      total_chunks;
    uint8_t *valid_masks;  /* [total_chunks]               — 1 byte/chunk  */
    int16_t *col_bases;    /* [total_chunks]               — 2 bytes/chunk */
    int16_t *col_deltas;   /* [total_chunks * CHUNK_SIZE]  — 8 bytes/chunk */
    double  *vals;         /* [total_chunks * CHUNK_SIZE]  — 32 bytes/chunk*/
    int     *row_ptr;      /* row_ptr[i] = índice do primeiro chunk da linha i */
    int     *row_nchunks;  /* quantos chunks tem a linha i */
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
