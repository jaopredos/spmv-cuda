#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/sp24.h"

SP24Matrix *sp24_from_matrix(const Matrix *m) {
    if (!m) return NULL;

    int rows        = m->rows;
    int cols        = m->cols;
    int cols_padded = (cols + 3) & ~3;  /* arredonda para múltiplo de 4 */

    SP24Matrix *A = malloc(sizeof(SP24Matrix));
    if (!A) return NULL;

    A->rows        = rows;
    A->cols        = cols;
    A->cols_padded = cols_padded;
    A->values      = calloc(rows * (cols_padded / 2), sizeof(double));
    A->metadata    = calloc(rows * (cols_padded / 4), sizeof(uint8_t));

    if (!A->values || !A->metadata) { sp24_free(A); return NULL; }

    for (int i = 0; i < rows; i++) {
        int num_groups = cols_padded / 4;
        for (int g = 0; g < num_groups; g++) {
            /* lê os 4 elementos do grupo (zero se fora dos limites) */
            double grp[4] = {0.0, 0.0, 0.0, 0.0};
            for (int t = 0; t < 4; t++) {
                int j = g * 4 + t;
                if (j < cols) grp[t] = matrix_get(m, i, j);
            }

            /*
             * Seleciona as 2 posições de maior |valor| (pruning 2:4).
             * Inicializa com as duas primeiras e percorre o restante.
             */
            int p0 = 0, p1 = 1;
            if (fabs(grp[1]) > fabs(grp[0])) { p0 = 1; p1 = 0; }
            for (int t = 2; t < 4; t++) {
                if (fabs(grp[t]) > fabs(grp[p0])) { p1 = p0; p0 = t; }
                else if (fabs(grp[t]) > fabs(grp[p1])) { p1 = t; }
            }
            if (p0 > p1) { int tmp = p0; p0 = p1; p1 = tmp; } /* ordem canônica */

            int vi = i * (cols_padded / 2) + g * 2;
            int mi = i * (cols_padded / 4) + g;

            A->values[vi]     = grp[p0];
            A->values[vi + 1] = grp[p1];
            A->metadata[mi]   = (uint8_t)((p0 & 0x3) | ((p1 & 0x3) << 2));
        }
    }

    return A;
}

void sp24_free(SP24Matrix *A) {
    if (!A) return;
    free(A->values);
    free(A->metadata);
    free(A);
}

void sp24_print(const SP24Matrix *A) {
    int groups = A->cols_padded / 4;
    printf("SP24 %dx%d (cols_padded=%d)  nnz=%d\n",
           A->rows, A->cols, A->cols_padded,
           A->rows * (A->cols_padded / 2));

    for (int i = 0; i < A->rows; i++) {
        printf("  row %d: ", i);
        for (int g = 0; g < groups; g++) {
            int vi = i * (A->cols_padded / 2) + g * 2;
            int mi = i * (A->cols_padded / 4) + g;
            int c0 = g * 4 + (A->metadata[mi] & 0x3);
            int c1 = g * 4 + ((A->metadata[mi] >> 2) & 0x3);
            printf("[col%d=%.4f col%d=%.4f] ", c0, A->values[vi],
                                               c1, A->values[vi + 1]);
        }
        printf("\n");
    }
}
