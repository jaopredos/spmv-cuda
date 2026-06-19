#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/matrix.h"
#include "../include/csr.h"
#include "../include/sp24.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <dimensao> <esparsidade>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    double sparsity = atof(argv[2]);

    if (n <= 0 || sparsity < 0.0 || sparsity > 1.0) {
        fprintf(stderr, "Erro: dimensao > 0 e esparsidade em [0,1].\n");
        return 1;
    }

    srand((unsigned int)time(NULL));

    Matrix *m = generate_sparse_matrix(n, sparsity);
    if (!m) { fprintf(stderr, "Erro: falha ao alocar matriz.\n"); return 1; }

    printf("=== Matriz densa (%dx%d, esparsidade=%.2f) ===\n", n, n, sparsity);
    matrix_print(m);

    CSRMatrix *csr = csr_from_matrix(m);
    if (!csr) { fprintf(stderr, "Erro: falha CSR.\n"); matrix_free(m); return 1; }
    printf("\n=== CSR ===\n");
    csr_print(csr);

    SP24Matrix *sp24 = sp24_from_matrix(m);
    matrix_free(m);
    if (!sp24) { fprintf(stderr, "Erro: falha SP24.\n"); csr_free(csr); return 1; }
    printf("\n=== 2:4 Structured Sparse (SP24) ===\n");
    sp24_print(sp24);

    double *x     = malloc(n * sizeof(double));
    double *y     = malloc(n * sizeof(double));
    double *y_sp24 = malloc(n * sizeof(double));
    if (!x || !y || !y_sp24) {
        fprintf(stderr, "Erro: falha ao alocar vetores.\n");
        csr_free(csr); sp24_free(sp24); free(x); free(y); free(y_sp24);
        return 1;
    }
    for (int i = 0; i < n; i++)
        x[i] = rand() / (double)RAND_MAX + 1e-9;

    csr_spmv(csr, x, y);
    printf("\n=== SpMV CSR (y = A*x) ===\n");
    for (int i = 0; i < n; i++) printf("  y[%d] = %.6f\n", i, y[i]);
    printf("\n=== Verificacao CSR ===\n");
    printf("%s\n", csr_verify(csr, x, y, 1e-9) ? "OK" : "FALHOU");

    int groups = sp24->cols_padded / 4;
    for (int i = 0; i < sp24->rows; i++) {
        double acc = 0.0;
        for (int g = 0; g < groups; g++) {
            int vi = i * (sp24->cols_padded / 2) + g * 2;
            int mi = i * (sp24->cols_padded / 4) + g;
            int c0 = g * 4 + (sp24->metadata[mi] & 0x3);
            int c1 = g * 4 + ((sp24->metadata[mi] >> 2) & 0x3);
            acc += sp24->values[vi]     * x[c0];
            acc += sp24->values[vi + 1] * x[c1];
        }
        y_sp24[i] = acc;
    }
    printf("\n=== SpMV SP24 (y = A_podada*x) ===\n");
    for (int i = 0; i < n; i++) printf("  y[%d] = %.6f\n", i, y_sp24[i]);

    printf("\n=== Diferenca CSR vs SP24 (impacto do pruning 2:4) ===\n");
    for (int i = 0; i < n; i++)
        printf("  linha %d: csr=%.6f  sp24=%.6f  diff=%.2e\n",
               i, y[i], y_sp24[i], y[i] - y_sp24[i]);

    csr_free(csr); sp24_free(sp24);
    free(x); free(y); free(y_sp24);
    return 0;
}
