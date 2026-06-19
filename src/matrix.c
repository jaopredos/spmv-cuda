#include <stdio.h>
#include <stdlib.h>
#include "../include/matrix.h"

Matrix *matrix_create(int rows, int cols) {
    Matrix *m = malloc(sizeof(Matrix));
    if (!m) return NULL;
    m->rows = rows;
    m->cols = cols;
    m->data = calloc(rows * cols, sizeof(double));
    if (!m->data) { free(m); return NULL; }
    return m;
}

void matrix_free(Matrix *m) {
    if (!m) return;
    free(m->data);
    free(m);
}

double matrix_get(const Matrix *m, int i, int j) {
    return m->data[i * m->cols + j];
}

void matrix_set(Matrix *m, int i, int j, double val) {
    m->data[i * m->cols + j] = val;
}

void matrix_print(const Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++)
            printf("%6.2f ", matrix_get(m, i, j));
        printf("\n");
    }
}

Matrix *generate_sparse_matrix(int n, double sparsity) {
    if (n <= 0 || sparsity < 0.0 || sparsity > 1.0) return NULL;

    Matrix *m = matrix_create(n, n);
    if (!m) return NULL;

    int threshold = (int)(sparsity * RAND_MAX);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int r = rand();
            matrix_set(m, i, j, (r > threshold) ? (r / (double)RAND_MAX) : 0.0);
        }
    return m;
}
