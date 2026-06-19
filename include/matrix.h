#ifndef MATRIX_H
#define MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;

Matrix *matrix_create(int rows, int cols);
void    matrix_free(Matrix *m);
double  matrix_get(const Matrix *m, int i, int j);
void    matrix_set(Matrix *m, int i, int j, double val);
void    matrix_print(const Matrix *m);
Matrix *generate_sparse_matrix(int n, double sparsity);

#ifdef __cplusplus
}
#endif

#endif
