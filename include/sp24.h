#ifndef SP24_H
#define SP24_H

#include <stdint.h>
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * 2:4 Structured Sparse format (padrão NVIDIA Ampere):
 * A cada grupo de 4 elementos consecutivos de uma linha, exatamente 2 são
 * não-zero.
 *
 *   values      : valores não-zero, 2 por grupo de 4 colunas, linha por linha
 *                 tamanho = rows * (cols_padded / 2)
 *
 *   metadata    : um byte por grupo de 4 colunas, codificando os offsets
 *                 dentro do grupo das duas posições não-zero:
 *                   bits [1:0] = offset do 1º não-zero (0..3)
 *                   bits [3:2] = offset do 2º não-zero (0..3)
 *                 tamanho = rows * (cols_padded / 4)
 *
 *   cols_padded : cols arredondado para cima ao próximo múltiplo de 4
 *
 * A conversão de uma Matrix densa mantém os 2 elementos de maior |valor|
 * em cada grupo (pruning para garantir o padrão 2:4).
 */
typedef struct {
    int      rows;
    int      cols;
    int      cols_padded;
    double  *values;
    uint8_t *metadata;
} SP24Matrix;

SP24Matrix *sp24_from_matrix(const Matrix *m);
void        sp24_free(SP24Matrix *A);
void        sp24_print(const SP24Matrix *A);

#ifdef __cplusplus
}
#endif

#endif
