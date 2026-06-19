#include <stdio.h>
#include <stdlib.h>
#include <cusparse.h>
#include "../include/spmv_cuda.h"

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define CUSPARSE_CHECK(call)                                               \
    do {                                                                   \
        cusparseStatus_t err = (call);                                     \
        if (err != CUSPARSE_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuSPARSE error %s:%d — code %d\n",            \
                    __FILE__, __LINE__, (int)err);                         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* ------------------------------------------------------------------ */
/* Kernel: um thread por linha                                          */
/* ------------------------------------------------------------------ */

__global__ void kernel_spmv_csr_thread_per_row(
        int            rows,
        const double  *values,
        const int     *col_idx,
        const int     *row_ptr,
        const double  *x,
        double        *y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    double acc = 0.0;
    for (int k = row_ptr[row]; k < row_ptr[row + 1]; k++)
        acc += values[k] * x[col_idx[k]];
    y[row] = acc;
}

/* ------------------------------------------------------------------ */
/* SpMV completo (conveniente para uso fora do benchmark)              */
/* ------------------------------------------------------------------ */

extern "C"
void spmv_cuda_csr(const CSRMatrix *A, const double *x, double *y) {
    double *d_values, *d_x, *d_y;
    int    *d_col_idx, *d_row_ptr;

    CUDA_CHECK(cudaMalloc(&d_values,  A->nnz        * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, A->nnz        * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (A->rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x,       A->cols       * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y,       A->rows       * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_values,  A->values,  A->nnz        * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, A->col_idx, A->nnz        * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, A->row_ptr, (A->rows + 1) * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x,       x,          A->cols       * sizeof(double), cudaMemcpyHostToDevice));

    int tpb    = 256;
    int blocks = (A->rows + tpb - 1) / tpb;
    kernel_spmv_csr_thread_per_row<<<blocks, tpb>>>(
        A->rows, d_values, d_col_idx, d_row_ptr, d_x, d_y);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(y, d_y, A->rows * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_values); cudaFree(d_col_idx); cudaFree(d_row_ptr);
    cudaFree(d_x);      cudaFree(d_y);
}

/* ------------------------------------------------------------------ */
/* Contexto de benchmark: buffers persistem entre runs                 */
/* ------------------------------------------------------------------ */

struct CUDASpMVCtx {
    double      *d_values;
    int         *d_col_idx;
    int         *d_row_ptr;
    double      *d_x;
    double      *d_y;
    int          rows;
    int          tpb;
    int          blocks;
    cudaEvent_t  ev_start;
    cudaEvent_t  ev_stop;
};

extern "C"
CUDASpMVCtx *cuda_spmv_ctx_create(const CSRMatrix *A, const double *x) {
    CUDASpMVCtx *ctx = (CUDASpMVCtx *)malloc(sizeof(CUDASpMVCtx));
    if (!ctx) return NULL;

    ctx->rows   = A->rows;
    ctx->tpb    = 256;
    ctx->blocks = (A->rows + ctx->tpb - 1) / ctx->tpb;

    CUDA_CHECK(cudaMalloc(&ctx->d_values,  A->nnz        * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_col_idx, A->nnz        * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_row_ptr, (A->rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_x,       A->cols       * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_y,       A->rows       * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(ctx->d_values,  A->values,  A->nnz        * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_col_idx, A->col_idx, A->nnz        * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_row_ptr, A->row_ptr, (A->rows + 1) * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_x,       x,          A->cols       * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventCreate(&ctx->ev_start));
    CUDA_CHECK(cudaEventCreate(&ctx->ev_stop));

    return ctx;
}

/* Lança o kernel e retorna o tempo de execução em ms via cudaEvent */
extern "C"
float cuda_spmv_run(CUDASpMVCtx *ctx) {
    CUDA_CHECK(cudaEventRecord(ctx->ev_start));
    kernel_spmv_csr_thread_per_row<<<ctx->blocks, ctx->tpb>>>(
        ctx->rows, ctx->d_values, ctx->d_col_idx, ctx->d_row_ptr,
        ctx->d_x, ctx->d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ctx->ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ctx->ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ctx->ev_start, ctx->ev_stop));
    return ms;
}

extern "C"
void cuda_spmv_ctx_result(CUDASpMVCtx *ctx, double *y) {
    CUDA_CHECK(cudaMemcpy(y, ctx->d_y, ctx->rows * sizeof(double), cudaMemcpyDeviceToHost));
}

extern "C"
void cuda_spmv_ctx_destroy(CUDASpMVCtx *ctx) {
    if (!ctx) return;
    cudaEventDestroy(ctx->ev_start);
    cudaEventDestroy(ctx->ev_stop);
    cudaFree(ctx->d_values);
    cudaFree(ctx->d_col_idx);
    cudaFree(ctx->d_row_ptr);
    cudaFree(ctx->d_x);
    cudaFree(ctx->d_y);
    free(ctx);
}

/* ------------------------------------------------------------------ */
/* cuSPARSE: Contexto e wrappers                                      */
/* ------------------------------------------------------------------ */

struct CUDACuSparseCtx {
    cusparseHandle_t     handle;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    double              *d_values;
    int                 *d_col_idx;
    int                 *d_row_ptr;
    double              *d_x;
    double              *d_y;
    void                *d_buffer;
    size_t               bufferSize;
    int                  rows;
    double               alpha;
    double               beta;
    cudaEvent_t          ev_start;
    cudaEvent_t          ev_stop;
};

extern "C"
CUDACuSparseCtx *cuda_cusparse_ctx_create(const CSRMatrix *A, const double *x) {
    CUDACuSparseCtx *ctx = (CUDACuSparseCtx *)malloc(sizeof(CUDACuSparseCtx));
    if (!ctx) return NULL;

    ctx->rows   = A->rows;
    ctx->alpha  = 1.0;
    ctx->beta   = 0.0;

    CUDA_CHECK(cudaMalloc(&ctx->d_values,  A->nnz        * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_col_idx, A->nnz        * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_row_ptr, (A->rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_x,       A->cols       * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_y,       A->rows       * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(ctx->d_values,  A->values,  A->nnz        * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_col_idx, A->col_idx, A->nnz        * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_row_ptr, A->row_ptr, (A->rows + 1) * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_x,       x,          A->cols       * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx->d_y,       0,          A->rows       * sizeof(double)));

    CUSPARSE_CHECK(cusparseCreate(&ctx->handle));

    CUSPARSE_CHECK(cusparseCreateCsr(&ctx->matA, A->rows, A->cols, A->nnz,
                                     ctx->d_row_ptr, ctx->d_col_idx, ctx->d_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx->vecX, A->cols, ctx->d_x, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx->vecY, A->rows, ctx->d_y, CUDA_R_64F));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(ctx->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &ctx->alpha, ctx->matA, ctx->vecX, &ctx->beta, ctx->vecY,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &ctx->bufferSize));

    CUDA_CHECK(cudaMalloc(&ctx->d_buffer, ctx->bufferSize));

    CUDA_CHECK(cudaEventCreate(&ctx->ev_start));
    CUDA_CHECK(cudaEventCreate(&ctx->ev_stop));

    return ctx;
}

extern "C"
float cuda_cusparse_run(CUDACuSparseCtx *ctx) {
    CUDA_CHECK(cudaEventRecord(ctx->ev_start));
    CUSPARSE_CHECK(cusparseSpMV(ctx->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &ctx->alpha, ctx->matA, ctx->vecX, &ctx->beta, ctx->vecY,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, ctx->d_buffer));
    CUDA_CHECK(cudaEventRecord(ctx->ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ctx->ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ctx->ev_start, ctx->ev_stop));
    return ms;
}

extern "C"
void cuda_cusparse_ctx_result(CUDACuSparseCtx *ctx, double *y) {
    CUDA_CHECK(cudaMemcpy(y, ctx->d_y, ctx->rows * sizeof(double), cudaMemcpyDeviceToHost));
}

extern "C"
void cuda_cusparse_ctx_destroy(CUDACuSparseCtx *ctx) {
    if (!ctx) return;
    cudaEventDestroy(ctx->ev_start);
    cudaEventDestroy(ctx->ev_stop);
    cusparseDestroySpMat(ctx->matA);
    cusparseDestroyDnVec(ctx->vecX);
    cusparseDestroyDnVec(ctx->vecY);
    cusparseDestroy(ctx->handle);
    cudaFree(ctx->d_values);
    cudaFree(ctx->d_col_idx);
    cudaFree(ctx->d_row_ptr);
    cudaFree(ctx->d_x);
    cudaFree(ctx->d_y);
    cudaFree(ctx->d_buffer);
    free(ctx);
}
