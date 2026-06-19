#include <stdio.h>
#include <stdlib.h>
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

/* ------------------------------------------------------------------ */
/* Kernel 1: um thread por linha (baseline MACKO GPU)                   */
/* ------------------------------------------------------------------ */

__global__ void kernel_macko_thread_per_row(
        int             rows,
        const uint8_t  *valid_masks,
        const int16_t  *col_bases,
        const int16_t  *col_deltas,
        const double   *vals,
        const int      *row_ptr,
        const int      *row_nchunks,
        const double   *x,
        double         *y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    double acc   = 0.0;
    int    first = row_ptr[row];
    int    last  = first + row_nchunks[row];
    for (int c = first; c < last; c++) {
        uint8_t mask = valid_masks[c];
        int16_t base = col_bases[c];
        for (int k = 0; k < MACKO_CHUNK_SIZE; k++) {
            if (mask & (1 << k)) {
                int col = (int)base + (int)col_deltas[c * MACKO_CHUNK_SIZE + k];
                acc += vals[c * MACKO_CHUNK_SIZE + k] * x[col];
            }
        }
    }
    y[row] = acc;
}

/* ------------------------------------------------------------------ */
/* Kernel 2: um warp por linha (otimizado para linhas densas)           */
/* ------------------------------------------------------------------ */

__global__ void kernel_macko_warp_per_row(
        int             rows,
        const uint8_t  *valid_masks,
        const int16_t  *col_bases,
        const int16_t  *col_deltas,
        const double   *vals,
        const int      *row_ptr,
        const int      *row_nchunks,
        const double   *x,
        double         *y)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (warp_id >= rows) return;

    double acc           = 0.0;
    int    primeiro_chunk = row_ptr[warp_id];
    int    nchunks         = row_nchunks[warp_id];

    for (int c = lane; c < nchunks; c += 32) {
        int     chunk_idx = primeiro_chunk + c;
        uint8_t mask      = valid_masks[chunk_idx];
        int16_t base      = col_bases[chunk_idx];
        for (int k = 0; k < MACKO_CHUNK_SIZE; k++) {
            if (mask & (1 << k)) {
                int col = (int)base + (int)col_deltas[chunk_idx * MACKO_CHUNK_SIZE + k];
                acc += vals[chunk_idx * MACKO_CHUNK_SIZE + k] * x[col];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0) y[warp_id] = acc;
}

/* ------------------------------------------------------------------ */
/* Contexto de benchmark: buffers persistem entre runs                 */
/* ------------------------------------------------------------------ */

struct CUDAMACKOCtx {
    uint8_t     *d_valid_masks;
    int16_t     *d_col_bases;
    int16_t     *d_col_deltas;
    double      *d_vals;
    int         *d_row_ptr;
    int         *d_row_nchunks;
    double      *d_x;
    double      *d_y;
    int          rows;
    int          tpb;
    int          blocks;
    cudaEvent_t  ev_start;
    cudaEvent_t  ev_stop;
};

extern "C"
CUDAMACKOCtx *cuda_macko_ctx_create(const MACKOMatrix *A, const double *x) {
    CUDAMACKOCtx *ctx = (CUDAMACKOCtx *)malloc(sizeof(CUDAMACKOCtx));
    if (!ctx) return NULL;

    ctx->rows   = A->rows;
    ctx->tpb    = 256;                                  /* 8 warps por bloco */
    ctx->blocks = (A->rows * 32 + ctx->tpb - 1) / ctx->tpb; /* um warp por linha */

    long tc = A->total_chunks;
    long cs = MACKO_CHUNK_SIZE;

    CUDA_CHECK(cudaMalloc(&ctx->d_valid_masks, tc * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_col_bases,  tc * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_col_deltas, tc * cs * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&ctx->d_vals,       tc * cs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_row_ptr,    A->rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_row_nchunks, A->rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_x,          A->cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_y,          A->rows * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(ctx->d_valid_masks, A->valid_masks,
        tc * sizeof(uint8_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_col_bases, A->col_bases,
        tc * sizeof(int16_t),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_col_deltas, A->col_deltas,
        tc * cs * sizeof(int16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_vals, A->vals,
        tc * cs * sizeof(double),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_row_ptr,    A->row_ptr,    A->rows * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_row_nchunks, A->row_nchunks, A->rows * sizeof(int),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_x,          x,             A->cols * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventCreate(&ctx->ev_start));
    CUDA_CHECK(cudaEventCreate(&ctx->ev_stop));

    return ctx;
}

/* Lança o kernel warp-per-row (mais eficiente para linhas com muitos nnz) */
extern "C"
float cuda_macko_run(CUDAMACKOCtx *ctx) {
    CUDA_CHECK(cudaEventRecord(ctx->ev_start));
    kernel_macko_warp_per_row<<<ctx->blocks, ctx->tpb>>>(
        ctx->rows,
        ctx->d_valid_masks, ctx->d_col_bases, ctx->d_col_deltas, ctx->d_vals,
        ctx->d_row_ptr, ctx->d_row_nchunks,
        ctx->d_x, ctx->d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ctx->ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ctx->ev_stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ctx->ev_start, ctx->ev_stop));
    return ms;
}

extern "C"
void cuda_macko_ctx_result(CUDAMACKOCtx *ctx, double *y) {
    CUDA_CHECK(cudaMemcpy(y, ctx->d_y, ctx->rows * sizeof(double), cudaMemcpyDeviceToHost));
}

extern "C"
void cuda_macko_ctx_destroy(CUDAMACKOCtx *ctx) {
    if (!ctx) return;
    cudaEventDestroy(ctx->ev_start);
    cudaEventDestroy(ctx->ev_stop);
    cudaFree(ctx->d_valid_masks);
    cudaFree(ctx->d_col_bases);
    cudaFree(ctx->d_col_deltas);
    cudaFree(ctx->d_vals);
    cudaFree(ctx->d_row_ptr);
    cudaFree(ctx->d_row_nchunks);
    cudaFree(ctx->d_x);
    cudaFree(ctx->d_y);
    free(ctx);
}
