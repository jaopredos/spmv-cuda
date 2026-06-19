#ifndef SPMV_CUDA_H
#define SPMV_CUDA_H

#include "csr.h"
#include "macko.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Executa SpMV completo (alloc + H2D + kernel + D2H + free) */
void spmv_cuda_csr(const CSRMatrix *A, const double *x, double *y);

/*
 * API de contexto para benchmark:
 * mantém buffers no device entre runs para medir só o kernel.
 *
 *   cuda_spmv_ctx_create  — aloca device e faz upload de A e x (uma vez)
 *   cuda_spmv_run         — lança o kernel e retorna tempo em ms (cudaEvent)
 *   cuda_spmv_ctx_result  — copia y do device para o host
 *   cuda_spmv_ctx_destroy — libera device
 */
typedef struct CUDASpMVCtx CUDASpMVCtx;

CUDASpMVCtx *cuda_spmv_ctx_create(const CSRMatrix *A, const double *x);
float        cuda_spmv_run(CUDASpMVCtx *ctx);
void         cuda_spmv_ctx_result(CUDASpMVCtx *ctx, double *y);
void         cuda_spmv_ctx_destroy(CUDASpMVCtx *ctx);

/* ---- MACKO GPU ---- */
typedef struct CUDAMACKOCtx CUDAMACKOCtx;  /* opaque */

CUDAMACKOCtx *cuda_macko_ctx_create(const MACKOMatrix *A, const double *x);
float          cuda_macko_run(CUDAMACKOCtx *ctx);           /* retorna ms */
void           cuda_macko_ctx_result(CUDAMACKOCtx *ctx, double *y);
void           cuda_macko_ctx_destroy(CUDAMACKOCtx *ctx);

/* ---- MACKO GPU — precisão mista FP16/FP64 ---- */
typedef struct CUDAMACKOFp16Ctx CUDAMACKOFp16Ctx;  /* opaque */

CUDAMACKOFp16Ctx *cuda_macko_fp16_ctx_create(const MACKOMatrix *A, const double *x);
float              cuda_macko_fp16_run(CUDAMACKOFp16Ctx *ctx);
void               cuda_macko_fp16_ctx_result(CUDAMACKOFp16Ctx *ctx, double *y);
void               cuda_macko_fp16_ctx_destroy(CUDAMACKOFp16Ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif
