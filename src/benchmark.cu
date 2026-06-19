#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <cuda_fp16.h>
#include "../include/matrix.h"
#include "../include/csr.h"
#include "../include/sp24.h"
#include "../include/macko.h"
#include "../include/spmv_cpu.h"
#include "../include/spmv_cuda.h"

/* ------------------------------------------------------------------ */
/* Temporização CPU                                                     */
/* ------------------------------------------------------------------ */

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

/* ------------------------------------------------------------------ */
/* Verificação de corretude                                             */
/* ------------------------------------------------------------------ */

static int verify(const double *ref, const double *got, int n, double tol) {
    for (int i = 0; i < n; i++) {
        double diff = fabs(got[i] - ref[i]);
        double base = fabs(ref[i]);
        if ((base > 1e-12 ? diff / base : diff) > tol) return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Tabela                                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    char  name[40];
    double avg_ms;
    double min_ms;
    double max_ms;
    double gflops;
    double gbs;
    int    correct; /* -1=referência, 1=OK, 0=FALHOU */
} Row;

static void print_table(const Row *rows, int nrows) {
    const char *sep =
        "+-----------------------------------------+----------+----------+----------+---------+--------+--------+";
    printf("%s\n", sep);
    printf("| %-39s | %8s | %8s | %8s | %7s | %6s | %6s |\n",
           "Implementação", "Méd(ms)", "Mín(ms)", "Máx(ms)", "GFLOP/s", "GB/s", "Corret");
    printf("%s\n", sep);
    for (int i = 0; i < nrows; i++) {
        const char *ok = rows[i].correct < 0 ? "  ref"
                       : rows[i].correct     ? "   OK"
                                             : "FALHOU";
        printf("| %-39s | %8.3f | %8.3f | %8.3f | %7.2f | %6.2f | %6s |\n",
               rows[i].name,
               rows[i].avg_ms, rows[i].min_ms, rows[i].max_ms,
               rows[i].gflops, rows[i].gbs, ok);
    }
    printf("%s\n", sep);
}

/* ------------------------------------------------------------------ */
/* Largura de banda de referência para cada formato                    */
/* ------------------------------------------------------------------ */

/* CSR: lê values(nnz*8), col_idx(nnz*4), row_ptr((n+1)*4), x(n*8); escreve y(n*8) */
static double bw_csr_gb(int nnz, int n) {
    return (nnz * 12.0 + (n + 1) * 4.0 + n * 16.0) / 1e9;
}

/* SP24: lê values(n*n/2*8), metadata(n*n/4), x(n*8); escreve y(n*8) */
static double bw_sp24_gb(int n) {
    return ((long)n * n / 2 * 8.0 + (long)n * n / 4.0 + n * 16.0) / 1e9;
}

/* MACKO (SoA): lê valid_masks(total_chunks*1), col_bases(total_chunks*2),
   col_deltas(total_chunks*4*2), vals(total_chunks*4*8), row_ptr(n*4),
   row_nchunks(n*4), x(n*8); escreve y(n*8) */
static double bw_macko_gb(const MACKOMatrix *A, int n) {
    long tc = A->total_chunks;
    long cs = MACKO_CHUNK_SIZE;
    long chunk_bytes = tc * ((long)sizeof(uint8_t)          /* valid_masks */
                           + (long)sizeof(int16_t)          /* col_bases   */
                           + cs * (long)sizeof(int16_t)     /* col_deltas  */
                           + cs * (long)sizeof(double));    /* vals        */
    long meta_bytes  = (long)n * (2 * (long)sizeof(int) + 2 * (long)sizeof(double));
    return (chunk_bytes + meta_bytes) / 1e9;
}

/* MACKO FP16: igual ao MACKO SoA mas vals ocupa 2 bytes por slot (FP16)
   em vez de 8 (double) — redução de 32→8 bytes/chunk em vals            */
static double bw_macko_fp16_gb(const MACKOMatrix *A, int n) {
    long tc = A->total_chunks;
    long cs = MACKO_CHUNK_SIZE;
    long chunk_bytes = tc * ((long)sizeof(uint8_t)          /* valid_masks */
                           + (long)sizeof(int16_t)          /* col_bases   */
                           + cs * (long)sizeof(int16_t)     /* col_deltas  */
                           + cs * (long)sizeof(__half));    /* vals FP16   */
    long meta_bytes  = (long)n * (2 * (long)sizeof(int) + 2 * (long)sizeof(double));
    return (chunk_bytes + meta_bytes) / 1e9;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Uso: %s <dimensao> <esparsidade> [runs=50]\n", argv[0]);
        return 1;
    }

    int    n        = atoi(argv[1]);
    double sparsity = atof(argv[2]);
    int    runs     = (argc == 4) ? atoi(argv[3]) : 50;

    if (n <= 0 || sparsity < 0.0 || sparsity > 1.0 || runs < 2) {
        fprintf(stderr, "Erro: n>0, esparsidade em [0,1], runs>=2.\n");
        return 1;
    }

    srand((unsigned int)time(NULL));

    /* ---- geração e conversão ---- */
    Matrix *m = generate_sparse_matrix(n, sparsity);
    if (!m) { fprintf(stderr, "Erro: falha ao gerar matriz.\n"); return 1; }

    CSRMatrix   *csr   = csr_from_matrix(m);
    SP24Matrix  *sp24  = sp24_from_matrix(m);
    matrix_free(m);

    if (!csr || !sp24) { fprintf(stderr, "Erro: falha na conversão.\n"); return 1; }

    MACKOMatrix *macko = macko_from_csr(csr);
    if (!macko) { fprintf(stderr, "Erro: falha na conversão MACKO.\n"); return 1; }

    /* ---- vetor x aleatório ---- */
    double *x = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        x[i] = rand() / (double)RAND_MAX + 1e-9;

    /* ---- buffers de saída ---- */
    double *y_ref   = (double *)calloc(n, sizeof(double)); /* referência CSR seq */
    double *y_tmp   = (double *)calloc(n, sizeof(double));

    /* métricas fixas */
    double flops_csr  = 2.0 * csr->nnz;
    double flops_sp24 = 2.0 * (long)n * (n / 2); /* sempre n*n/2 nz após pruning */
    double bw_csr     = bw_csr_gb(csr->nnz, n);
    double bw_sp24    = bw_sp24_gb(n);

    printf("\nBenchmark SpMV  n=%d  esparsidade=%.2f  nnz=%d  runs=%d (1 warm-up descartado)\n\n",
           n, sparsity, csr->nnz, runs);

#define N_IMPL 9
    Row table[N_IMPL];
    int ri = 0;

    /* ================================================================
     * Macro que roda uma implementação CPU N vezes, descarta o 1º run,
     * calcula avg/min/max e preenche a linha da tabela.
     * ================================================================ */
#define BENCH_CPU(label, call, y_out, ref, flops, bw)                     \
    do {                                                                   \
        double sum = 0, mn = DBL_MAX, mx = 0;                             \
        for (int r = 0; r < runs; r++) {                                  \
            double t0 = now_ms();                                          \
            call;                                                          \
            double dt = now_ms() - t0;                                    \
            if (r == 0) continue; /* warm-up */                           \
            sum += dt; if (dt < mn) mn = dt; if (dt > mx) mx = dt;       \
        }                                                                  \
        double avg = sum / (runs - 1);                                     \
        snprintf(table[ri].name, sizeof(table[ri].name), "%s", label);    \
        table[ri].avg_ms  = avg;                                           \
        table[ri].min_ms  = mn;                                            \
        table[ri].max_ms  = mx;                                            \
        table[ri].gflops  = (flops) / (avg * 1e-3) / 1e9;                 \
        table[ri].gbs     = (bw)    / (avg * 1e-3);                       \
        table[ri].correct = (ref == NULL) ? -1                            \
                          : verify(ref, y_out, n, 1e-9);                  \
        ri++;                                                              \
    } while (0)

    /* ---- 1. CPU sequencial CSR (referência) ---- */
    BENCH_CPU("CPU sequencial (CSR)",
              spmv_cpu_sequential(csr, x, y_ref),
              y_ref, NULL, flops_csr, bw_csr);

    /* ---- 2. CPU OpenMP CSR ---- */
    BENCH_CPU("CPU OpenMP (CSR)",
              spmv_cpu_openmp(csr, x, y_tmp),
              y_tmp, y_ref, flops_csr, bw_csr);

    /* ---- 3. CPU sequencial SP24 (referência SP24) ---- */
    double *y_ref_sp24 = (double *)calloc(n, sizeof(double));
    BENCH_CPU("CPU sequencial (SP24)",
              spmv_cpu_sp24_sequential(sp24, x, y_ref_sp24),
              y_ref_sp24, NULL, flops_sp24, bw_sp24);

    /* ---- 4. CPU OpenMP SP24 ---- */
    BENCH_CPU("CPU OpenMP (SP24)",
              spmv_cpu_sp24_openmp(sp24, x, y_tmp),
              y_tmp, y_ref_sp24, flops_sp24, bw_sp24);

    /* ---- 5. GPU CUDA CSR (mede só kernel, dados já no device) ---- */
    {
        CUDASpMVCtx *ctx = cuda_spmv_ctx_create(csr, x);

        double sum = 0, mn = DBL_MAX, mx = 0;
        for (int r = 0; r < runs; r++) {
            float ms = cuda_spmv_run(ctx);
            if (r == 0) continue; /* warm-up */
            sum += ms;
            if (ms < mn) mn = ms;
            if (ms > mx) mx = ms;
        }
        double avg = sum / (runs - 1);

        cuda_spmv_ctx_result(ctx, y_tmp);
        cuda_spmv_ctx_destroy(ctx);

        snprintf(table[ri].name, sizeof(table[ri].name), "GPU CUDA kernel (CSR)");
        table[ri].avg_ms  = avg;
        table[ri].min_ms  = mn;
        table[ri].max_ms  = mx;
        table[ri].gflops  = flops_csr / (avg * 1e-3) / 1e9;
        table[ri].gbs     = bw_csr    / (avg * 1e-3);
        table[ri].correct = verify(y_ref, y_tmp, n, 1e-6);
        ri++;
    }

    /* ---- 6. CPU sequencial MACKO ---- */
    double *y_ref_macko = (double *)calloc(n, sizeof(double));
    BENCH_CPU("CPU sequencial (MACKO)",
              spmv_cpu_macko_sequential(macko, x, y_ref_macko),
              y_ref_macko, NULL, flops_csr, bw_macko_gb(macko, n));

    /* ---- 7. CPU OpenMP MACKO ---- */
    BENCH_CPU("CPU OpenMP (MACKO)",
              spmv_cpu_macko_openmp(macko, x, y_tmp),
              y_tmp, y_ref_macko, flops_csr, bw_macko_gb(macko, n));

    /* ---- 8. GPU CUDA MACKO (warp per row) ---- */
    {
        CUDAMACKOCtx *mctx = cuda_macko_ctx_create(macko, x);
        double sum = 0, mn = DBL_MAX, mx = 0;
        for (int r = 0; r < runs; r++) {
            float ms = cuda_macko_run(mctx);
            if (r == 0) continue;
            sum += ms;
            if (ms < mn) mn = ms;
            if (ms > mx) mx = ms;
        }
        double avg = sum / (runs - 1);
        cuda_macko_ctx_result(mctx, y_tmp);
        cuda_macko_ctx_destroy(mctx);

        snprintf(table[ri].name, sizeof(table[ri].name), "GPU CUDA warp/row (MACKO)");
        table[ri].avg_ms  = avg;
        table[ri].min_ms  = mn;
        table[ri].max_ms  = mx;
        table[ri].gflops  = flops_csr / (avg * 1e-3) / 1e9;
        table[ri].gbs     = bw_macko_gb(macko, n) / (avg * 1e-3);
        table[ri].correct = verify(y_ref_macko, y_tmp, n, 1e-6);
        ri++;
    }

    /* ---- 9. GPU CUDA MACKO FP16 (precisão mista, warp per row) ---- */
    {
        CUDAMACKOFp16Ctx *mctx16 = cuda_macko_fp16_ctx_create(macko, x);
        double sum = 0, mn = DBL_MAX, mx = 0;
        for (int r = 0; r < runs; r++) {
            float ms = cuda_macko_fp16_run(mctx16);
            if (r == 0) continue;
            sum += ms;
            if (ms < mn) mn = ms;
            if (ms > mx) mx = ms;
        }
        double avg = sum / (runs - 1);
        cuda_macko_fp16_ctx_result(mctx16, y_tmp);
        cuda_macko_fp16_ctx_destroy(mctx16);

        snprintf(table[ri].name, sizeof(table[ri].name), "GPU warp/row (MACKO FP16)");
        table[ri].avg_ms  = avg;
        table[ri].min_ms  = mn;
        table[ri].max_ms  = mx;
        table[ri].gflops  = flops_csr / (avg * 1e-3) / 1e9;
        table[ri].gbs     = bw_macko_fp16_gb(macko, n) / (avg * 1e-3);
        /* tolerância maior: FP16 tem ~3 casas decimais de precisão */
        table[ri].correct = verify(y_ref_macko, y_tmp, n, 1e-2);
        ri++;
    }

    /* ---- impressão ---- */
    print_table(table, N_IMPL);
    printf("\nNota: SP24 usa pruning 2:4 (preserva os 2 maiores |v| por grupo de 4).\n"
           "      Sua referência é a coluna CPU sequencial (SP24), não a CSR.\n"
           "      GFLOP/s e GB/s são teóricos (baseados em nnz e bytes transferidos).\n");

    /* ---- limpeza ---- */
    csr_free(csr);
    sp24_free(sp24);
    macko_free(macko);
    free(x); free(y_ref); free(y_ref_sp24); free(y_ref_macko); free(y_tmp);
    return 0;
}
