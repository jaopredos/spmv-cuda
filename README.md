# Benchmark de SpMV para LLMs Podados — CAD/UFG

Benchmark de **Sparse Matrix-Vector Multiplication (SpMV)** comparando formatos de
armazenamento esparso para matrizes de pesos de LLMs podados (esparsidade
30–90%), em CPU (sequencial/OpenMP) e GPU (CUDA), incluindo um formato
customizado próprio (**MACKO**) e a biblioteca de referência da NVIDIA
(**cuSPARSE**).

Trabalho da disciplina de Computação de Alto Desempenho — UFG.

---

## Formatos implementados

| Formato | Descrição |
|---|---|
| **CSR** | Compressed Sparse Row — referência clássica (`values`, `col_idx`, `row_ptr`) |
| **SP24** | 2:4 Structured Sparse (padrão NVIDIA Ampere) — pruning que mantém 2 de cada 4 elementos por grupo de colunas |
| **MACKO** | Formato customizado próprio, inspirado no paper MACKO: agrupa elementos em *chunks* de tamanho fixo, com layout SoA (Structure of Arrays), coluna delta-encoded em `int16` e máscara de validade por bit |
| **MACKO FP16** | Variante de precisão mista do MACKO: valores armazenados em `__half` (2 bytes) no device, com acumulação em `double` |
| **cuSPARSE** | Biblioteca oficial de álgebra linear esparsa da NVIDIA, usada como baseline de referência da indústria (API genérica `cusparseSpMV`) |

Cada formato tem (quando aplicável) uma implementação CPU sequencial, CPU
OpenMP e kernel CUDA, todas comparadas na mesma tabela do benchmark.

---

## Estrutura do repositório

```
include/
  matrix.h        — Matrix densa (double, row-major)
  csr.h           — CSRMatrix + protótipos
  sp24.h          — SP24Matrix (2:4 structured sparse) + protótipos
  macko.h         — MACKOMatrix (SoA) + protótipos
  spmv_cpu.h      — protótipos das implementações CPU
  spmv_cuda.h     — contextos e protótipos GPU (CSR, MACKO, MACKO FP16, cuSPARSE)

src/
  matrix.c        — geração de matriz esparsa aleatória, get/set/print
  csr.c           — conversão Matrix→CSR, SpMV sequencial de referência, verificação
  sp24.c          — conversão Matrix→SP24 (pruning 2:4)
  macko.c         — conversão CSR→MACKO (SoA), SpMV CPU sequencial/OpenMP
  spmv_cpu.c      — SpMV CPU sequencial/OpenMP (CSR e SP24)
  spmv_cuda.cu    — kernel CSR "thread per row", contexto persistente, cuSPARSE
  spmv_cuda_macko.cu — kernels MACKO (thread/warp per row) FP64 e FP16
  generator.c     — utilitário de debug: gera e imprime uma matriz pequena
  benchmark.cu    — harness principal com a tabela de comparação

run_benchmarks.sh — roda o benchmark em várias configurações (n, esparsidade)
                    e salva as saídas em arquivos organizados
Makefile          — build (gcc para C, nvcc para CUDA)
```

---

## Requisitos

- GPU NVIDIA com suporte a CUDA (testado em RTX 4090, `sm_89`)
- CUDA Toolkit 12.x (`nvcc`, headers `cuda_fp16.h`, biblioteca `cusparse`)
- `gcc` com suporte a OpenMP (`-fopenmp`)
- Linux (testado em Ubuntu, ambiente de laboratório)

Se a GPU não for uma RTX 4090 (`sm_89`), ajuste a variável `ARCH` no `Makefile`
para a arquitetura correta (ex.: `-arch=sm_86` para Ampere, `-arch=sm_90` para
Hopper).

---

## Build

```bash
make            # compila benchmark e generator
make clean      # remove binários e objetos
```

Gera dois executáveis:
- `benchmark` — harness principal de comparação
- `generator` — utilitário de debug (imprime matriz densa, CSR e SP24 de uma matriz pequena)

---

## Uso

### Benchmark principal

```bash
./benchmark <dimensao> <esparsidade> [runs=50]
```

- `dimensao` — tamanho da matriz quadrada (`n x n`)
- `esparsidade` — fração de elementos zero, entre `0.0` e `1.0`
- `runs` — número de execuções por implementação (1ª descartada como warm-up); padrão 50

Exemplos:

```bash
./benchmark 512 0.7 10          # teste rápido
./benchmark 4096 0.85 101       # benchmark típico de esparsidade de LLM podado
./benchmark 32768 0.85 101      # escala maior, regime de banda real (fora do L2 cache)
```

A saída é uma tabela com tempo médio/mínimo/máximo, GFLOP/s, GB/s (teóricos) e
verificação de corretude para cada implementação.

### Utilitário de debug

```bash
./generator <dimensao> <esparsidade>
```

Imprime a matriz densa, sua conversão CSR e SP24, e o resultado do SpMV —
útil para inspecionar manualmente matrizes pequenas (`n <= 16` é razoável).

### Automatizando várias configurações

```bash
./run_benchmarks.sh
```

Roda `./benchmark` em várias combinações de `n` e esparsidade (editáveis no
topo do script) e salva cada saída em `results/n<N>_s<SPARSITY>_r<RUNS>.txt`,
além de um log combinado `results/all_<timestamp>.log`.

---

## Notas sobre os resultados

- **GFLOP/s e GB/s são teóricos** — calculados a partir do número de
  operações/bytes esperados pela estrutura de dados, divididos pelo tempo
  medido. Não são medidos via profiler.
- **Em matrizes pequenas (`n` baixo), os dados podem caber no L2 cache da
  GPU** (72 MB na RTX 4090), inflando o `GB/s` aparente acima do pico nominal
  de banda da memória principal. Para avaliar ganhos de banda reais, teste em
  `n` grande o suficiente para exceder o cache (`n >= 16384` com esparsidade
  0.85, nesta GPU).
- A referência de corretude do **SP24** é a própria sequencial SP24, não o
  CSR — o pruning 2:4 altera o resultado numérico por design.
- A tolerância de verificação é `1e-9` para CPU, `1e-6` para GPU FP64, e
  `1e-2` para MACKO FP16 (compatível com a precisão de ~3 dígitos decimais do
  `__half`).

---

## Profiling

- **`nvprof` não funciona** em GPUs Ampere/Ada (`sm_80+`) — use as ferramentas
  Nsight do CUDA 12.x:
  - `nsys profile -o trace ./benchmark <n> <esparsidade> <runs>` seguido de
    `nsys stats trace.nsys-rep` — não exige permissões elevadas, dá tempo de
    kernel e contagem de chamadas de API/memória.
  - `ncu` (Nsight Compute) dá métricas de hardware (cache hit rate,
    ocupância, divergência), mas exige acesso a contadores de performance da
    GPU, frequentemente bloqueado em laboratórios compartilhados
    (`ERR_NVGPUCTRPERM`). Nesse caso, peça ao administrador para rodar com
    `sudo` ou liberar `NVreg_RestrictProfilingToAdminUsers=0`.

---

## Documentos de desenvolvimento

- `evolucao_macko.md` — análise da evolução do formato MACKO em três
  iterações (AoS int32 → AoS int16+mask → SoA), com hipóteses testadas para
  cada resultado de desempenho.
- `resultados.md` — saídas brutas do benchmark coletadas durante o
  desenvolvimento.
- `prompt_*.md` — especificações usadas para guiar cada etapa da
  implementação (formato MACKO inicial, delta encoding, SoA, FP16).
