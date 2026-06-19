# Análise dos resultados

## Antes da mudança (v1)
```
aluno@ubuntu-lab150-e4:~/Downloads/trabalho-cad$ ./benchmark 4096 0.85 101

Benchmark SpMV  n=4096  esparsidade=0.85  nnz=2517868  runs=101 (1 warm-up descartado)

+-----------------------------------------+----------+----------+----------+---------+--------+--------+
| Implementação                         | Méd(ms) | Mín(ms) | Máx(ms) | GFLOP/s |   GB/s | Corret |
+-----------------------------------------+----------+----------+----------+---------+--------+--------+
| CPU sequencial (CSR)                    |    1.138 |    1.082 |    1.503 |    4.43 |  26.63 |    ref |
| CPU OpenMP (CSR)                        |    0.158 |    0.130 |    0.566 |   31.79 | 191.25 |     OK |
| CPU sequencial (SP24)                   |    4.604 |    4.516 |    5.064 |    3.64 |  15.50 |    ref |
| CPU OpenMP (SP24)                       |    0.618 |    0.555 |    1.537 |   27.13 | 115.43 |     OK |
| GPU CUDA kernel (CSR)                   |    0.069 |    0.069 |    0.071 |   72.50 | 436.17 |     OK |
| CPU sequencial (MACKO)                  |    1.305 |    1.261 |    1.931 |    3.86 |  23.29 |    ref |
| CPU OpenMP (MACKO)                      |    0.107 |    0.081 |    0.586 |   46.85 | 282.69 |     OK |
| GPU CUDA warp/row (MACKO)               |    0.017 |    0.016 |    0.017 |  295.96 | 1785.95 |     OK |
+-----------------------------------------+----------+----------+----------+---------+--------+--------+

Nota: SP24 usa pruning 2:4 (preserva os 2 maiores |v| por grupo de 4).
      Sua referência é a coluna CPU sequencial (SP24), não a CSR.
      GFLOP/s e GB/s são teóricos (baseados em nnz e bytes transferidos).

```


## Depois da mudança (v2):
```
aluno@ubuntu-lab150-e4:~/Downloads/trabalho-cad$ ./benchmark 4096 0.85 101

Benchmark SpMV  n=4096  esparsidade=0.85  nnz=2518625  runs=101 (1 warm-up descartado)

+-----------------------------------------+----------+----------+----------+---------+--------+--------+
| Implementação                         | Méd(ms) | Mín(ms) | Máx(ms) | GFLOP/s |   GB/s | Corret |
+-----------------------------------------+----------+----------+----------+---------+--------+--------+
| CPU sequencial (CSR)                    |    1.126 |    1.064 |    1.302 |    4.47 |  26.90 |    ref |
| CPU OpenMP (CSR)                        |    0.154 |    0.133 |    0.496 |   32.64 | 196.36 |     OK |
| CPU sequencial (SP24)                   |    4.605 |    4.515 |    5.070 |    3.64 |  15.50 |    ref |
| CPU OpenMP (SP24)                       |    0.605 |    0.540 |    1.443 |   27.75 | 118.06 |     OK |
| GPU CUDA kernel (CSR)                   |    0.070 |    0.069 |    0.071 |   72.26 | 434.72 |     OK |
| CPU sequencial (MACKO)                  |    1.426 |    1.333 |    2.024 |    3.53 |  21.32 |    ref |
| CPU OpenMP (MACKO)                      |    0.128 |    0.094 |    0.653 |   39.44 | 237.96 |     OK |
| GPU CUDA warp/row (MACKO)               |    0.020 |    0.019 |    0.021 |  256.72 | 1549.05 |     OK |
+-----------------------------------------+----------+----------+----------+---------+--------+--------+

Nota: SP24 usa pruning 2:4 (preserva os 2 maiores |v| por grupo de 4).
      Sua referência é a coluna CPU sequencial (SP24), não a CSR.
      GFLOP/s e GB/s são teóricos (baseados em nnz e bytes transferidos).
```



## Ideia de por que a v2 ficou ruim

Boa notícia pro controle do experimento: CSR e SP24 (código que não tocamos) ficaram praticamente idênticos entre as
  duas rodadas — CPU sequencial (CSR) 1.138→1.126 ms, GPU CUDA kernel (CSR) 0.069→0.070 ms. Isso é só ruído normal
  (~1%), confirma que o ambiente estava estável e que a comparação é válida.

  Já o MACKO regrediu de forma consistente nas três implementações:

  ┌────────────────────────┬──────────────────────┬──────────────────────┬─────────────────┐
  │     Implementação      │      Antes (v1)      │     Depois (v2)      │    Variação     │
  ├────────────────────────┼──────────────────────┼──────────────────────┼─────────────────┤
  │ CPU sequencial (MACKO) │ 1.305 ms             │ 1.426 ms             │ +9% mais lento  │
  ├────────────────────────┼──────────────────────┼──────────────────────┼─────────────────┤
  │ CPU OpenMP (MACKO)     │ 0.107 ms             │ 0.128 ms             │ +20% mais lento │
  ├────────────────────────┼──────────────────────┼──────────────────────┼─────────────────┤
  │ GPU warp/row (MACKO)   │ 0.017 ms (1786 GB/s) │ 0.020 ms (1549 GB/s) │ +18% mais lento │
  └────────────────────────┴──────────────────────┴──────────────────────┴─────────────────┘

  Corret continua OK em tudo — a lógica está certa, é puramente uma regressão de desempenho. E isso é exatamente o que
  eu antecipei quando expliquei o problema do sizeof. Vamos decompor por que:

  Por que ficou mais lento, não mais rápido

  1. Não há economia de banda real. Como medimos antes, sizeof(MACKOChunk) continua 48 bytes, idêntico ao v1 — o double
  vals[4] força alinhamento de 8 bytes pro struct inteiro, e o espaço que o delta encoding "economizou" (4 bytes por
  slot) volta como padding do compilador. A métrica bw_macko_gb não mudou, e de fato o GB/s da tabela confirma isso: não
  subiu, só caiu.

  2. Custo computacional extra, sem compensação. Antes, o teste de padding era uma comparação direta: col != -1. Agora,
  pra cada elemento válido você paga:
  - um shift (1 << k) + AND (mask & ...) só pra saber se é válido,
  - mais uma soma (col_base + col_delta[k]) pra reconstruir a coluna absoluta antes de indexar x[].

  Isso é estritamente mais instruções por elemento do que antes, e isso acontece mesmo nos elementos válidos (não só no
  padding).

  3. A divergência que o valid_mask deveria eliminar provavelmente já era marginal. No kernel_macko_warp_per_row, cada
  lane processa chunks diferentes em sequência — o loop for k in 0..3 dentro de um chunk é executado por uma única
  thread, não é um ponto onde lanes do mesmo warp literalmente tomam caminhos diferentes ao mesmo tempo (divergência
  SIMT clássica). Compiladores de GPU normalmente já "predicam" ifs curtos e sem chamada de função (como esse) em vez de
  gerar branch real — então o ganho teórico de eliminar o branch provavelmente já estava sendo resolvido pelo
  compilador antes, e o valid_mask só trocou "uma comparação simples" por "ler um byte extra + shift + and", sem ganho
  real de divergência pra compensar.

  4. Possível perda de vetorização automática na CPU. O struct antigo (int cols[4] + double vals[4]) é mais regular pro
  auto-vetorizador do GCC/OpenMP processar; misturar uint8_t + int16_t[4] + double[4] no mesmo struct, com a
  decodificação col_base + col_delta[k], é um padrão de acesso mais irregular — pode estar dificultando a vetorização
  SIMD no loop OpenMP, o que ajudaria a explicar por que a versão OpenMP (que depende mais de vetorização) piorou
  proporcionalmente mais (+20%) que a sequencial (+9%).

  Conclusão honesta pro relatório

  A v2, como especificada (sem packed), não entrega o ganho prometido — é uma regressão medida e repetível, não ruído.
  Isso é, na verdade, um resultado interessante e legítimo pra documentar: mostra que "comprimir metadados" só compensa
  se a compressão de fato reduzir bytes transferidos/processados, e que alinhamento de struct em C pode neutralizar
  silenciosamente uma otimização que parece boa no papel.

## Nova mudança (v3):

```
aluno@ubuntu-lab150-e4:~/Downloads/trabalho-cad$ ./benchmark 4096 0.85 101

Benchmark SpMV  n=4096  esparsidade=0.85  nnz=2517316  runs=101 (1 warm-up descartado)

+-----------------------------------------+----------+----------+----------+---------+--------+--------+
| Implementação                         | Méd(ms) | Mín(ms) | Máx(ms) | GFLOP/s |   GB/s | Corret |
+-----------------------------------------+----------+----------+----------+---------+--------+--------+
| CPU sequencial (CSR)                    |    1.240 |    1.174 |    1.448 |    4.06 |  24.42 |    ref |
| CPU OpenMP (CSR)                        |    0.154 |    0.131 |    0.444 |   32.60 | 196.10 |     OK |
| CPU sequencial (SP24)                   |    4.640 |    4.541 |    5.101 |    3.62 |  15.38 |    ref |
| CPU OpenMP (SP24)                       |    0.638 |    0.549 |    1.447 |   26.31 | 111.93 |     OK |
| GPU CUDA kernel (CSR)                   |    0.069 |    0.069 |    0.070 |   72.66 | 437.17 |     OK |
| CPU sequencial (MACKO)                  |    1.226 |    1.185 |    1.696 |    4.11 |  22.21 |    ref |
| CPU OpenMP (MACKO)                      |    0.134 |    0.104 |    0.586 |   37.46 | 202.57 |     OK |
| GPU CUDA warp/row (MACKO)               |    0.014 |    0.014 |    0.015 |  348.77 | 1886.02 |     OK |
+-----------------------------------------+----------+----------+----------+---------+--------+--------+

Nota: SP24 usa pruning 2:4 (preserva os 2 maiores |v| por grupo de 4).
      Sua referência é a coluna CPU sequencial (SP24), não a CSR.
      GFLOP/s e GB/s são teóricos (baseados em nnz e bytes transferidos).
```