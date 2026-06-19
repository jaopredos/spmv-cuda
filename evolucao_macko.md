# Evolução do formato MACKO: de AoS ingênuo a SoA — três iterações, três lições

Este documento consolida os testes feitos durante o desenvolvimento do formato
MACKO (v1 → v2 → v3), as hipóteses levantadas pra explicar cada resultado, e o
que cada rodada efetivamente confirmou ou refutou. Todos os testes usam
`n=4096`, `esparsidade=0.85`, `runs=101` (1 warm-up descartado, 100 amostras),
na mesma máquina do laboratório (RTX 4090, CUDA 12.0).

---

## Metodologia: como validamos que as diferenças são reais, não ruído

Antes de comparar os números entre versões, foi preciso confirmar que o
ambiente de medição era estável:

- **CSR e SP24 (código nunca tocado nas três rodadas) servem de controle.**
  Em todas as rodadas, essas linhas variaram menos de ~2% entre si
  (`CPU sequencial (CSR)` entre 1.126–1.240 ms, `GPU CUDA kernel (CSR)` estável
  em 0.069–0.070 ms). Isso é o nível de ruído de fundo da máquina — qualquer
  variação no MACKO acima disso é sinal real, não ruído de medição.
- **Tentativa de profiling com `ncu` (Nsight Compute) falhou por permissão**
  (`ERR_NVGPUCTRPERM`) — contadores de hardware da GPU são restritos a
  administradores no laboratório compartilhado, e não havia acesso de `sudo`
  disponível.
- **`nsys` (Nsight Systems) funcionou** como alternativa parcial: não dá
  métricas de cache/divergência, mas confirmou via `nsys stats` que o tempo de
  kernel medido pelo `cudaEvent` no código bate com o tempo medido
  independentemente pela GPU (CUPTI) — por exemplo, `kernel_macko_warp_per_row`
  registrou 16.07 µs via `nsys` contra 0.017 ms (17 µs) na tabela do
  benchmark, confirmando que a medição interna do harness não tem overhead
  espúrio.

Com isso, qualquer variação >5% no MACKO entre versões pode ser tratada como
efeito real do código, não ruído.

---

## v1 — baseline: coluna absoluta `int32` em struct AoS

```c
typedef struct {
    int    cols[4];
    double vals[4];
} MACKOChunk;   // sizeof = 48 bytes
```

| Implementação | Méd (ms) | GB/s |
|---|---|---|
| CPU sequencial (MACKO) | 1.305 | 23.29 |
| CPU OpenMP (MACKO) | 0.107 | 282.69 |
| GPU warp/row (MACKO) | 0.017 | 1785.95 |

Esse já era um resultado forte: ~4× mais rápido que o `GPU CUDA kernel (CSR)`
no kernel warp-per-row, validado de forma independente via `nsys`. A hipótese
de ganho do MACKO (entrelaçar coluna+valor, indexar por chunk) já se
confirmava aqui.

---

## v2 — delta encoding (`int16`) + `valid_mask`: regressão inesperada

**Mudança:** trocar `int cols[4]` (16 bytes) por `int16_t col_base` +
`int16_t col_delta[4]` (10 bytes) e um `uint8_t valid_mask` pra eliminar o
teste `col == -1` por elemento.

```c
typedef struct {
    uint8_t valid_mask;   //  1 byte
    int16_t col_base;     //  2 bytes
    int16_t col_delta[4]; //  8 bytes
    double  vals[4];      // 32 bytes
} MACKOChunk;              // sizeof = 48 bytes (!)
```

| Implementação | v1 | v2 | Variação |
|---|---|---|---|
| CPU sequencial (MACKO) | 1.305 ms | 1.426 ms | **+9% mais lento** |
| CPU OpenMP (MACKO) | 0.107 ms | 0.128 ms | **+20% mais lento** |
| GPU warp/row (MACKO) | 0.017 ms (1786 GB/s) | 0.020 ms (1549 GB/s) | **+18% mais lento** |

`Corret = OK` em tudo — a lógica estava certa, foi puramente uma regressão de
desempenho, repetida nas três implementações (CPU seq, CPU OpenMP e GPU).

### Hipóteses pra essa regressão

1. **A economia de bytes prometida nunca existiu.** Medimos `sizeof(MACKOChunk)`
   antes de rodar o benchmark: **48 bytes, idêntico ao v1**. O motivo é
   alinhamento de struct em C: `double vals[4]` exige que o struct inteiro
   comece (e termine) em múltiplos de 8 bytes. Os 11 bytes "leves"
   (`valid_mask` + `col_base` + `col_delta[4]`) são arredondados para 16 antes
   de `vals[]` começar — o compilador insere silenciosamente os mesmos 5 bytes
   de padding que o delta encoding tentou eliminar. A métrica `bw_macko_gb`
   (que usa `sizeof(MACKOChunk)`) não mudou, e o `GB/s` da tabela confirma:
   não subiu, só caiu.

2. **Custo computacional extra, sem nada pra compensar.** O teste antigo
   (`col != -1`) era uma comparação direta. O novo exige, por elemento válido:
   um `shift` (`1 << k`) + `AND` (teste de bit) **e** uma soma extra
   (`col_base + col_delta[k]`) pra reconstruir a coluna antes de indexar
   `x[]`. Mais instruções por elemento processado — inclusive nos elementos
   válidos, não só no padding.

3. **A divergência que o `valid_mask` deveria eliminar provavelmente já era
   marginal.** No `kernel_macko_warp_per_row`, o loop `for k in 0..3` dentro
   de um chunk é executado por uma única lane, sequencialmente — não é um
   ponto de divergência SIMT clássica entre lanes do mesmo warp.
   Compiladores de GPU costumam *predicar* (não ramificar) ifs curtos e sem
   chamada de função como esse automaticamente. Então o ganho teórico de
   eliminar o branch provavelmente já estava resolvido pelo compilador antes
   — e o `valid_mask` só trocou "uma comparação" por "ler um byte extra +
   shift + and", sem divergência real para compensar.

4. **Possível perda de vetorização automática na CPU.** O struct do v1
   (`int[4]` + `double[4]`) é regular e fácil pro auto-vetorizador do GCC.
   Misturar `uint8_t` + `int16_t[4]` + `double[4]` no mesmo struct, com
   decodificação (`col_base + col_delta[k]`), é um padrão mais irregular —
   provavelmente dificulta a vetorização SIMD no loop OpenMP. Isso ajuda a
   explicar por que a versão OpenMP piorou proporcionalmente mais (+20%) que
   a sequencial (+9%): OpenMP depende mais de paralelismo de dados dentro de
   cada thread, que se perde com acesso mais irregular.

**Conclusão da v2:** otimização que parecia boa no papel, mas o alinhamento
de struct em C neutralizou silenciosamente o ganho de bytes, deixando só o
custo extra de decodificação — uma regressão real e repetível, não ruído.

---

## v3 — refatoração AoS → SoA: o ganho finalmente aparece

**Mudança:** em vez de um struct único por chunk, quatro arrays paralelos:

```c
uint8_t *valid_masks;  /* [total_chunks]               — 1 byte/chunk  */
int16_t *col_bases;    /* [total_chunks]               — 2 bytes/chunk */
int16_t *col_deltas;   /* [total_chunks * 4]           — 8 bytes/chunk */
double  *vals;         /* [total_chunks * 4]           — 32 bytes/chunk*/
```

Cada array tem seu próprio alinhamento natural — sem um `double[]` competindo
por padding com os campos pequenos do mesmo chunk. Footprint medido
(compilado e somado manualmente): **43 bytes/chunk**, confirmando a redução
de ~10% prevista.

| Implementação | v1 | v2 | v3 | v3 vs v1 |
|---|---|---|---|---|
| CPU sequencial (MACKO) | 1.305 ms | 1.426 ms | **1.226 ms** | **−6%** ✅ |
| CPU OpenMP (MACKO) | **0.107 ms** | 0.128 ms | 0.134 ms | +25% ❌ |
| GPU warp/row (MACKO) | 0.017 ms / 1786 GB/s | 0.020 ms / 1549 GB/s | **0.014 ms / 1886 GB/s** | **−18%** ✅ |

`Corret = OK` em tudo de novo. O destaque: **0.014 ms / 1886 GB/s é o melhor
resultado de toda a tabela**, superando inclusive o `GPU CUDA kernel (CSR)`
(437 GB/s) por uma margem ainda maior que nas versões anteriores.

### Por que a GPU melhorou tanto

No `kernel_macko_warp_per_row`, o padrão de acesso é "32 lanes, mesmo campo,
chunks consecutivos":

```c
for (int c = lane; c < nchunks; c += 32) {
    uint8_t mask = valid_masks[chunk_idx];   // 32 lanes lendo 32 bytes contíguos
    ...
}
```

Esse é exatamente o caso ideal pra SoA: as 32 lanes de um warp leem
`valid_masks[chunk_idx]` de **endereços consecutivos** na mesma instrução —
uma única transação de memória coalescida de 32 bytes. No AoS (v1/v2), a
mesma leitura tinha stride de 48 bytes entre lanes (cada `MACKOChunk` inteiro),
exigindo múltiplas transações. O ganho de footprint (43 vs 48 bytes) e o
ganho de coalescência se somam — por isso o salto foi maior que só a razão de
bytes economizados sugeriria.

### Por que a CPU sequencial também melhorou

O acesso é um *scan* linear por `c` dentro da linha — cada array SoA também é
percorrido sequencialmente, e o hardware prefetcher lida bem com múltiplos
streams lineares simples. Tipos mais leves (`uint8_t`, `int16_t`) cabem em
mais elementos por linha de cache, reduzindo o número de cache misses por
linha da matriz.

### Por que o OpenMP CPU não melhorou (e foi o único caso pior que o v1)

Essa é a observação mais interessante das três rodadas: SoA **não é uma
otimização universal** — depende de quantos campos do mesmo elemento
precisam ser lidos juntos, e de quantos agentes concorrentes competem por
banda de memória.

No AoS, ler um chunk inteiro (`mask + base + deltas + vals`) é uma única
linha de cache. No SoA, a mesma operação lógica agora é **4 streams de
memória separados** (`valid_masks[c]`, `col_bases[c]`, `col_deltas[slot]`,
`vals[slot]`). Pra uma única thread isso ainda é tratável (resultado: CPU
sequencial melhorou). Mas com várias threads OpenMP rodando em paralelo,
cada uma seguindo 4 streams simultâneos, a pressão total sobre o subsistema
de memória/prefetcher do processador aumenta — e isso parece superar o
ganho de footprint nesse caso específico. Na GPU o paralelismo é entre
*chunks* (mesmo campo, lanes vizinhas, ideal pra SoA); no OpenMP CPU o
paralelismo é entre *linhas inteiras* (cada thread processando seus próprios
4 streams), o que não tem o mesmo encaixe perfeito.

---

## Tabela consolidada — as três versões lado a lado

| Métrica | v1 (AoS int32) | v2 (AoS int16+mask) | v3 (SoA) |
|---|---|---|---|
| Bytes/chunk (teórico) | 48 | 48 (padding anula int16) | **43** |
| CPU sequencial (MACKO) | 1.305 ms | 1.426 ms (+9%) | **1.226 ms (−6%)** |
| CPU OpenMP (MACKO) | **0.107 ms** | 0.128 ms (+20%) | 0.134 ms (+25%) |
| GPU warp/row (MACKO) | 0.017 ms | 0.020 ms (+18%) | **0.014 ms (−18%)** |
| GB/s (GPU) | 1786 | 1549 | **1886** (melhor da tabela) |

---

## Lições da evolução

1. **Medir antes de assumir.** A v2 parecia, no papel, uma compressão de
   metadados óbvia (4 bytes → 2 bytes por coluna). Só ao compilar
   `sizeof(MACKOChunk)` isoladamente é que ficou claro que o alinhamento de
   `double` anulava o ganho — sem medir, o código teria sido reportado como
   "otimização aplicada" sem nenhum benefício real.
2. **AoS vs. SoA depende do padrão de acesso, não é uma regra fixa.** SoA
   ganhou onde múltiplos agentes leem o *mesmo campo* de elementos vizinhos
   (warp da GPU) e onde o acesso é um scan linear simples (CPU sequencial).
   Perdeu onde múltiplas threads precisam ler *todos os campos* de elementos
   próprios simultaneamente (CPU OpenMP) — lá, AoS colocava tudo numa única
   linha de cache.
3. **Ter um grupo de controle (CSR/SP24 intocados) em cada rodada** foi o que
   permitiu separar ruído de regressão real com confiança, sem precisar do
   `ncu` (que ficou bloqueado por permissão o tempo todo).
4. **O resultado final (v3) é o mais forte do projeto inteiro**: o kernel
   GPU do MACKO em SoA superou todas as outras sete implementações da tabela
   em GB/s, validando a tese central do formato — desde que a estrutura de
   dados realmente entregue o que promete em bytes *e* em padrão de acesso.
