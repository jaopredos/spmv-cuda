#!/bin/bash
# Roda ./benchmark em várias combinações de (n, esparsidade) e salva as
# saídas em arquivos organizados, mais um log combinado com tudo em ordem.
#
# Uso:
#   ./run_benchmarks.sh
#
# Edite os arrays NS e SPARSITIES abaixo para mudar as configurações testadas.

set -euo pipefail

BENCH="./benchmark"
OUTDIR="results"
RUNS=101

# Tamanhos de matriz a testar (cresce bastante o tempo de CPU sequencial
# em n grande — veja a nota de duração estimada ao final do script).
NS=(4096 16384 32768)

# Esparsidades a testar em cada tamanho.
SPARSITIES=(0.5 0.7 0.85 0.9)

if [ ! -x "$BENCH" ]; then
    echo "Erro: '$BENCH' não encontrado ou não é executável. Rode 'make' primeiro." >&2
    exit 1
fi

mkdir -p "$OUTDIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$OUTDIR/all_${TIMESTAMP}.log"

total=$(( ${#NS[@]} * ${#SPARSITIES[@]} ))
i=0

echo "Iniciando $total combinações (runs=$RUNS cada). Log combinado: $LOGFILE"
echo

for n in "${NS[@]}"; do
    for s in "${SPARSITIES[@]}"; do
        i=$((i + 1))
        outfile="$OUTDIR/n${n}_s${s}_r${RUNS}.txt"

        echo "[$i/$total] n=$n esparsidade=$s runs=$RUNS -> $outfile"
        t0=$(date +%s)

        {
            echo "=== n=$n esparsidade=$s runs=$RUNS  $(date) ==="
            "$BENCH" "$n" "$s" "$RUNS"
            echo
        } > "$outfile"

        cat "$outfile" >> "$LOGFILE"

        t1=$(date +%s)
        echo "    concluído em $((t1 - t0))s"
    done
done

echo
echo "Tudo pronto."
echo "  Arquivos individuais: $OUTDIR/n<N>_s<SPARSITY>_r${RUNS}.txt"
echo "  Log combinado:        $LOGFILE"
