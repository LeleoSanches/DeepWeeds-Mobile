#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIGURAÇÃO =========
PY=python              # ou python3
SCRIPT=benchmark.py    # nome do seu script de benchmark
OUTDIR="bench_out"     # pasta de resultados
mkdir -p "$OUTDIR"

# Args comuns a todos os testes
ARGS="--labels-csv labels/labels.csv --images-root images/ --batch-size 1 --steps 200"

# Lista de modelos a testar (ajuste conforme seus arquivos em models/)
MODELS=(
  "mobilenetv2"
  "mobilenetv3large"
  "mobilenetv3small"
  "resnet50"
  "resnet101v2"
  "inceptionv3"
  "nasnetmobile"
  "efficientnetv2b0"
  "efficientnetv2b1"
  "efficientnetv2b2"
  "efficientnetv2b3"
)

# Se algum modelo foi salvo com preprocess ASSADO no grafo,
# coloque o nome aqui para rodar sem --model-key.
BAKED_PREPROCESS_MODELS=(
  # exemplo: "mobilenetv2"
)

# ========= FUNÇÕES AUX =========
is_baked() {
  local m="$1"
  for x in "${BAKED_PREPROCESS_MODELS[@]}"; do
    [[ "$x" == "$m" ]] && return 0
  done
  return 1
}

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

# ========= LOOP PRINCIPAL =========
echo "[INFO $(timestamp)] Iniciando benchmarks..."
ALL_CSVS=()

for m in "${MODELS[@]}"; do
  MPATH="models/best_${m}_finetune.keras"
  OUTCSV="${OUTDIR}/bench-${m}.csv"

  if [[ ! -f "$MPATH" ]]; then
    echo "[WARN $(timestamp)] Modelo não encontrado: $MPATH — pulando."
    continue
  fi

  if is_baked "$m"; then
    echo "[INFO $(timestamp)] Rodando $m (preprocess embutido; sem --model-key)"
    $PY "$SCRIPT" --model-path "$MPATH" $ARGS --out-csv "$OUTCSV"
  else
    echo "[INFO $(timestamp)] Rodando $m"
    $PY "$SCRIPT" --model-path "$MPATH" --model-key "$m" $ARGS --out-csv "$OUTCSV"
  fi

  ALL_CSVS+=("$OUTCSV")
done

# ========= CONSOLIDAÇÃO =========
# Junta todos os CSVs em um só, mantendo apenas um cabeçalho
COMBINED="${OUTDIR}/bench_all.csv"
if [[ ${#ALL_CSVS[@]} -gt 0 ]]; then
  echo "[INFO $(timestamp)] Consolidando CSVs em: $COMBINED"
  # copia o primeiro com cabeçalho
  cp "${ALL_CSVS[0]}" "$COMBINED"
  # acrescenta os demais sem o cabeçalho
  for ((i=1; i<${#ALL_CSVS[@]}; i++)); do
    tail -n +2 "${ALL_CSVS[$i]}" >> "$COMBINED"
  done
  echo "[OK   $(timestamp)] Consolidação concluída."
else
  echo "[WARN $(timestamp)] Nenhum CSV gerado — verifique a lista de modelos/paths."
fi

echo "[DONE $(timestamp)] Benchmarks finalizados."
