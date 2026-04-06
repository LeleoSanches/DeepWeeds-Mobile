#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-pipeline.py}"
LOGDIR="${LOGDIR:-logs_smoke}"

BATCH_SIZE="${BATCH_SIZE:-8}"
TRAINING_EPOCHS="${TRAINING_EPOCHS:-1}"
FINETUNNING_EPOCHS="${FINETUNNING_EPOCHS:-1}"

DATASETS=(
  deepweeds
  weed6c
)

MODELS=(
  mobilenetv2
  mobilenetv3large
  mobilenetv3small
  efficientnetv2b0
  efficientnetv2b1
  efficientnetv2b2
  efficientnetv2b3
  resnet50
  resnet101v2
  nasnetmobile
  inceptionv3
)

FOLDS=(
  fold_1
  fold_2
  fold_3
)

mkdir -p "$LOGDIR"

echo "[INFO] Iniciando smoke test completo..."
echo "[INFO] Logs: $LOGDIR"
echo

for dataset in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      echo "============================================================"
      echo "[TEST] dataset=$dataset | model=$m | fold=$fold"
      echo "============================================================"

      ts="$(date +'%Y%m%d_%H%M%S')"
      logfile="${LOGDIR}/smoke_${dataset}_${m}_${fold}_${ts}.log"

      "$PYTHON_BIN" "$PIPELINE_SCRIPT" \
        --dataset "$dataset" \
        --model "$m" \
        --fold "$fold" \
        --batch_size "$BATCH_SIZE" \
        --training_epochs "$TRAINING_EPOCHS" \
        --finetunning_epochs "$FINETUNNING_EPOCHS" \
        2>&1 | tee "$logfile"

      echo "[OK] dataset=$dataset | model=$m | fold=$fold"
      echo
    done
  done
done

echo "[DONE] Smoke test finalizado com sucesso."