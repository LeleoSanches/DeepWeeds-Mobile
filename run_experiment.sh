#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-pipeline.py}"
CV_SCRIPT="${CV_SCRIPT:-cross-validation.py}"

LOGDIR="${LOGDIR:-logs_runs}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAINING_EPOCHS="${TRAINING_EPOCHS:-200}"
FINETUNNING_EPOCHS="${FINETUNNING_EPOCHS:-100}"

# 1 = roda o script de geração de folds antes dos treinos
RUN_CV_FIRST="${RUN_CV_FIRST:-0}"

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

echo "[INFO] Script: $PIPELINE_SCRIPT"
echo "[INFO] Cross-validation script: $CV_SCRIPT"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Logs: $LOGDIR"
echo "[INFO] Batch size: $BATCH_SIZE"
echo "[INFO] Training epochs: $TRAINING_EPOCHS"
echo "[INFO] Finetunning epochs: $FINETUNNING_EPOCHS"
echo "[INFO] Run CV first: $RUN_CV_FIRST"
echo

# ============================================================
# 1) (Opcional) Gera folds por dataset
# ============================================================
if [[ "$RUN_CV_FIRST" == "1" ]]; then
  for dataset in "${DATASETS[@]}"; do
    ts_cv="$(date +'%Y%m%d_%H%M%S')"
    cv_log="${LOGDIR}/cross_validation_${dataset}_${ts_cv}.log"

    echo "============================================================"
    echo "[RUN] Gerando folds | dataset=$dataset"
    echo "============================================================"

    "$PYTHON_BIN" "$CV_SCRIPT" \
      --dataset "$dataset" \
      2>&1 | tee "$cv_log"

    echo "[INFO] Cross-validation finalizado para $dataset. Log: $cv_log"
    echo
  done
else
  echo "[INFO] RUN_CV_FIRST=0 -> reutilizando folds existentes."
  echo
fi

# ============================================================
# 2) Treinos
# ============================================================
for dataset in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      echo "============================================================"
      echo "[RUN] dataset=$dataset | model=$m | fold=$fold"
      echo "============================================================"

      ts="$(date +'%Y%m%d_%H%M%S')"
      logfile="${LOGDIR}/${dataset}_${m}_${fold}_bs${BATCH_SIZE}_ep${TRAINING_EPOCHS}_ft${FINETUNNING_EPOCHS}_${ts}.log"

      {
        echo "[INFO] Início: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "[INFO] Dataset: $dataset"
        echo "[INFO] Model: $m"
        echo "[INFO] Fold: $fold"
        echo "[INFO] Batch size: $BATCH_SIZE"
        echo "[INFO] Training epochs: $TRAINING_EPOCHS"
        echo "[INFO] Finetunning epochs: $FINETUNNING_EPOCHS"
        echo "------------------------------------------------------------"
      } | tee "$logfile"

      "$PYTHON_BIN" "$PIPELINE_SCRIPT" \
        --dataset "$dataset" \
        --model "$m" \
        --fold "$fold" \
        --batch_size "$BATCH_SIZE" \
        --training_epochs "$TRAINING_EPOCHS" \
        --finetunning_epochs "$FINETUNNING_EPOCHS" \
        2>&1 | tee -a "$logfile"

      echo "[INFO] Fim: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$logfile"
      echo "[DONE] dataset=$dataset | model=$m | fold=$fold" | tee -a "$logfile"
      echo | tee -a "$logfile"
    done
  done
done

echo "[DONE] Todos os experimentos foram executados."