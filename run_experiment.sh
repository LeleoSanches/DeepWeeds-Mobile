#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-pipeline.py}"
CV_SCRIPT="${CV_SCRIPT:-cross-validation.py}"

LOGDIR="${LOGDIR:-logs_runs}"
FOLDS_ROOT="${FOLDS_ROOT:-labels}"

BATCH_SIZE="${BATCH_SIZE:-32}"
TRAINING_EPOCHS="${TRAINING_EPOCHS:-200}"
FINETUNNING_EPOCHS="${FINETUNNING_EPOCHS:-100}"

# controla se vai reconstruir os folds antes de iniciar os treinos
RUN_CV_FIRST="${RUN_CV_FIRST:-1}"

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
mkdir -p "$FOLDS_ROOT"

echo "[INFO] Script: $PIPELINE_SCRIPT"
echo "[INFO] Cross-validation script: $CV_SCRIPT"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Logs: $LOGDIR"
echo "[INFO] Folds root: $FOLDS_ROOT"
echo "[INFO] Batch size: $BATCH_SIZE"
echo "[INFO] Training epochs: $TRAINING_EPOCHS"
echo "[INFO] Finetunning epochs: $FINETUNNING_EPOCHS"
echo "[INFO] Run CV first: $RUN_CV_FIRST"
echo

# ============================================================
# 1) Gera folds uma única vez
# ============================================================
if [[ "$RUN_CV_FIRST" == "1" ]]; then
  ts_cv="$(date +'%Y%m%d_%H%M%S')"
  cv_log="${LOGDIR}/cross_validation_${ts_cv}.log"

  echo "============================================================"
  echo "[RUN] Gerando folds com $CV_SCRIPT"
  echo "============================================================"

  "$PYTHON_BIN" "$CV_SCRIPT" 2>&1 | tee "$cv_log"

  echo "[INFO] Cross-validation finalizado. Log salvo em: $cv_log"
  echo
else
  echo "[INFO] RUN_CV_FIRST=0 -> reutilizando folds existentes."
  echo
fi

# ============================================================
# 2) Verifica se os folds existem antes de treinar
# ============================================================
for fold in "${FOLDS[@]}"; do
  train_csv="${FOLDS_ROOT}/${fold}/train.csv"
  val_csv="${FOLDS_ROOT}/${fold}/val.csv"
  test_csv="${FOLDS_ROOT}/${fold}/test.csv"

  if [[ ! -f "$train_csv" ]]; then
    echo "[ERROR] Arquivo não encontrado: $train_csv"
    exit 1
  fi

  if [[ ! -f "$val_csv" ]]; then
    echo "[ERROR] Arquivo não encontrado: $val_csv"
    exit 1
  fi

  if [[ ! -f "$test_csv" ]]; then
    echo "[ERROR] Arquivo não encontrado: $test_csv"
    exit 1
  fi
done

echo "[INFO] Todos os folds foram encontrados com sucesso."
echo

# ============================================================
# 3) Roda os treinamentos usando os mesmos folds
# ============================================================
for m in "${MODELS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    echo "============================================================"
    echo "[RUN] model=$m | fold=$fold"
    echo "============================================================"

    train_csv="${FOLDS_ROOT}/${fold}/train.csv"
    val_csv="${FOLDS_ROOT}/${fold}/val.csv"
    test_csv="${FOLDS_ROOT}/${fold}/test.csv"

    ts="$(date +'%Y%m%d_%H%M%S')"
    logfile="${LOGDIR}/${m}_${fold}_bs${BATCH_SIZE}_ep${TRAINING_EPOCHS}_ft${FINETUNNING_EPOCHS}_${ts}.log"

    {
      echo "[INFO] Início: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "[INFO] Model: $m"
      echo "[INFO] Fold: $fold"
      echo "[INFO] Train CSV disponível em: $train_csv"
      echo "[INFO] Val CSV disponível em: $val_csv"
      echo "[INFO] Test CSV disponível em: $test_csv"
      echo "[INFO] Batch size: $BATCH_SIZE"
      echo "[INFO] Training epochs: $TRAINING_EPOCHS"
      echo "[INFO] Finetunning epochs: $FINETUNNING_EPOCHS"
      echo "------------------------------------------------------------"
    } | tee "$logfile"

    "$PYTHON_BIN" "$PIPELINE_SCRIPT" \
      --model "$m" \
      --fold "$fold" \
      --batch_size "$BATCH_SIZE" \
      --training_epochs "$TRAINING_EPOCHS" \
      --finetunning_epochs "$FINETUNNING_EPOCHS" \
      2>&1 | tee -a "$logfile"

    echo "[INFO] Fim: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$logfile"
    echo "[DONE] model=$m | fold=$fold" | tee -a "$logfile"
    echo | tee -a "$logfile"
  done
done

echo "[DONE] Todos os experimentos foram executados."