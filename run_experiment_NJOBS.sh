#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-pipeline.py}"
CV_SCRIPT="${CV_SCRIPT:-cross-validation.py}"

LOGDIR="${LOGDIR:-logs_runs_parallel}"
FOLDS_ROOT="${FOLDS_ROOT:-labels}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

BATCH_SIZE="${BATCH_SIZE:-32}"
TRAINING_EPOCHS="${TRAINING_EPOCHS:-200}"
FINETUNNING_EPOCHS="${FINETUNNING_EPOCHS:-100}"

RUN_CV_FIRST="${RUN_CV_FIRST:-1}"
MAX_JOBS="${MAX_JOBS:-2}"

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

mkdir -p "$LOGDIR" "$RESULTS_ROOT" "$FOLDS_ROOT"

if [[ "$RUN_CV_FIRST" == "1" ]]; then
  ts_cv="$(date +'%Y%m%d_%H%M%S')"
  cv_log="${LOGDIR}/cross_validation_${ts_cv}.log"
  "$PYTHON_BIN" "$CV_SCRIPT" 2>&1 | tee "$cv_log"
fi

for fold in "${FOLDS[@]}"; do
  [[ -f "${FOLDS_ROOT}/${fold}/train.csv" ]] || { echo "[ERROR] train.csv ausente em ${fold}"; exit 1; }
  [[ -f "${FOLDS_ROOT}/${fold}/val.csv" ]]   || { echo "[ERROR] val.csv ausente em ${fold}"; exit 1; }
  [[ -f "${FOLDS_ROOT}/${fold}/test.csv" ]]  || { echo "[ERROR] test.csv ausente em ${fold}"; exit 1; }
done

run_job() {
  local model="$1"
  local fold="$2"

  local train_csv="${FOLDS_ROOT}/${fold}/train.csv"
  local val_csv="${FOLDS_ROOT}/${fold}/val.csv"
  local test_csv="${FOLDS_ROOT}/${fold}/test.csv"
  local out_dir="${RESULTS_ROOT}/${model}/${fold}"

  mkdir -p "$out_dir"

  local ts
  ts="$(date +'%Y%m%d_%H%M%S')"
  local logfile="${LOGDIR}/${model}_${fold}_bs${BATCH_SIZE}_ep${TRAINING_EPOCHS}_ft${FINETUNNING_EPOCHS}_${ts}.log"

  {
    echo "[INFO] Início: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[INFO] Model: $model"
    echo "[INFO] Fold: $fold"
    echo "[INFO] Output dir: $out_dir"
    echo "------------------------------------------------------------"
  } | tee "$logfile"

  "$PYTHON_BIN" "$PIPELINE_SCRIPT" \
    --model "$model" \
    --fold "$fold" \
    --train_csv "$train_csv" \
    --val_csv "$val_csv" \
    --test_csv "$test_csv" \
    --output_dir "$out_dir" \
    --batch_size "$BATCH_SIZE" \
    --training_epochs "$TRAINING_EPOCHS" \
    --finetunning_epochs "$FINETUNNING_EPOCHS" \
    2>&1 | tee -a "$logfile"

  echo "[DONE] model=$model | fold=$fold" | tee -a "$logfile"
}

running_jobs=0

for model in "${MODELS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    run_job "$model" "$fold" &
    ((running_jobs+=1))

    if (( running_jobs >= MAX_JOBS )); then
      wait -n
      ((running_jobs-=1))
    fi
  done
done

wait

echo "[DONE] Todos os experimentos paralelos foram executados."