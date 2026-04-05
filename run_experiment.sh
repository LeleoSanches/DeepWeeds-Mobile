#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-pipeline.py}"
LOGDIR="${LOGDIR:-logs_runs}"

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

mkdir -p "$LOGDIR"

echo "[INFO] Script: $PIPELINE_SCRIPT"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Logs: $LOGDIR"
echo

for m in "${MODELS[@]}"; do
  echo "============================================================"
  echo "[RUN] model=$m"
  echo "============================================================"

  ts="$(date +'%Y%m%d_%H%M%S')"
  logfile="${LOGDIR}/${m}_${ts}.log"

  "$PYTHON_BIN" "$PIPELINE_SCRIPT" --model "$m" 2>&1 | tee "$logfile"
done

echo
echo "[DONE] Todos os experimentos foram executados."