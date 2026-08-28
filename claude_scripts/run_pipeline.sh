#!/usr/bin/env bash
# run_pipeline.sh — Full pipeline: data extraction → training → evaluation
#
# Usage:
#   ./run_pipeline.sh [corpus_dir]
#
# Arguments:
#   corpus_dir   Root directory of HTML corpus (default: ../corpora/taz.de)
#
# After running, annotate new text with:
#   echo "Die Lehrer:innen kommen morgen." | python 04_annotate.py model/

set -euo pipefail

CORPUS_DIR="${1:-../corpora/taz.de}"
DATA_DIR="data"
MODEL_DIR="model"
BASE_MODEL="deepset/gbert-base"

echo "========================================"
echo "Gendercheck annotation model pipeline"
echo "========================================"
echo "Corpus : $CORPUS_DIR"
echo "Data   : $DATA_DIR"
echo "Model  : $MODEL_DIR"
echo "Base   : $BASE_MODEL"
echo ""

# --- Step 1: Extract training data ---
echo "[1/3] Extracting training data ..."
python 01_extract_training_data.py "$CORPUS_DIR" "$DATA_DIR" \
  --max-positive 50000 \
  --neg-ratio 1.0 \
  --seed 42
echo ""

# --- Step 2: Fine-tune model ---
echo "[2/3] Fine-tuning model ..."
python 02_train_model.py "$DATA_DIR" "$MODEL_DIR" \
  --base-model "$BASE_MODEL" \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5
echo ""

# --- Step 3: Evaluate on test set ---
echo "[3/3] Evaluating ..."
python 03_evaluate.py "$DATA_DIR" "$MODEL_DIR"
echo ""

echo "========================================"
echo "Pipeline complete."
echo "Model saved to: $MODEL_DIR"
echo ""
echo "To annotate text:"
echo "  echo 'Die Lehrer:innen kommen morgen.' | python 04_annotate.py $MODEL_DIR"
echo "========================================"
