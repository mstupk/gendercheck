#!/usr/bin/env bash
# run_stage1_pipeline.sh
#
# Stage 1 (SPECIFICATION.md §3.2) equivalent of run_corpora_pipeline.sh:
# extracts MASCULINE-candidate-span data (via 01b_extract_masculine_spans.py,
# not 01_extract_training_data.py) from every corpus source under corpora/,
# with the same equal-quota-per-source strategy, then fine-tunes the
# gbert-base token classifier on it.
#
# Writes to claude_pipeline_output_stage1/ by convention -- kept completely
# separate from claude_pipeline_output/ (Approach A/B, gendered-span
# detection) so the two datasets/models are never confused.
#
# Usage:
#   ./run_stage1_pipeline.sh [OPTIONS]
#
# Options: identical to run_corpora_pipeline.sh -- see that script's header.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CORPORA_DIR="$REPO_ROOT/corpora"
OUTPUT_DIR="$REPO_ROOT/claude_pipeline_output_stage1"
MAX_POSITIVE=10000
NEG_RATIO=1.0
BASE_MODEL="deepset/gbert-base"
EPOCHS=3
BATCH_SIZE=16
SEED=42
SKIP_TRAINING=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --corpora-dir)   CORPORA_DIR="$2";  shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";   shift 2 ;;
        --max-positive)  MAX_POSITIVE="$2"; shift 2 ;;
        --neg-ratio)     NEG_RATIO="$2";    shift 2 ;;
        --base-model)    BASE_MODEL="$2";   shift 2 ;;
        --epochs)        EPOCHS="$2";       shift 2 ;;
        --batch-size)    BATCH_SIZE="$2";   shift 2 ;;
        --seed)          SEED="$2";         shift 2 ;;
        --skip-training) SKIP_TRAINING=true; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '/^Usage/,/^$/p'
            exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

MERGED_DIR="$OUTPUT_DIR/data_merged"
MODEL_DIR="$OUTPUT_DIR/model"

banner() {
    echo
    echo "════════════════════════════════════════════════"
    printf "  %s\n" "$*"
    echo "════════════════════════════════════════════════"
}

if [[ ! -d "$CORPORA_DIR" ]]; then
    echo "ERROR: corpora directory not found: $CORPORA_DIR" >&2
    exit 1
fi

mapfile -t SOURCES < <(find "$CORPORA_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#SOURCES[@]} -eq 0 ]]; then
    echo "ERROR: no source directories found under $CORPORA_DIR" >&2
    exit 1
fi

banner "Gendercheck Stage 1 pipeline (masculine-span detection)"
echo "  Corpora dir   : $CORPORA_DIR"
echo "  Output dir    : $OUTPUT_DIR"
echo "  Sources (${#SOURCES[@]})  :"
for s in "${SOURCES[@]}"; do
    n_files="$(find "$s" -type f | wc -l)"
    printf "    • %-30s  %6d files\n" "$(basename "$s")" "$n_files"
done
echo "  Max positive  : $MAX_POSITIVE per source  →  ~$((MAX_POSITIVE * ${#SOURCES[@]})) total"
echo "  Neg ratio     : $NEG_RATIO"
echo "  Base model    : $BASE_MODEL"
echo "  Epochs        : $EPOCHS  |  Batch: $BATCH_SIZE  |  Seed: $SEED"
echo "  Skip training : $SKIP_TRAINING"

mkdir -p "$OUTPUT_DIR" "$MERGED_DIR" "$MODEL_DIR"

banner "Step 1/3 — Extracting Stage 1 (masculine-span) data (equal quota per source)"

SOURCE_DATA_DIRS=()
for SOURCE_DIR in "${SOURCES[@]}"; do
    SOURCE_NAME="$(basename "$SOURCE_DIR")"
    SOURCE_DATA="$OUTPUT_DIR/data_${SOURCE_NAME}"
    SOURCE_DATA_DIRS+=("$SOURCE_DATA")

    echo
    echo "  ── $SOURCE_NAME ──"

    ALL_FILES_FLAG=""
    SAMPLE_EXT_COUNT="$(find "$SOURCE_DIR" -maxdepth 3 -type f ! -name '*.html' \
                        ! -name '*.htm' | head -5 | wc -l)"
    if [[ "$SAMPLE_EXT_COUNT" -gt 0 ]]; then
        echo "  (extensionless files detected — using --all-files)"
        ALL_FILES_FLAG="--all-files"
    fi

    python3 "$SCRIPT_DIR/01b_extract_masculine_spans.py" \
        "$SOURCE_DIR"  "$SOURCE_DATA" \
        --max-positive "$MAX_POSITIVE" \
        --neg-ratio    "$NEG_RATIO" \
        --seed         "$SEED" \
        --train-ratio  0.8 \
        --valid-ratio  0.1 \
        $ALL_FILES_FLAG

    echo "  Results for $SOURCE_NAME:"
    for split in train valid test; do
        f="$SOURCE_DATA/${split}.jsonl"
        if [[ -f "$f" ]]; then
            printf "    %-8s %6d examples\n" "$split" "$(wc -l < "$f")"
        fi
    done
done

banner "Step 2/3 — Merging splits (equal weight per source, then shuffle)"

python3 "$SCRIPT_DIR/_merge_splits.py" \
    --source-dirs "${SOURCE_DATA_DIRS[@]}" \
    --output-dir  "$MERGED_DIR" \
    --seed        "$SEED"

echo
echo "  Final merged dataset:"
TOTAL=0
for split in train valid test; do
    f="$MERGED_DIR/${split}.jsonl"
    if [[ -f "$f" ]]; then
        n="$(wc -l < "$f")"
        TOTAL=$((TOTAL + n))
        printf "    %-8s %6d examples\n" "$split" "$n"
    fi
done
echo "    ─────────────────────"
printf "    %-8s %6d total\n" "" "$TOTAL"

if [[ "$SKIP_TRAINING" == "true" ]]; then
    banner "Skipping training (--skip-training set)"
    echo "  Merged data: $MERGED_DIR"
    exit 0
fi

banner "Step 3a/3 — Fine-tuning $BASE_MODEL"
python3 "$SCRIPT_DIR/02_train_model.py" \
    "$MERGED_DIR" "$MODEL_DIR" \
    --base-model  "$BASE_MODEL" \
    --epochs      "$EPOCHS" \
    --batch-size  "$BATCH_SIZE"

banner "Step 3b/3 — Evaluating on test set"
python3 "$SCRIPT_DIR/03_evaluate.py" \
    "$MERGED_DIR" "$MODEL_DIR"

banner "Pipeline complete"
echo "  Model : $MODEL_DIR"
