#!/usr/bin/env bash
# run_corpora_pipeline.sh
#
# Applies the full annotation-model pipeline to every corpus source under
# corpora/, ensuring EQUAL representation of each source in every split.
#
# Strategy for equal representation:
#   - Each source is extracted independently with the same --max-positive cap.
#   - The resulting per-source train/valid/test JSONL files are merged and
#     shuffled so each split contains an equal share from every source.
#   - The merged data is fed to the model trainer.
#
# Usage:
#   ./run_corpora_pipeline.sh [OPTIONS]
#
# Options:
#   --corpora-dir DIR     Root directory with one subdir per source
#                         (default: ../corpora)
#   --output-dir DIR      Where to write per-source data, merged data, and model
#                         (default: ../claude_pipeline_output)
#   --max-positive N      Max positive examples extracted per source (default: 10000)
#   --neg-ratio R         Negatives-to-positives ratio per source (default: 1.0)
#   --base-model MODEL    HuggingFace model name (default: deepset/gbert-base)
#   --epochs N            Training epochs (default: 3)
#   --batch-size N        Training batch size (default: 16)
#   --seed N              Random seed — applied to extraction, merging, training
#                         (default: 42)
#   --skip-training       Extract and merge only; skip model training/eval
#   -h, --help            Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────────────
CORPORA_DIR="$REPO_ROOT/corpora"
OUTPUT_DIR="$REPO_ROOT/claude_pipeline_output"
MAX_POSITIVE=10000
NEG_RATIO=1.0
BASE_MODEL="deepset/gbert-base"
EPOCHS=3
BATCH_SIZE=16
SEED=42
SKIP_TRAINING=false

# ── Argument parsing ──────────────────────────────────────────────────────────
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

# ── Helper ────────────────────────────────────────────────────────────────────
banner() {
    echo
    echo "════════════════════════════════════════════════"
    printf "  %s\n" "$*"
    echo "════════════════════════════════════════════════"
}

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -d "$CORPORA_DIR" ]]; then
    echo "ERROR: corpora directory not found: $CORPORA_DIR" >&2
    exit 1
fi

# Collect corpus source directories; skip plain files (e.g. index.html).
mapfile -t SOURCES < <(find "$CORPORA_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

if [[ ${#SOURCES[@]} -eq 0 ]]; then
    echo "ERROR: no source directories found under $CORPORA_DIR" >&2
    exit 1
fi

# ── Print plan ────────────────────────────────────────────────────────────────
banner "Gendercheck corpora pipeline"
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

# ── Step 1: Extract data from each source independently ───────────────────────
banner "Step 1/3 — Extracting training data (equal quota per source)"

SOURCE_DATA_DIRS=()
for SOURCE_DIR in "${SOURCES[@]}"; do
    SOURCE_NAME="$(basename "$SOURCE_DIR")"
    SOURCE_DATA="$OUTPUT_DIR/data_${SOURCE_NAME}"
    SOURCE_DATA_DIRS+=("$SOURCE_DATA")

    echo
    echo "  ── $SOURCE_NAME ──"

    # Detect whether this source stores articles without a .html extension.
    # www.woz.ch is the known case: articles are saved as bare paths (no extension).
    # We sniff by checking if any non-.html file in the first directory level looks
    # like HTML.  If so we pass --all-files so the extractor checks content, not ext.
    ALL_FILES_FLAG=""
    SAMPLE_EXT_COUNT="$(find "$SOURCE_DIR" -maxdepth 3 -type f ! -name '*.html' \
                        ! -name '*.htm' | head -5 | wc -l)"
    if [[ "$SAMPLE_EXT_COUNT" -gt 0 ]]; then
        echo "  (extensionless files detected — using --all-files)"
        ALL_FILES_FLAG="--all-files"
    fi

    python3 "$SCRIPT_DIR/01_extract_training_data.py" \
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

# ── Step 2: Merge per-source splits and shuffle ───────────────────────────────
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

# ── Step 3: Train and evaluate ────────────────────────────────────────────────
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
echo
echo "  Annotate new text:"
echo "    echo 'Die Lehrer:innen kommen morgen.' | \\"
echo "      python3 $SCRIPT_DIR/04_annotate.py $MODEL_DIR"
