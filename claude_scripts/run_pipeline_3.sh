#!/usr/bin/env bash
# run_pipeline_3.sh
#
# Full pipeline that produces an OpenNMT-py 3.x training setup for the
# German gender-term annotation task (seq2seq, TOKEN-BASED vocabulary).
#
# The task is sequence-to-sequence annotation:
#   src:  Die Lehrer:innen kommen morgen .
#   tgt:  Die <gender> Lehrer:innen </gender> kommen morgen .
#
# <gender> and </gender> are plain whitespace-delimited tokens so they
# land in the target vocabulary as regular word types — no BPE or subword
# transforms needed.
#
# Steps
# ──────────────────────────────
#   1. Extract BIO-labeled data from each corpus source independently
#      (equal --max-positive quota per source, article-level splits).
#   2. Merge + shuffle per-source JSONL splits.
#   3. Convert BIO JSONL → OpenNMT .src / .tgt plain-text files.
#   4. Build token-based vocabulary (build_opennmt_vocab.py).
#   5. Write train_config.yaml ready for onmt_train.
#
# Training is NOT started.  The exact command is printed at the end.
#
# Usage
# ──────────────────────────────
#   cd claude_scripts/
#   ./run_pipeline_3.sh [OPTIONS]
#
# Options
#   --corpora-dir DIR      Root dir with one subdir per corpus source
#                          (default: ../corpora)
#   --output-dir DIR       All outputs land here (default: ../opennmt_run)
#   --max-positive N       Max positive examples per source (default: 10000)
#   --neg-ratio R          Negatives-to-positives ratio (default: 1.0)
#   --src-vocab-size N     Max src vocab entries (default: 50000)
#   --tgt-vocab-size N     Max tgt vocab entries (default: 50000)
#   --train-steps N        train_steps in generated config (default: 50000)
#   --valid-steps N        valid_steps in generated config (default: 5000)
#   --save-steps N         save_checkpoint_steps (default: 5000)
#   --seed N               Shared random seed (default: 42)
#   --gpu N                GPU index for training config (-1 = CPU, default: 0)
#   --onmt-dir DIR         Path to OpenNMT-py source tree
#                          (default: ../testruns/OpenNMT-py)
#   --skip-extract         Skip steps 1–2 (reuse existing merged JSONL)
#   --skip-convert         Skip step 3  (reuse existing .src/.tgt files)
#   --skip-vocab           Skip step 4  (reuse existing vocab files)
#   -h, --help             Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────────────
CORPORA_DIR="$REPO_ROOT/corpora"
OUTPUT_DIR="$REPO_ROOT/opennmt_run"
MAX_POSITIVE=10000
NEG_RATIO=1.0
SRC_VOCAB_SIZE=50000
TGT_VOCAB_SIZE=50000
TRAIN_STEPS=50000
VALID_STEPS=5000
SAVE_STEPS=5000
SEED=42
GPU=0
ONMT_DIR="$REPO_ROOT/testruns/OpenNMT-py"
SKIP_EXTRACT=false
SKIP_CONVERT=false
SKIP_VOCAB=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --corpora-dir)    CORPORA_DIR="$2";     shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";      shift 2 ;;
        --max-positive)   MAX_POSITIVE="$2";    shift 2 ;;
        --neg-ratio)      NEG_RATIO="$2";       shift 2 ;;
        --src-vocab-size) SRC_VOCAB_SIZE="$2";  shift 2 ;;
        --tgt-vocab-size) TGT_VOCAB_SIZE="$2";  shift 2 ;;
        --train-steps)    TRAIN_STEPS="$2";     shift 2 ;;
        --valid-steps)    VALID_STEPS="$2";     shift 2 ;;
        --save-steps)     SAVE_STEPS="$2";      shift 2 ;;
        --seed)           SEED="$2";            shift 2 ;;
        --gpu)            GPU="$2";             shift 2 ;;
        --onmt-dir)       ONMT_DIR="$2";        shift 2 ;;
        --skip-extract)   SKIP_EXTRACT=true;    shift ;;
        --skip-convert)   SKIP_CONVERT=true;    shift ;;
        --skip-vocab)     SKIP_VOCAB=true;      shift ;;
        -h|--help)
            sed -n '/^# Usage/,/^[^#]/{ /^[^#]/q; s/^# \{0,1\}//; p }' "$0"
            exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Derived paths ─────────────────────────────────────────────────────────────
MERGED_DIR="$OUTPUT_DIR/data_merged"
OPENNMT_DIR="$OUTPUT_DIR/opennmt_data"
VOCAB_DIR="$OUTPUT_DIR/vocab"
MODEL_DIR="$OUTPUT_DIR/model"
PREPARED_DIR="$OUTPUT_DIR/prepared"
CONFIG_FILE="$OUTPUT_DIR/train_config.yaml"

# ── Helper ────────────────────────────────────────────────────────────────────
banner() {
    echo
    echo "════════════════════════════════════════════════════════"
    printf "  %s\n" "$*"
    echo "════════════════════════════════════════════════════════"
}

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -d "$CORPORA_DIR" && "$SKIP_EXTRACT" == "false" ]]; then
    echo "ERROR: corpora directory not found: $CORPORA_DIR" >&2; exit 1
fi

mapfile -t SOURCES < <(find "$CORPORA_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
if [[ ${#SOURCES[@]} -eq 0 && "$SKIP_EXTRACT" == "false" ]]; then
    echo "ERROR: no source subdirectories found under $CORPORA_DIR" >&2; exit 1
fi

mkdir -p "$OUTPUT_DIR" "$MERGED_DIR" "$OPENNMT_DIR" "$VOCAB_DIR" \
         "$MODEL_DIR" "$PREPARED_DIR"

# ── Print plan ────────────────────────────────────────────────────────────────
banner "Gendercheck  ·  OpenNMT-py token-based annotation pipeline"
echo "  Task          : seq2seq gender annotation (token vocab, no BPE)"
echo "  Output        : $OUTPUT_DIR"
if [[ "$SKIP_EXTRACT" == "false" ]]; then
    echo "  Corpora       : $CORPORA_DIR"
    echo "  Sources (${#SOURCES[@]})    :"
    for s in "${SOURCES[@]}"; do
        printf "    • %-30s  %6d files\n" "$(basename "$s")" \
               "$(find "$s" -type f | wc -l)"
    done
    echo "  Max positive  : $MAX_POSITIVE per source  →  ~$((MAX_POSITIVE * ${#SOURCES[@]})) merged"
fi
echo "  Neg ratio     : $NEG_RATIO  |  Seed: $SEED"
echo "  Vocab size    : src $SRC_VOCAB_SIZE  tgt $TGT_VOCAB_SIZE"
echo "  Train steps   : $TRAIN_STEPS  |  Valid: $VALID_STEPS  |  Save: $SAVE_STEPS"
echo "  GPU index     : $GPU  (use --gpu -1 for CPU)"
echo "  OpenNMT-py    : $ONMT_DIR"
echo "  Config        : $CONFIG_FILE"

# ──────────────────────────────────────────────────────────────────────────────
# STEPS 1 & 2  Extract per-source data and merge
# ──────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_EXTRACT" == "false" ]]; then

    banner "Step 1/4 — Extracting training data (equal quota per source)"

    SOURCE_DATA_DIRS=()
    for SOURCE_DIR in "${SOURCES[@]}"; do
        SOURCE_NAME="$(basename "$SOURCE_DIR")"
        SOURCE_DATA="$OUTPUT_DIR/data_${SOURCE_NAME}"
        SOURCE_DATA_DIRS+=("$SOURCE_DATA")

        echo
        echo "  ── $SOURCE_NAME ──"

        # Detect extensionless HTML (www.woz.ch stores articles without .html).
        # Avoid pipefail/SIGPIPE by collecting results into an array first.
        ALL_FILES_FLAG=""
        mapfile -t _SAMPLE < <(find "$SOURCE_DIR" -maxdepth 3 -type f \
                                ! -name '*.html' ! -name '*.htm' 2>/dev/null \
                                | head -5)
        SAMPLE_EXT_COUNT="${#_SAMPLE[@]}"
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

        for split in train valid test; do
            f="$SOURCE_DATA/${split}.jsonl"
            [[ -f "$f" ]] && printf "    %-8s %6d examples\n" "$split" "$(wc -l < "$f")"
        done
    done

    banner "Step 2/4 — Merging splits (equal weight, then shuffle)"

    python3 "$SCRIPT_DIR/_merge_splits.py" \
        --source-dirs "${SOURCE_DATA_DIRS[@]}" \
        --output-dir  "$MERGED_DIR" \
        --seed        "$SEED"

    echo
    echo "  Merged dataset:"
    for split in train valid test; do
        f="$MERGED_DIR/${split}.jsonl"
        [[ -f "$f" ]] && printf "    %-8s %6d examples\n" "$split" "$(wc -l < "$f")"
    done

else
    banner "Steps 1–2 skipped (--skip-extract)"
    echo "  Reusing: $MERGED_DIR"
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3  BIO JSONL → OpenNMT .src / .tgt
# ──────────────────────────────────────────────────────────────────────────────
if [[ "$SKIP_CONVERT" == "false" ]]; then

    banner "Step 3/4 — Converting BIO labels → OpenNMT src/tgt format"
    echo "  src : plain whitespace-tokenized sentence"
    echo "  tgt : sentence with <gender> … </gender> tokens around gender spans"
    echo

    python3 "$SCRIPT_DIR/convert_bio_to_opennmt.py" \
        "$MERGED_DIR" "$OPENNMT_DIR"

    echo
    echo "  Verifying src/tgt line alignment ..."
    for split in train valid; do
        SRC_N="$(wc -l < "$OPENNMT_DIR/${split}.src")"
        TGT_N="$(wc -l < "$OPENNMT_DIR/${split}.tgt")"
        if [[ "$SRC_N" -ne "$TGT_N" ]]; then
            echo "ERROR: line mismatch in $split: src=$SRC_N tgt=$TGT_N" >&2
            exit 1
        fi
        echo "    $split  OK  ($SRC_N lines)"
    done

else
    banner "Step 3 skipped (--skip-convert)"
    echo "  Reusing: $OPENNMT_DIR"
fi

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4  Build vocabulary + generate config
# ──────────────────────────────────────────────────────────────────────────────
banner "Step 4/4 — Building token-based vocabulary"

if [[ "$SKIP_VOCAB" == "false" ]]; then
    python3 "$SCRIPT_DIR/build_opennmt_vocab.py" \
        --src       "$OPENNMT_DIR/train.src" \
        --tgt       "$OPENNMT_DIR/train.tgt" \
        --src-vocab "$VOCAB_DIR/vocab.src" \
        --tgt-vocab "$VOCAB_DIR/vocab.tgt" \
        --src-size  "$SRC_VOCAB_SIZE" \
        --tgt-size  "$TGT_VOCAB_SIZE" \
        --tgt-extra "<gender> </gender>"
else
    echo "  Skipped (--skip-vocab) — reusing $VOCAB_DIR"
fi

echo
echo "  Vocabulary:"
for v in vocab.src vocab.tgt; do
    f="$VOCAB_DIR/$v"
    [[ -f "$f" ]] && printf "    %-16s  %6d tokens\n" "$v" "$(wc -l < "$f")"
done

# ── Write train_config.yaml ───────────────────────────────────────────────────
ABS_OPENNMT="$(realpath "$OPENNMT_DIR")"
ABS_VOCAB="$(realpath "$VOCAB_DIR")"
ABS_PREPARED="$(realpath "$PREPARED_DIR")"
ABS_MODEL="$(realpath "$MODEL_DIR")"
ABS_ONMT="$(realpath "$ONMT_DIR")"

if [[ "$GPU" -ge 0 ]]; then
    GPU_SECTION="world_size: 1
gpu_ranks: [$GPU]"
else
    GPU_SECTION="world_size: 1
# No GPU — remove gpu_ranks to train on CPU (add: gpu_ranks: [])"
fi

cat > "$CONFIG_FILE" <<YAML
# train_config.yaml — generated by run_pipeline_3.sh
# OpenNMT-py 3.x  ·  German gender annotation  ·  token-based vocabulary
#
# Task: seq2seq annotation
#   src: plain German sentence (whitespace-tokenized)
#   tgt: same sentence with <gender> and </gender> around gender spans
#
# No 'transforms' entry = default whitespace tokenizer.
# <gender> and </gender> are regular vocabulary items in vocab.tgt.

# ── Data ──────────────────────────────────────────────────────────────────────
save_data: ${ABS_PREPARED}/example

data:
  corpus_1:
    path_src: ${ABS_OPENNMT}/train.src
    path_tgt: ${ABS_OPENNMT}/train.tgt
  valid:
    path_src: ${ABS_OPENNMT}/valid.src
    path_tgt: ${ABS_OPENNMT}/valid.tgt

# ── Vocabulary ────────────────────────────────────────────────────────────────
src_vocab: ${ABS_VOCAB}/vocab.src
tgt_vocab: ${ABS_VOCAB}/vocab.tgt
overwrite: false

src_vocab_size: ${SRC_VOCAB_SIZE}
tgt_vocab_size: ${TGT_VOCAB_SIZE}

skip_empty_level: silent
src_seq_length: 200
tgt_seq_length: 220

# ── Model (Transformer) ───────────────────────────────────────────────────────
encoder_type: transformer
decoder_type: transformer
enc_layers: 6
dec_layers: 6
heads: 8
hidden_size: 512
word_vec_size: 512
transformer_ff: 2048
dropout: [0.1]
attention_dropout: [0.1]

# ── Optimization ──────────────────────────────────────────────────────────────
optim: adam
learning_rate: 2
warmup_steps: 8000
decay_method: noam
adam_beta2: 0.998
max_grad_norm: 0
label_smoothing: 0.1
param_init: 0
param_init_glorot: true
normalization: tokens

# ── Training ──────────────────────────────────────────────────────────────────
batch_type: tokens
batch_size: 4096
valid_batch_size: 64
train_steps: ${TRAIN_STEPS}
valid_steps: ${VALID_STEPS}
save_checkpoint_steps: ${SAVE_STEPS}
keep_checkpoint: 5
save_model: ${ABS_MODEL}/model

# ── Hardware ──────────────────────────────────────────────────────────────────
${GPU_SECTION}
YAML

echo
echo "  Config written → $CONFIG_FILE"

# ── Summary ───────────────────────────────────────────────────────────────────
banner "Dataset ready"

echo "  Data files:"
for split in train valid test; do
    for ext in src tgt; do
        f="$OPENNMT_DIR/${split}.${ext}"
        [[ -f "$f" ]] && printf "    %-22s  %7d lines\n" "${split}.${ext}" "$(wc -l < "$f")"
    done
done

echo
echo "  Vocabulary:"
for v in vocab.src vocab.tgt; do
    f="$VOCAB_DIR/$v"
    [[ -f "$f" ]] && printf "    %-20s  %6d tokens\n" "$v" "$(wc -l < "$f")"
done

echo
echo "  To train (requires OpenNMT-py deps — run from OpenNMT-py source dir):"
echo
echo "    cd ${ABS_ONMT}"
echo "    pip install -e ."
echo "    onmt_train -config ${CONFIG_FILE}"
echo
echo "  To translate after training:"
echo
echo "    onmt_translate \\"
echo "      -model ${ABS_MODEL}/model_step_${TRAIN_STEPS}.pt \\"
echo "      -src   INPUT.txt \\"
echo "      -output OUTPUT_ANNOTATED.txt \\"
echo "      -gpu   $GPU"
echo
echo "  Test set:"
echo "    src : $OPENNMT_DIR/test.src"
echo "    tgt : $OPENNMT_DIR/test.tgt  (reference)"
