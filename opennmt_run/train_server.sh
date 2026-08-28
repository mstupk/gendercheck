#!/bin/bash
# =============================================================================
# train_server.sh  —  Train gendercheck on a dedicated server (RTX 4090 / 24 GB)
#
# Uses train_config.yaml (fp16, batch_size 4096, accum 4).
# For an A40 (40 GB), change CONFIG to train_config_a40.yaml below.
#
# Run from anywhere — the script locates itself:
#   bash opennmt_run/train_server.sh
#
# Prerequisites:
#   - conda environment devenv_2 with OpenNMT-py v3 and sacrebleu
#     (if not installed: pip install -e ~/testruns/OpenNMT-py/)
#   - Corpus files already generated in opennmt_run/
#     (run: cd claude_scripts && ./run_pipeline_3.sh)
# =============================================================================
set -eo pipefail

CONDA_ENV="sala_1"
CONFIG="opennmt_run/train_config.yaml"
CORPUS_DIR="opennmt_run"
VOCAB_DIR="opennmt_run"
MODEL_DIR="opennmt_run/model"
MODEL_PREFIX="gendercheck"
LOG_DIR="$MODEL_DIR"

# ── Locate ~/  ───────────────────────────────────────────────────────────────
# Script lives at  ~/opennmt_run/train_server.sh
# SCRIPT_DIR  →   ~/opennmt_run/
# REPO_ROOT   →   ~/  — same base as SLURM's cd ~/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "Working directory: $REPO_ROOT"
echo "Config           : $CONFIG"
echo "Conda env        : $CONDA_ENV"
echo ""

# ── Activate conda ────────────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

pip install -q "numpy<2" sacrebleu

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p "$MODEL_DIR"
mkdir -p "opennmt_run/prepared"

# ── Check corpus files ────────────────────────────────────────────────────────
echo "Checking corpus files..."
MISSING=0
for f in "$CORPUS_DIR/train.src" "$CORPUS_DIR/train.tgt" \
          "$CORPUS_DIR/valid.src" "$CORPUS_DIR/valid.tgt" \
          "$CORPUS_DIR/test.src"  "$CORPUS_DIR/test.tgt"; do
    if [[ ! -f "$f" ]]; then
        echo "  MISSING: $f"
        MISSING=1
    else
        echo "  OK: $f  ($(wc -l < "$f") lines)"
    fi
done
if [[ $MISSING -eq 1 ]]; then
    echo ""
    echo "ERROR: corpus files missing.  Run first:"
    echo "  cd ~/claude_scripts && ./run_pipeline_3.sh"
    exit 1
fi
echo ""

# ── Check vocabulary ──────────────────────────────────────────────────────────
# Vocabulary is built by build_opennmt_vocab.py (onmt_build_vocab unavailable —
# pyonmttok is not installable in this environment).
if [[ ! -f "$VOCAB_DIR/vocab.src" ]]; then
    echo "ERROR: vocabulary missing.  Rebuild with:"
    echo "  cd ~/claude_scripts && python3 build_opennmt_vocab.py"
    exit 1
else
    echo "Vocabulary present  ($(wc -l < "$VOCAB_DIR/vocab.src") src tokens," \
         "$(wc -l < "$VOCAB_DIR/vocab.tgt") tgt tokens)"
fi
echo ""

# ── Train ─────────────────────────────────────────────────────────────────────
echo "Starting training  (log → $LOG_DIR/train.log) ..."
onmt_train -config "$CONFIG" 2>&1 | tee "$LOG_DIR/train.log"
echo ""
echo "Training complete."
echo ""

# ── Translate test set with best (most recent) checkpoint ────────────────────
BEST_MODEL=$(ls -t "$MODEL_DIR/${MODEL_PREFIX}_step_"*.pt 2>/dev/null | head -1)
if [[ -z "$BEST_MODEL" ]]; then
    echo "ERROR: no checkpoint found in $MODEL_DIR"
    exit 1
fi
echo "Translating with $BEST_MODEL ..."
onmt_translate \
    -model  "$BEST_MODEL" \
    -src    "$CORPUS_DIR/test.src" \
    -output "$CORPUS_DIR/test.pred" \
    -gpu 0 \
    --beam_size 5 \
    --batch_size 4096 \
    --batch_type tokens \
    2>&1 | tee "$LOG_DIR/translate.log"
echo ""

# ── Score: annotated prediction vs annotated reference ───────────────────────
echo "Scoring..."
sacrebleu "$CORPUS_DIR/test.tgt" \
    -i "$CORPUS_DIR/test.pred" -m bleu chrf ter \
    2>&1 | tee "$LOG_DIR/scores.txt"

# ── Faithfulness check: strip <gender> tags, compare to plain source ─────────
python3 - <<'PYEOF'
import re, pathlib, os
corpus = pathlib.Path(os.environ.get("CORPUS_DIR", "opennmt_run"))
pred   = (corpus / "test.pred").read_text(encoding="utf-8").splitlines()
plain  = [re.sub(r"</?gender>\s*", "", l).strip() for l in pred]
(corpus / "test.pred.plain").write_text("\n".join(plain) + "\n", encoding="utf-8")
print(f"Tag-stripped predictions  →  {corpus}/test.pred.plain")
PYEOF

sacrebleu "$CORPUS_DIR/test.src" \
    -i "$CORPUS_DIR/test.pred.plain" -m bleu chrf \
    2>&1 | tee -a "$LOG_DIR/scores.txt"

echo ""
echo "Done.  Results saved to $LOG_DIR/scores.txt"

conda deactivate
