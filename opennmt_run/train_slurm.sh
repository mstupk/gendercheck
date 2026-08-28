#!/bin/bash
#SBATCH --mail-user=michael.stupka@unibe.ch
#SBATCH --mail-type=fail,end
#SBATCH --job-name="gendercheck"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=23:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx4090:1

# Submit from ~/opennmt_run/ on the cluster:
#   sbatch opennmt_run/train_slurm.sh
#
# Transfer data first (from local project root):
#   rsync -avz ./ <user>@submit03.unibe.ch:~/opennmt_run/

# Put your code below this line
module --ignore-cache load CUDA
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate devenv_2

# Fix NumPy 2.x / OpenNMT-py ABI incompatibility; install sacrebleu
pip install -q "numpy<2" sacrebleu

# Create output directories
mkdir -p ~/opennmt_run/model
mkdir -p ~/opennmt_run/prepared

cd ~/ || { echo "ERROR: working directory not found"; exit 1; }

# Verify corpus files are present
echo "Checking corpus files..."
MISSING=0
for f in opennmt_run/train.src opennmt_run/train.tgt \
          opennmt_run/valid.src opennmt_run/valid.tgt \
          opennmt_run/test.src  opennmt_run/test.tgt; do
    if [[ ! -f "$f" ]]; then
        echo "  MISSING: $f"
        MISSING=1
    else
        echo "  OK: $f ($(wc -l < "$f") lines)"
    fi
done
if [[ $MISSING -eq 1 ]]; then
    echo ""
    echo "ERROR: corpus files missing.  Transfer them from the preparation machine:"
    echo "  rsync -avz ./ <user>@submit03.unibe.ch:~/opennmt_run/"
    exit 1
fi

# Vocabulary was built by build_opennmt_vocab.py (onmt_build_vocab unavailable —
# pyonmttok is not installable in this environment).
if [[ ! -f opennmt_run/vocab.src ]]; then
    echo "ERROR: vocabulary missing.  Rebuild on the preparation machine:"
    echo "  cd claude_scripts && python3 build_opennmt_vocab.py"
    exit 1
else
    echo "Vocabulary present ($(wc -l < opennmt_run/vocab.src) src tokens," \
         "$(wc -l < opennmt_run/vocab.tgt) tgt tokens)"
fi

# Train
echo "Starting training..."
onmt_train -config opennmt_run/train_config.yaml 2>&1 | tee opennmt_run/model/train.log
TRAIN_EXIT=${PIPESTATUS[0]}
if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "Training FAILED (exit $TRAIN_EXIT)"
    mail -s "gendercheck job ${SLURM_JOB_ID} FAILED" michael.s01@gmx.net \
        < opennmt_run/model/train.log
    exit $TRAIN_EXIT
fi

# Translate test set with best checkpoint
BEST_MODEL=$(ls -t opennmt_run/model/gendercheck_step_*.pt 2>/dev/null | head -1)
if [[ -z "$BEST_MODEL" ]]; then
    echo "ERROR: no checkpoint found after training"
    exit 1
fi
echo "Translating with $BEST_MODEL ..."
onmt_translate \
    -model  "$BEST_MODEL" \
    -src    opennmt_run/test.src \
    -output opennmt_run/test.pred \
    -gpu 0 \
    --beam_size 5 \
    --batch_size 4096 \
    --batch_type tokens \
    2>&1 | tee opennmt_run/model/translate.log

# Score: annotated prediction vs annotated reference
sacrebleu opennmt_run/test.tgt \
    -i opennmt_run/test.pred -m bleu chrf ter \
    2>&1 | tee opennmt_run/model/scores.txt

# Faithfulness check: strip <gender> tags, compare prediction to plain source
python3 -c "
import re, pathlib
pred = pathlib.Path('opennmt_run/test.pred').read_text(encoding='utf-8').splitlines()
plain = [re.sub(r'</?gender>\s*', '', l).strip() for l in pred]
pathlib.Path('opennmt_run/test.pred.plain').write_text('\n'.join(plain)+'\n', encoding='utf-8')
print(f'Tag-stripped predictions → opennmt_run/test.pred.plain')
"
sacrebleu opennmt_run/test.src \
    -i opennmt_run/test.pred.plain -m bleu chrf \
    2>&1 | tee -a opennmt_run/model/scores.txt

mail -s "gendercheck job ${SLURM_JOB_ID} done" michael.s01@gmx.net \
    < opennmt_run/model/scores.txt

conda deactivate
