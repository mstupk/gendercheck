#!/bin/bash
# Transfer opennmt_run/ to the cluster/server.
# Run from the project root:  /home/prokhor/sysback/work/gendercheck/
#
# Usage:
#   bash opennmt_run/transfer_to_cluster.sh <user>@submit03.unibe.ch

set -euo pipefail

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
    echo "Usage: $0 <user>@<host>"
    echo "Example: $0 ms01@submit03.unibe.ch"
    exit 1
fi

echo "Transferring opennmt_run/ to ${TARGET}:~/opennmt_run/ ..."

rsync -avz --progress \
    opennmt_run/ "${TARGET}:~/opennmt_run/"

echo ""
echo "Done.  To submit the training job:"
echo "  ssh ${TARGET}"
echo "  sbatch ~/opennmt_run/train_slurm.sh"
echo ""
echo "Or to train directly on the server:"
echo "  ssh ${TARGET}"
echo "  bash ~/opennmt_run/train_server.sh"
