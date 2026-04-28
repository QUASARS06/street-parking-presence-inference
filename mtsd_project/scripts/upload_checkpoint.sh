#!/bin/bash
# =============================================================================
# upload_checkpoint.sh
# Downloads a checkpoint from Kaggle output and sets it up for resuming.
#
# WORKFLOW:
#   1. On Kaggle: go to Output tab, download the runs/ folder or specific .pt
#   2. On your Mac: scp the file to CloudLab
#   3. On CloudLab: bash upload_checkpoint.sh
#
# SCP command to run on your MAC (not on CloudLab):
#   scp -i ~/.ssh/id_rsa epoch25.pt USER@NODE.cloudlab.us:/mydata/checkpoints/
#
# Then on CloudLab:
#   bash scripts/upload_checkpoint.sh /mydata/checkpoints/epoch25.pt
# =============================================================================

WORK_DIR="/mydata/mtsd_project"
CHECKPOINT_DIR="/mydata/checkpoints"

mkdir -p "$CHECKPOINT_DIR"

echo "============================================================"
echo "CHECKPOINT UPLOAD HELPER"
echo "============================================================"
echo ""

if [ -z "$1" ]; then
    echo "Usage: bash upload_checkpoint.sh <checkpoint.pt>"
    echo ""
    echo "Steps to get checkpoint from Kaggle:"
    echo ""
    echo "  1. In Kaggle notebook Output tab, find your runs folder"
    echo "  2. Download the .pt file you want (best.pt or epochN.pt)"
    echo "  3. On your MAC terminal, run:"
    echo "     scp -i ~/.ssh/id_rsa /path/to/checkpoint.pt \\"
    echo "         USER@NODE.cloudlab.us:/mydata/checkpoints/"
    echo ""
    echo "  4. Then back on CloudLab:"
    echo "     bash scripts/upload_checkpoint.sh /mydata/checkpoints/checkpoint.pt"
    echo ""
    echo "Available checkpoints in $CHECKPOINT_DIR:"
    ls -lh "$CHECKPOINT_DIR"/*.pt 2>/dev/null || echo "  (none yet)"
    exit 0
fi

CHECKPOINT="$1"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: File not found: $CHECKPOINT"
    exit 1
fi

# ── Validate it's a real PyTorch file ─────────────────────────────────────────
echo "Validating checkpoint..."
source "$WORK_DIR/activate.sh"
python3 -c "
import torch, sys
try:
    ckpt = torch.load('$CHECKPOINT', map_location='cpu')
    epoch = ckpt.get('epoch', 'unknown')
    print(f'Checkpoint valid. Epoch: {epoch}')
except Exception as e:
    print(f'ERROR: Invalid checkpoint: {e}')
    sys.exit(1)
"

echo ""
echo "Checkpoint ready. To resume training run:"
echo "  bash $WORK_DIR/scripts/resume_training.sh $CHECKPOINT"