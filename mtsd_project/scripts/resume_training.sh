#!/bin/bash
# =============================================================================
# resume_training.sh
# Resumes training from the latest checkpoint (last.pt).
# Usage:
#   bash resume_training.sh                          # auto-find latest run
#   bash resume_training.sh <path/to/last.pt>        # explicit checkpoint
#
# Use this when:
#   - Kaggle run was interrupted and you have a checkpoint to continue from
#   - CloudLab session expired mid-run
#   - You want to continue a run for more epochs
# =============================================================================

WORK_DIR="/mydata/mtsd_project"
SESSION_NAME="training"

echo "============================================================"
echo "RESUME TRAINING"
echo "============================================================"

# ── Find checkpoint ───────────────────────────────────────────────────────────
if [ -n "$1" ]; then
    CHECKPOINT="$1"
    echo "Using provided checkpoint: $CHECKPOINT"
else
    # Auto-find the most recent last.pt
    CHECKPOINT=$(find "$WORK_DIR/runs" -name "last.pt" -printf "%T@ %p\n" 2>/dev/null \
                 | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found in $WORK_DIR/runs"
        echo "Usage: bash resume_training.sh /path/to/last.pt"
        exit 1
    fi
    echo "Auto-found checkpoint: $CHECKPOINT"
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# ── Show what we're resuming from ────────────────────────────────────────────
RUN_DIR=$(dirname $(dirname "$CHECKPOINT"))
echo "Run directory: $RUN_DIR"

# Show last few lines of results if available
RESULTS_CSV="$RUN_DIR/results.csv"
if [ -f "$RESULTS_CSV" ]; then
    echo ""
    echo "Last 3 epochs from previous run:"
    tail -3 "$RESULTS_CSV"
fi

# ── Check if screen session already exists ────────────────────────────────────
if screen -list | grep -q "$SESSION_NAME"; then
    echo ""
    echo "WARNING: A screen session named '$SESSION_NAME' already exists."
    echo "  Reattach : screen -r $SESSION_NAME"
    echo "  Kill it  : screen -X -S $SESSION_NAME quit"
    exit 1
fi

# ── Write resume command ──────────────────────────────────────────────────────
cat > "$WORK_DIR/run_resume.sh" << EOF
#!/bin/bash
source "$WORK_DIR/activate.sh"
cd "$WORK_DIR"

python scripts/train.py \\
    --resume \\
    --model "$CHECKPOINT" \\
    --device 0

echo ""
echo "Resume training finished."
EOF

chmod +x "$WORK_DIR/run_resume.sh"

# ── Launch in screen ──────────────────────────────────────────────────────────
screen -dmS "$SESSION_NAME" bash "$WORK_DIR/run_resume.sh"

echo ""
echo "Resumed training in background screen session: $SESSION_NAME"
echo ""
echo "Useful commands:"
echo "  Watch live progress : screen -r $SESSION_NAME"
echo "  Detach from screen  : Ctrl+A then D"
echo "  Check GPU usage     : watch -n5 nvidia-smi"