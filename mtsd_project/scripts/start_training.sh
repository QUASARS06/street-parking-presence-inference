#!/bin/bash
# =============================================================================
# start_training.sh
# Starts a fresh YOLOv8m training run inside a screen session.
# Usage: bash start_training.sh
#
# The training runs inside 'screen' so it survives SSH disconnection.
# Reattach anytime with: screen -r training
# =============================================================================

WORK_DIR="/mydata/mtsd_project"
SESSION_NAME="training"

echo "============================================================"
echo "STARTING FRESH TRAINING RUN"
echo "============================================================"

# ── Check if screen session already exists ────────────────────────────────────
if screen -list | grep -q "$SESSION_NAME"; then
    echo "WARNING: A screen session named '$SESSION_NAME' already exists."
    echo "Options:"
    echo "  Reattach to it : screen -r $SESSION_NAME"
    echo "  Kill it first  : screen -X -S $SESSION_NAME quit"
    echo "  Then re-run    : bash start_training.sh"
    exit 1
fi

# ── Write the training command to a temp script ───────────────────────────────
cat > "$WORK_DIR/run_train.sh" << EOF
#!/bin/bash
source "$WORK_DIR/activate.sh"
cd "$WORK_DIR"

python scripts/train.py \\
    --model yolov8m.pt \\
    --epochs 50 \\
    --batch 48 \\
    --imgsz 640 \\
    --cls 2.0 \\
    --device 0

echo ""
echo "Training finished. Check runs/ for results."
EOF

chmod +x "$WORK_DIR/run_train.sh"

# ── Launch in screen ──────────────────────────────────────────────────────────
screen -dmS "$SESSION_NAME" bash "$WORK_DIR/run_train.sh"

echo ""
echo "Training started in background screen session: $SESSION_NAME"
echo ""
echo "Useful commands:"
echo "  Watch live progress : screen -r $SESSION_NAME"
echo "  Detach from screen  : Ctrl+A then D"
echo "  Check GPU usage     : watch -n5 nvidia-smi"
echo "  Check latest epoch  : tail -f \$(ls -t $WORK_DIR/runs/detect/*/results.csv | head -1)"
echo ""
echo "Checkpoints saved every 5 epochs to:"
echo "  $WORK_DIR/runs/detect/<run_name>/weights/"