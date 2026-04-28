#!/bin/bash
# =============================================================================
# status.sh
# Shows current training status, GPU usage, and latest metrics at a glance.
# Usage: bash status.sh
# =============================================================================

WORK_DIR="/mydata/mtsd_project"

echo "============================================================"
echo "TRAINING STATUS"
echo "============================================================"

# ── GPU ───────────────────────────────────────────────────────────────────────
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits \
    | awk -F',' '{printf "  %-20s | GPU: %s%% | VRAM: %s/%s MB | Temp: %s°C\n", $1, $2, $3, $4, $5}'

# ── Screen sessions ───────────────────────────────────────────────────────────
echo ""
echo "Screen sessions:"
screen -list 2>/dev/null | grep -v "No Sockets" || echo "  No active screen sessions"

# ── Latest run ────────────────────────────────────────────────────────────────
echo ""
echo "Latest run:"
LATEST_RUN=$(ls -td "$WORK_DIR/runs/detect/"*/ 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "  No runs found yet."
else
    echo "  Run dir: $LATEST_RUN"

    # Latest weights
    echo ""
    echo "  Checkpoints:"
    ls -lh "$LATEST_RUN/weights/"*.pt 2>/dev/null \
        | awk '{printf "    %-20s %s\n", $9, $5}' \
        | sed "s|$LATEST_RUN/weights/||g" \
        || echo "    (none yet)"

    # Latest metrics from results.csv
    RESULTS="$LATEST_RUN/results.csv"
    if [ -f "$RESULTS" ]; then
        echo ""
        echo "  Latest epoch metrics:"
        # Print header and last row
        HEAD=$(head -1 "$RESULTS")
        LAST=$(tail -1 "$RESULTS")
        # Extract key columns: epoch, box_loss, cls_loss, mAP50, mAP50-95
        python3 - << PYEOF
import csv, io

header = """$HEAD""".strip().split(',')
last   = """$LAST""".strip().split(',')

# Map column names to values
row = dict(zip(header, last))

keys = [
    ('                  epoch', 'epoch'),
    ('         train/box_loss', 'train/box_loss'),
    ('         train/cls_loss', 'train/cls_loss'),
    ('   metrics/precision(B)', 'metrics/precision(B)'),
    ('      metrics/recall(B)', 'metrics/recall(B)'),
    ('        metrics/mAP50(B)', 'metrics/mAP50(B)'),
    ('     metrics/mAP50-95(B)', 'metrics/mAP50-95(B)'),
]

for display, key in keys:
    val = row.get(key.strip(), row.get(key, 'N/A')).strip()
    print(f"    {key.strip():<25} : {val}")
PYEOF
    else
        echo "    (training not started yet)"
    fi
fi

# ── Disk space ────────────────────────────────────────────────────────────────
echo ""
echo "Disk space:"
df -h /mydata | awk 'NR==2 {printf "  Used: %s / %s (%s free)\n", $3, $2, $4}'

echo ""
echo "============================================================"