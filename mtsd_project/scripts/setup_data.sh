#!/bin/bash
# =============================================================================
# setup_data.sh
# Unzips uploaded data and scripts, writes the correct YAML for this machine.
# Usage: bash setup_data.sh
#
# Expected uploads in /mydata/:
#   data.zip     — your prepared YOLO dataset (data/images, data/labels)
#   scripts.zip  — your scripts folder + parking_signs.yaml
#
# Safe to re-run — skips unzip if already done.
# =============================================================================

set -e

WORK_DIR="/mydata/mtsd_project"
UPLOAD_DIR="/mydata"

echo "============================================================"
echo "DATA SETUP"
echo "============================================================"

source "$WORK_DIR/activate.sh"

# ── Unzip scripts ─────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Setting up scripts..."
if [ ! -f "$WORK_DIR/scripts/train.py" ]; then
    if [ -f "$UPLOAD_DIR/scripts.zip" ]; then
        unzip -q "$UPLOAD_DIR/scripts.zip" -d "$WORK_DIR/"
        echo "Scripts unzipped."
    else
        echo "ERROR: scripts.zip not found at $UPLOAD_DIR/scripts.zip"
        echo "Upload it first: scp scripts.zip user@node:/mydata/"
        exit 1
    fi
else
    echo "Scripts already present, skipping."
fi

# ── Unzip data ────────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Setting up dataset..."
if [ ! -d "$WORK_DIR/data/images/train" ]; then
    if [ -f "$UPLOAD_DIR/data.zip" ]; then
        echo "Unzipping data (this may take a few minutes)..."
        unzip -q "$UPLOAD_DIR/data.zip" -d "$WORK_DIR/"
        echo "Data unzipped."
    else
        echo "ERROR: data.zip not found at $UPLOAD_DIR/data.zip"
        echo "Upload it first: scp data.zip user@node:/mydata/"
        exit 1
    fi
else
    echo "Data already present, skipping unzip."
fi

# ── Verify data structure ─────────────────────────────────────────────────────
echo ""
echo "Dataset structure:"
echo "  train images : $(ls $WORK_DIR/data/images/train | wc -l)"
echo "  val images   : $(ls $WORK_DIR/data/images/val | wc -l)"
echo "  train labels : $(ls $WORK_DIR/data/labels/train | wc -l)"
echo "  val labels   : $(ls $WORK_DIR/data/labels/val | wc -l)"

# ── Write YAML with correct paths for this machine ───────────────────────────
echo ""
echo "[3/3] Writing parking_signs.yaml..."
cat > "$WORK_DIR/parking_signs.yaml" << EOF
path: $WORK_DIR/data
train: images/train
val:   images/val

nc: 1
names:
  0: parking_sign
EOF

echo "YAML written:"
cat "$WORK_DIR/parking_signs.yaml"

echo ""
echo "============================================================"
echo "DATA SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Next: bash /mydata/mtsd_project/scripts/start_training.sh"