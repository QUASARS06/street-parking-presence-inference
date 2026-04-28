#!/bin/bash
# =============================================================================
# setup_env.sh
# Run ONCE on a fresh CloudLab node to install all dependencies.
# Usage: bash setup_env.sh
# Safe to re-run — checks if things are already installed.
# =============================================================================

set -e  # exit on any error

echo "============================================================"
echo "ENVIRONMENT SETUP"
echo "============================================================"

# ── 1. Working directory ──────────────────────────────────────────────────────
WORK_DIR="/mydata/mtsd_project"
mkdir -p "$WORK_DIR"
echo "Working directory: $WORK_DIR"

# ── 2. System packages ────────────────────────────────────────────────────────
echo ""
echo "[1/6] Installing system packages..."
sudo apt-get update -y -q
sudo apt-get install -y -q \
    python3-pip \
    python3-venv \
    unzip \
    screen \
    htop \
    tree \
    curl \
    wget \
    git

# ── 3. Verify GPU ─────────────────────────────────────────────────────────────
echo ""
echo "[2/6] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "ERROR: nvidia-smi not found. GPU drivers may not be installed."
    echo "Try: sudo ubuntu-drivers autoinstall && sudo reboot"
    exit 1
fi

# ── 4. Python virtual environment ─────────────────────────────────────────────
echo ""
echo "[3/6] Setting up Python virtual environment..."
if [ ! -d "$WORK_DIR/venv" ]; then
    python3 -m venv "$WORK_DIR/venv"
    echo "Created new venv at $WORK_DIR/venv"
else
    echo "venv already exists, skipping creation"
fi

source "$WORK_DIR/venv/bin/activate"

# ── 5. Python packages ────────────────────────────────────────────────────────
echo ""
echo "[4/6] Installing Python packages..."
pip install --upgrade pip -q
pip install ultralytics -q

# Verify torch sees the GPU
python3 -c "
import torch
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU             : {torch.cuda.get_device_name(0)}')
    print(f'VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: CUDA not available — check drivers')
"

# ── 6. Directory structure ────────────────────────────────────────────────────
echo ""
echo "[5/6] Creating project directory structure..."
mkdir -p "$WORK_DIR/scripts"
mkdir -p "$WORK_DIR/outputs"
mkdir -p "$WORK_DIR/runs"

# ── 7. Write activation helper ────────────────────────────────────────────────
echo ""
echo "[6/6] Writing convenience scripts..."

cat > "$WORK_DIR/activate.sh" << 'EOF'
#!/bin/bash
# Source this to activate the project environment
# Usage: source /mydata/mtsd_project/activate.sh
export WORK_DIR="/mydata/mtsd_project"
source "$WORK_DIR/venv/bin/activate"
cd "$WORK_DIR"
echo "Environment activated. Working dir: $WORK_DIR"
EOF

chmod +x "$WORK_DIR/activate.sh"

echo ""
echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Upload your data:   bash /mydata/mtsd_project/scripts/upload_data.sh"
echo "  2. Start training:     bash /mydata/mtsd_project/scripts/start_training.sh"
echo "  3. Resume training:    bash /mydata/mtsd_project/scripts/resume_training.sh"
echo ""
echo "To activate environment manually:"
echo "  source /mydata/mtsd_project/activate.sh"