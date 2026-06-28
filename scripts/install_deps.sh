#!/bin/bash
# ============================================================
# Traffic-Aware MetaDrive Agent — Dependency Installer
# Target: legalgpu conda env (Python 3.11, torch 2.3.1+cu121)
# ============================================================


set -e

CONDA_BASE="/home/user29/miniforge3"
ENV_NAME="legalgpu"
PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

echo "============================================"
echo " Installing Traffic-Aware Agent Dependencies"
echo " Conda env: ${ENV_NAME}"
echo "============================================"

# ---- Core RL & Simulation ----
echo "[1/5] Installing metadrive-simulator..."
$PIP install metadrive-simulator --quiet

# ---- Stable-Baselines3 (SAC) ----
echo "[2/5] Installing stable-baselines3..."
$PIP install stable-baselines3 --quiet

# ---- Gymnasium (gym API) ----
echo "[3/5] Installing gymnasium..."
$PIP install gymnasium --quiet

# ---- Video recording (headless) ----
echo "[4/5] Installing opencv-python-headless + imageio..."
$PIP install opencv-python-headless imageio imageio-ffmpeg --quiet

# ---- Tensorboard for monitoring ----
echo "[5/5] Installing tensorboard..."
$PIP install tensorboard --quiet

# ---- Optional: ffmpeg system package check ----
if ! command -v ffmpeg &> /dev/null; then
    echo "[INFO] ffmpeg not found in PATH. Video writing will use imageio-ffmpeg instead."
else
    echo "[OK] ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
fi

# ---- CUDA check ----
echo ""
echo "---- CUDA Availability Check ----"
$PYTHON -c "
import torch
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU             : {torch.cuda.get_device_name(0)}')
    print(f'VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('[WARN] CUDA unavailable — will train on CPU (driver may not be loaded)')
"

echo ""
echo "============================================"
echo " All dependencies installed successfully!"
echo " Workspace: /home/user29/RL/"
echo " Run:  bash scripts/run_training.sh"
echo "============================================"
