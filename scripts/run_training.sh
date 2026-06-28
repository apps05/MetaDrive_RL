#!/bin/bash
# ============================================================
# Traffic-Aware MetaDrive Agent — Training Launch Script
#
# Sets up headless display (Xvfb) and launches training with
# the legalgpu conda environment.
#
# Usage:
#   bash scripts/run_training.sh              # full training
#   bash scripts/run_training.sh --test 1000  # smoke test
#   bash scripts/run_training.sh --resume checkpoints/sac_agent_step0500000
# ============================================================


set -e

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_BASE="/home/user29/miniforge3"
ENV_NAME="legalgpu"
PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
LOG_FILE="${PROJECT_DIR}/logs/training.log"
TENSORBOARD_LOG="${PROJECT_DIR}/logs"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/videos" "${PROJECT_DIR}/checkpoints"

echo "============================================"
echo "  Traffic-Aware MetaDrive SAC — Training"
echo "  Project : ${PROJECT_DIR}"
echo "  Python  : ${PYTHON}"
echo "  Log     : ${LOG_FILE}"
echo "============================================"

# ── Xvfb headless display ──────────────────────────────────────────────────
# Needed for MetaDrive's Panda3D / OpenGL context even in headless mode
if ! pgrep -x "Xvfb" > /dev/null 2>&1; then
    echo "[Setup] Starting Xvfb virtual display on :99 ..."
    Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
    XVFB_PID=$!
    echo "[Setup] Xvfb PID: ${XVFB_PID}"
    sleep 2
else
    echo "[Setup] Xvfb already running."
fi

export DISPLAY=:99
export SDL_VIDEODRIVER=offscreen
export MESA_GL_VERSION_OVERRIDE=3.3
export LIBGL_ALWAYS_SOFTWARE=1  # fallback if no GPU driver
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# ── CUDA check ──────────────────────────────────────────────────────────────
echo ""
echo "[Setup] CUDA status:"
${PYTHON} -c "
import torch
print(f'  PyTorch : {torch.__version__}')
print(f'  CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU     : {torch.cuda.get_device_name(0)}')
" 2>/dev/null || echo "  [WARN] Could not check CUDA status"

echo ""
echo "[Setup] Starting training (output → ${LOG_FILE})"
echo "  Monitor: tail -f ${LOG_FILE}"
echo "  TBoard : tensorboard --logdir ${TENSORBOARD_LOG}"
echo ""

# ── Launch training ─────────────────────────────────────────────────────────
# Pass all arguments through to train.py
echo "[Launch] Running training in foreground per user request..."
${PYTHON} -u "${PROJECT_DIR}/training/train.py" "$@" | tee -a "${LOG_FILE}"

