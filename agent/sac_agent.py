"""
SAC Agent Wrapper

Wraps stable_baselines3.SAC with project-specific configuration.
Handles device detection, replay buffer sizing, checkpoint I/O.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import SAC_CONFIG, CHECKPOINT_DIR, LOG_DIR


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'. Prints diagnostic."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[SAC] GPU detected: {name} ({vram:.1f} GB VRAM) — using CUDA")
        return "cuda"
    else:
        print("[SAC] CUDA unavailable — using CPU (check nvidia driver)")
        return "cpu"


def build_sac(env, tensorboard_log: str | None = None, seed: int = 42):
    """
    Instantiate a stable-baselines3 SAC agent.

    Parameters
    ──────────
    env             : Gymnasium-compatible environment (or VecEnv)
    tensorboard_log : Directory for TensorBoard logs
    seed            : Random seed

    Returns
    ───────
    model : stable_baselines3.SAC
    """
    from stable_baselines3 import SAC

    device = get_device()
    cfg = {**SAC_CONFIG, "device": device}

    model = SAC(
        policy=cfg["policy"],
        env=env,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        ent_coef=cfg["ent_coef"],
        target_update_interval=cfg["target_update_interval"],
        gradient_steps=cfg["gradient_steps"],
        learning_starts=cfg["learning_starts"],
        train_freq=cfg["train_freq"],
        policy_kwargs=cfg["policy_kwargs"],
        verbose=cfg["verbose"],
        device=device,
        tensorboard_log=tensorboard_log or LOG_DIR,
        seed=seed,
    )

    param_count = sum(p.numel() for p in model.policy.parameters())
    print(f"[SAC] Policy parameters: {param_count:,}")
    print(f"[SAC] Replay buffer capacity: {cfg['buffer_size']:,} transitions")
    print(f"[SAC] Observation shape: {env.observation_space.shape}")
    print(f"[SAC] Action shape: {env.action_space.shape}")

    return model


def save_checkpoint(model, step: int, suffix: str = ""):
    """Save model checkpoint."""
    name = f"sac_agent_step{step:07d}{suffix}"
    path = os.path.join(CHECKPOINT_DIR, name)
    model.save(path)
    print(f"[SAC] Checkpoint saved: {path}.zip")
    return path


def load_checkpoint(env, path: str):
    """Load SAC model from checkpoint."""
    from stable_baselines3 import SAC
    model = SAC.load(path, env=env, device=get_device())
    print(f"[SAC] Loaded checkpoint: {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import gymnasium as gym

    # Use a simple Gym env for the smoke test (no MetaDrive needed)
    dummy_env = gym.make("Pendulum-v1")
    model = build_sac(dummy_env, seed=42)
    print("[SAC] Smoke test: learning_starts =", model.learning_starts)
    print("[SAC] Smoke test: device =", model.device)
    dummy_env.close()
    print("[SAC] Smoke test PASSED ✓")
