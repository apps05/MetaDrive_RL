"""
"""Main Training Script — Traffic-Aware MetaDrive SAC Agent

Features
────────
• SubprocVecEnv with N parallel environments
• SAC (stable-baselines3) with GPU acceleration
• Curriculum scheduler (traffic density ramp 0%→45%)
• Checkpoint saves every 100k steps
• Periodic video capture every 30 minutes (wall-clock)
• TensorBoard logging
• Clean keyboard-interrupt handling

Usage
─────
    python training/train.py              # full 4M step run
    python training/train.py --test 1000  # 1k-step smoke test
    python training/train.py --resume checkpoints/sac_agent_step0500000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import signal

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from configs.config import TRAIN_CONFIG, ENV_CONFIG, VIDEO_DIR, CHECKPOINT_DIR, LOG_DIR
from agent.sac_agent import build_sac, save_checkpoint, load_checkpoint
from training.curriculum import CurriculumScheduler
from video.recorder import PeriodicVideoRecorder


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env(rank: int = 0, seed: int = 42, traffic_density: float = 0.0):
    """Factory function for SubprocVecEnv."""
    def _init():
        import os as _os
        _os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
        _os.environ.setdefault("DISPLAY", ":99")

        from env.metadrive_env import MetaDriveAgentEnv
        env = MetaDriveAgentEnv(traffic_density=traffic_density)
        env.reset(seed=seed + rank)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Callback: checkpoints + curriculum + video step tracking
# ─────────────────────────────────────────────────────────────────────────────

class TrainingCallback:
    """
    Lightweight callback called from the manual training loop.
    (Not a SB3 BaseCallback — we control the loop directly for curriculum.)
    """

    def __init__(
        self,
        model,
        curriculum: CurriculumScheduler,
        video_recorder: PeriodicVideoRecorder,
        checkpoint_every: int,
        model_ref: dict,
    ):
        self.model = model
        self.curriculum = curriculum
        self.video_recorder = video_recorder
        self.checkpoint_every = checkpoint_every
        self.model_ref = model_ref
        self._last_checkpoint = 0
        self._start_time = time.time()

    def on_step(self, step: int):
        # Update model reference for video recorder
        self.model_ref["model"] = self.model

        # Update video recorder step counter
        self.video_recorder.update_step(step)

        # Curriculum update
        self.curriculum.update(step)

        # Checkpoint
        if step - self._last_checkpoint >= self.checkpoint_every:
            save_checkpoint(self.model, step)
            self._last_checkpoint = step

        # Progress log every 10k steps
        if step % 10_000 == 0 and step > 0:
            elapsed = time.time() - self._start_time
            fps = step / max(elapsed, 1)
            eta_h = (TRAIN_CONFIG["total_timesteps"] - step) / max(fps * 3600, 1)
            print(
                f"[Train] Step {step:>8,} | "
                f"Elapsed: {elapsed/3600:.2f}h | "
                f"FPS: {fps:.0f} | "
                f"ETA: {eta_h:.2f}h | "
                f"Phase: {self.curriculum.get_current_phase()} | "
                f"Traffic: {self.curriculum.get_current_density():.2f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    total_timesteps: int | None = None,
    n_envs: int | None = None,
    resume_path: str | None = None,
    seed: int = 42,
):
    cfg = TRAIN_CONFIG
    total_timesteps = total_timesteps or cfg["total_timesteps"]
    n_envs = n_envs or cfg["n_envs"]

    print("=" * 60)
    print("  MetaDrive Traffic-Aware Agent — SAC Training")
    print("=" * 60)
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Parallel envs   : {n_envs}")
    print(f"  Checkpoint dir  : {CHECKPOINT_DIR}")
    print(f"  Video dir       : {VIDEO_DIR}")
    print(f"  Log dir         : {LOG_DIR}")
    print(f"  Seed            : {seed}")
    print("=" * 60)

    # ── Headless display setup ────────────────────────────────────────────────
    os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
    os.environ.setdefault("DISPLAY", ":99")

    # ── Build parallel environments ───────────────────────────────────────────
    print(f"\n[Train] Creating {n_envs} parallel environments ...")
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    env_fns = [make_env(rank=i, seed=seed, traffic_density=0.0) for i in range(n_envs)]

    try:
        vec_env = SubprocVecEnv(env_fns, start_method="fork")
    except Exception as e:
        print(f"[Train] SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecMonitor(vec_env, filename=os.path.join(LOG_DIR, "monitor"))
    print(f"[Train] Obs shape: {vec_env.observation_space.shape}")
    print(f"[Train] Action shape: {vec_env.action_space.shape}")

    # ── Build / load SAC model ────────────────────────────────────────────────
    if resume_path:
        print(f"\n[Train] Resuming from checkpoint: {resume_path}")
        model = load_checkpoint(vec_env, resume_path)
    else:
        print("\n[Train] Building SAC model from scratch ...")
        model = build_sac(vec_env, tensorboard_log=LOG_DIR, seed=seed)

    # ── Shared mutable references (for thread-safe recorder access) ───────────
    model_ref = {"model": model}
    curriculum_ref = {"density": 0.0}

    # ── Curriculum scheduler ──────────────────────────────────────────────────
    curriculum = CurriculumScheduler(vec_env)

    # ── Periodic video recorder ───────────────────────────────────────────────
    video_recorder = PeriodicVideoRecorder(
        model_ref=model_ref,
        curriculum_ref=curriculum_ref,
        interval_minutes=cfg["video_interval_minutes"],
        max_steps=cfg["video_episode_steps"],
        fps=15,
    )

    # ── Callback ──────────────────────────────────────────────────────────────
    callback = TrainingCallback(
        model=model,
        curriculum=curriculum,
        video_recorder=video_recorder,
        checkpoint_every=cfg["checkpoint_every"],
        model_ref=model_ref,
    )

    # ── Signal handler for graceful shutdown ──────────────────────────────────
    _shutdown = {"flag": False}

    def _sigint_handler(sig, frame):
        print("\n[Train] Ctrl+C received — saving checkpoint and exiting cleanly ...")
        _shutdown["flag"] = True

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    # ── Start periodic video recording ────────────────────────────────────────
    # video_recorder.start() # Disabled per user request

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[Train] Starting training loop ...\n")
    start_time = time.time()

    try:
        # We use SB3's built-in learn() but with manual step-level callbacks
        # via a lightweight SB3 callback wrapper
        from stable_baselines3.common.callbacks import BaseCallback

        class _LoopCallback(BaseCallback):
            def __init__(self, cb: TrainingCallback):
                super().__init__(verbose=0)
                self._cb = cb
                self._shutdown = _shutdown

            def _on_step(self) -> bool:
                self._cb.on_step(self.num_timesteps)
                # Update curriculum ref for video recorder
                curriculum_ref["density"] = curriculum.get_current_density()
                if self._shutdown["flag"]:
                    return False  # stop training gracefully
                return True

        model.learn(
            total_timesteps=total_timesteps,
            callback=_LoopCallback(callback),
            reset_num_timesteps=(resume_path is None),
            tb_log_name="sac_agent",
            progress_bar=True,
        )

    except KeyboardInterrupt:
        print("\n[Train] KeyboardInterrupt — saving final checkpoint ...")

    finally:
        # ── Final checkpoint ──────────────────────────────────────────────────
        final_step = model.num_timesteps
        save_checkpoint(model, final_step, suffix="_final")

        # ── Final video ───────────────────────────────────────────────────────
        video_recorder.stop()
        print("[Train] Recording final episode video ...")
        from video.recorder import EpisodeRecorder
        final_recorder = EpisodeRecorder(
            model=model,
            traffic_density=curriculum.get_current_density(),
            max_steps=cfg["video_episode_steps"],
            fps=15,
        )
        final_recorder.record(final_step)

        # ── Cleanup ───────────────────────────────────────────────────────────
        vec_env.close()

        elapsed = time.time() - start_time
        print(f"\n[Train] Training complete!")
        print(f"  Total steps     : {final_step:,}")
        print(f"  Total time      : {elapsed / 3600:.2f} hours")
        print(f"  Total checkpoint: {CHECKPOINT_DIR}/sac_agent_step{final_step:07d}_final.zip")
        print(f"  Videos saved to : {VIDEO_DIR}/")
        print(f"  TensorBoard     : tensorboard --logdir {LOG_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaDrive SAC Training")
    parser.add_argument("--test", type=int, default=None,
                        help="Quick smoke test with N steps (e.g. --test 1000)")
    parser.add_argument("--envs", type=int, default=None,
                        help="Number of parallel environments (default from config)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    train(
        total_timesteps=args.test,
        n_envs=args.envs,
        resume_path=args.resume,
        seed=args.seed,
    )
