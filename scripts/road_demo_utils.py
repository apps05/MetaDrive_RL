#!/usr/bin/env python3
"""Shared runner for small road demo scripts."""

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("PYGLET_HEADLESS", "true")
os.environ.setdefault("DISPLAY", ":99")

try:
    import imageio.v2 as imageio
except Exception:
    import imageio

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.metadrive_env import MetaDriveAgentEnv
from configs.config import ENV_CONFIG, VIDEO_DIR


def run_demo(
    *,
    title: str,
    checkpoint_path: str,
    map_name: str,
    traffic_density: float,
    output_name: str,
    max_steps: int = 1500,
    frame_interval: int = 15,
    start_seed: int = 0,
):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

    print(f"\n[1] Loading checkpoint: {os.path.basename(checkpoint_path)}")
    model = SAC.load(checkpoint_path, device="cpu")
    print("    ✓ Weights loaded")

    demo_config = {
        **ENV_CONFIG,
        "map": map_name,
        "num_scenarios": 1,
        "start_seed": start_seed,
        "traffic_density": traffic_density,
        "use_render": True,
        "horizon": max_steps,
        "decision_repeat": 5,
    }

    print(f"\n[2] Creating environment for map: {map_name}")
    print(f"    - Traffic density: {traffic_density}")
    print(f"    - Max steps: {max_steps}")
    env = MetaDriveAgentEnv(config=demo_config, traffic_density=traffic_density)
    print("    ✓ Environment created")

    print("\n[3] Running agent...")
    obs, info = env.reset()
    frames = []
    step = 0
    start_time = time.time()

    try:
        while step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if step % frame_interval == 0:
                try:
                    if hasattr(env, "engine") and hasattr(env.engine, "screenshot"):
                        filename = env.engine.screenshot()
                        if filename is not None:
                            frame = imageio.imread(str(filename))
                            if isinstance(frame, np.ndarray):
                                frames.append(frame)
                            try:
                                os.remove(str(filename))
                            except Exception:
                                pass
                except Exception:
                    pass

            if step % 100 == 0:
                elapsed = time.time() - start_time
                print(f"    Step {step}/{max_steps} | Frames: {len(frames)} | Time: {elapsed:.1f}s")

            if terminated or truncated:
                print(f"    Episode ended at step {step}")
                break
    finally:
        elapsed = time.time() - start_time
        print(f"\n[4] COMPLETE!")
        print(f"    - Steps: {step}")
        print(f"    - Frames: {len(frames)}")
        print(f"    - Time: {elapsed:.1f}s")

        env.close()

    if frames:
        output_file = os.path.join(VIDEO_DIR, output_name)
        print(f"\n[5] Saving video with {len(frames)} frames...")
        imageio.mimwrite(output_file, frames, fps=15)
        print(f"    ✓ Saved: {output_file}")
    else:
        print(f"\n[5] No frames captured - the run still completed for {step} steps.")

    return step, len(frames)