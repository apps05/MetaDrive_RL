#!/usr/bin/env python3
"""Run a trained checkpoint on a simpler map and save periodic screenshots.

Usage:
  python scripts/run_checkpoint_simple_map.py --checkpoint checkpoints/sac_step4000000_final.zip --screenshot_every 50

The script will run until KeyboardInterrupt (Ctrl+C). Screenshots are saved to videos/.
"""

import os
import sys
import time
import argparse

# Headless flags
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("PYGLET_HEADLESS", "true")
os.environ.setdefault("DISPLAY", ":99")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import SAC
from env.metadrive_env import MetaDriveAgentEnv
from configs.config import ENV_CONFIG, VIDEO_DIR

try:
    import imageio
except Exception:
    imageio = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sac_agent_step2800000.zip")
    parser.add_argument("--screenshot_every", type=int, default=50, help="Steps between screenshots")
    parser.add_argument("--traffic", type=float, default=0.0)
    parser.add_argument("--map", type=str, default="OOO", help="Simple map string (default: OOO)")
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        print(f"[Error] Checkpoint not found: {ckpt}")
        return

    print("[1] Loading model:", ckpt)
    model = SAC.load(ckpt, device="cpu")
    print("    ✓ Model loaded")

    # Build a simple config based on ENV_CONFIG but with an easier map
    cfg = {**ENV_CONFIG}
    cfg.update({
        "map": args.map,
        "num_scenarios": 100,
        "start_seed": 0,
        "traffic_density": args.traffic,
        "use_render": True,
    })

    print(f"[2] Creating environment with map='{args.map}', traffic={args.traffic}")
    env = MetaDriveAgentEnv(config=cfg, traffic_density=args.traffic)

    print("    ✓ Environment created")
    obs, info = env.reset()
    done = False
    step = 0

    base_name = os.path.splitext(os.path.basename(ckpt))[0]
    os.makedirs(VIDEO_DIR, exist_ok=True)

    print("[3] Running until you press Ctrl+C. Will save screenshots to videos/ periodically.")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if step % args.screenshot_every == 0:
                # Try engine.screenshot()
                try:
                    if hasattr(env, 'engine') and hasattr(env.engine, 'screenshot'):
                        filename = env.engine.screenshot()
                        if filename is not None:
                            if imageio is not None:
                                try:
                                    img = imageio.imread(str(filename))
                                    out_png = os.path.join(VIDEO_DIR, f"{base_name}_step{step}.png")
                                    imageio.imwrite(out_png, img)
                                    print(f"[Screenshot] saved: {out_png}")
                                except Exception as e:
                                    print(f"[Warning] Failed to read screenshot file: {e}")
                            else:
                                print("[Warning] imageio not available to read screenshot file")
                            # try to remove file
                            try:
                                os.remove(str(filename))
                            except Exception:
                                pass
                except Exception as e:
                    pass

            if done:
                print(f"[Info] Episode ended at step {step}. Resetting environment.")
                obs, info = env.reset()
                done = False

            # small sleep to avoid hogging CPU/GPU
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user. Leaving environment open for inspection.")
        # Keep environment alive; do not call env.close() so you can attach or inspect
        try:
            print(f"You can find screenshots in: {VIDEO_DIR}")
        except Exception:
            pass


if __name__ == '__main__':
    main()
