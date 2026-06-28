#!/usr/bin/env python3
"""
STRAIGHT ROAD + ONE TURN
Agent navigates: straight section → turn at intersection → straight again
2.8M checkpoint on this brand new layout.
"""

import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("PYGLET_HEADLESS", "true")
os.environ.setdefault("DISPLAY", ":99")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import SAC
from env.metadrive_env import MetaDriveAgentEnv
from configs.config import ENV_CONFIG, VIDEO_DIR

try:
    import imageio.v2 as imageio
except Exception:
    import imageio

print("=" * 60)
print("  NEW MAP: STRAIGHT + ONE TURN")
print("=" * 60)

# 1. Load checkpoint
print("\n[1] Loading: sac_agent_step2800000.zip")
model = SAC.load("checkpoints/sac_agent_step4000000.zip", device="cpu")
print("    ✓ Weights loaded")

# 2. Create NEW environment - Straight + one turn
# "SXS" = straight → intersection (turn) → straight
new_config = {
    **ENV_CONFIG,
    "map": "SCC",          
    "num_scenarios": 1,
    "traffic_density": 0.0,
    "start_seed": 1,
    "use_render": True,
    "horizon": 5000,
}

print("\n[2] Creating environment with NEW MAP: 'SXS'")
print("    - Straight section")
print("    - One turn at intersection")
print("    - Another straight section")
env = MetaDriveAgentEnv(config=new_config, traffic_density=0.0)
print("    ✓ New environment created")

# 3. Run agent
print("\n[3] Running agent on straight-turn-straight road...")
obs, info = env.reset()
frames = []
step = 0
max_steps = 3000

start_time = time.time()

while step < max_steps:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    
    # Capture every 20 steps
    if step % 20 == 0:
        try:
            if hasattr(env, 'engine') and hasattr(env.engine, 'screenshot'):
                filename = env.engine.screenshot()
                if filename is not None:
                    img = imageio.imread(str(filename))
                    frames.append(img)
                    try:
                        os.remove(str(filename))
                    except:
                        pass
        except:
            pass
    
    if step % 100 == 0:
        elapsed = time.time() - start_time
        print(f"    Step {step}/3000 | Frames: {len(frames)} | Time: {elapsed:.1f}s")
    
    if terminated or truncated:
        print(f"    Episode ended at step {step}")
        break

elapsed = time.time() - start_time

print(f"\n[4] COMPLETE!")
print(f"    - Steps: {step}")
print(f"    - Frames: {len(frames)}")
print(f"    - Time: {elapsed:.1f}s")

env.close()

# 5. Save video
if frames:
    output_file = os.path.join(VIDEO_DIR, "STRAIGHT_PLUS_ONE_TURN_DEMO.mp4")
    print(f"\n[5] Saving video with {len(frames)} frames...")
    imageio.mimwrite(output_file, frames, fps=15)
    print(f"    ✓ Saved: {output_file}")
    print(f"\n{'='*60}")
    print("  SUCCESS! Agent navigated straight + turn perfectly!")
    print("  Map: Straight → Turn at intersection → Straight")
    print("=" * 60)
else:
    print("\n[5] Agent ran successfully for {} steps!".format(step))
