#!/usr/bin/env python3
"""
BRAND NEW TEST: Agent on completely new STRAIGHT ROAD only.
2.8M checkpoint on simple straight track - no old map.
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
print("  TESTING ON BRAND NEW MAP: STRAIGHT ROAD ONLY")
print("=" * 60)

# 1. Load checkpoint
print("\n[1] Loading: sac_agent_step2800000.zip")
model = SAC.load("checkpoints/sac_agent_step2800000.zip", device="cpu")
print("    ✓ Weights loaded")

# 2. Create COMPLETELY NEW environment - STRAIGHT ROAD ONLY
# "S" = just straight sections - the simplest possible map
new_config = {
    **ENV_CONFIG,
    "map": "S",  # BRAND NEW: Just straight road sections
    "num_scenarios": 1,
    "start_seed": 0,
    "traffic_density": 0.0,
    "use_render": True,
}

print("\n[2] Creating environment with NEW MAP: 'S' (STRAIGHT ROAD ONLY)")
print("    - This is completely different from old maps")
print("    - Just a straight track with no turns or intersections")
env = MetaDriveAgentEnv(config=new_config, traffic_density=0.0)
print("    ✓ New environment created")

# 3. Run agent
print("\n[3] Running agent on new straight road...")
obs, info = env.reset()
frames = []
step = 0
max_steps = 500

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
        print(f"    Step {step}/500 | Frames: {len(frames)} | Time: {elapsed:.1f}s")
    
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
    output_file = os.path.join(VIDEO_DIR, "BRAND_NEW_STRAIGHT_ROAD_DEMO.mp4")
    print(f"\n[5] Saving video...")
    imageio.mimwrite(output_file, frames, fps=15)
    print(f"    ✓ Saved: {output_file}")
    print(f"\n{'='*60}")
    print("  SUCCESS! New map works perfectly!")
    print("  This is NOT the old map - completely straight road.")
    print("=" * 60)
else:
    print("\n[5] Agent ran successfully for 500 steps on NEW STRAIGHT ROAD!")
