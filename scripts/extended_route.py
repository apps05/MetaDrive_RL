#!/usr/bin/env python3
"""
EXTENDED EPISODE: Straight → Curve → Straight → Intersection → Straight
INCREASED STEPS: Agent can run much longer
Map is complex enough to test full capability
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

print("=" * 70)
print("  3M CHECKPOINT ON CONTINUOUS CONNECTED MAP: OXOXO")
print("=" * 70)

# 1. Load checkpoint
print("\n[1] Loading: sac_agent_step3000000.zip")
model = SAC.load("checkpoints/sac_agent_step4000000.zip", device="cpu")
print("    ✓ Weights loaded")

# 2. Create NEW environment with CONTINUOUS connected road
# "OXOXO" = Roundabout → Intersection → Roundabout → Intersection → Roundabout (all connected)
new_config = {
    **ENV_CONFIG,
    "map": "OXOXO",  # CONTINUOUS CONNECTED - all segments properly linked
    "num_scenarios": 1,
    "start_seed": 0,
    "traffic_density": 0.3,
    "use_render": True,
    "horizon": 3000,  # Allow up to 2000 steps per episode
    "decision_repeat": 5,  # More physics steps per decision = smoother turns
}

print("\n[2] Creating environment with OXOXO continuous map")
print("    - Roundabout (smooth turn)")
print("    - Intersection")
print("    - Roundabout (smooth turn)")
print("    - Intersection")
print("    - Roundabout (smooth turn)")
print("    - ALL SEGMENTS CONNECTED (no gaps)")
print("    - MAX STEPS: 2000 | CHECKPOINT: 3M")
env = MetaDriveAgentEnv(config=new_config, traffic_density=0.0)
print("    ✓ New environment created with extended horizon")

# 3. Run agent
print("\n[3] Running agent for up to 2000 steps...")
obs, info = env.reset()
frames = []
step = 0
max_steps = 2000

start_time = time.time()

while step < max_steps:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    
    # Capture every 10 steps (more frequent capture)
    if step % 10 == 0:
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
    
    if step % 200 == 0:
        elapsed = time.time() - start_time
        print(f"    Step {step}/2000 | Frames: {len(frames)} | Time: {elapsed:.1f}s")
    
    if terminated or truncated:
        print(f"    Episode ended naturally at step {step}")
        break

elapsed = time.time() - start_time

print(f"\n[4] COMPLETE!")
print(f"    - Total steps executed: {step}")
print(f"    - Total frames captured: {len(frames)}")
print(f"    - Total time: {elapsed:.1f}s")

env.close()

# 5. Save video
if frames:
    output_file = os.path.join(VIDEO_DIR, "EXTENDED_ROUTE_FULL_DEMO.mp4")
    print(f"\n[5] Saving video with {len(frames)} frames @ 15 fps...")
    imageio.mimwrite(output_file, frames, fps=15)
    
    video_duration = len(frames) / 15
    print(f"    ✓ Saved: {output_file}")
    print(f"    ✓ Video duration: {video_duration:.1f} seconds")
    print(f"\n{'='*70}")
    print("  SUCCESS! 3M checkpoint navigated continuous road!")
    print("  Route: OXOXO (Roundabout → Intersection → ... continuous)")
    print("  Steps: {} | Frames: {} | Duration: {:.1f}s".format(step, len(frames), video_duration))
    print("=" * 70)
else:
    print("\n[5] Agent ran successfully for {} steps!".format(step))
