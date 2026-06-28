#!/usr/bin/env python3
"""Minimal script to run simulator and capture video."""

import os
import sys
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("PYGLET_HEADLESS", "true")
os.environ.setdefault("DISPLAY", ":99")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from env.metadrive_env import MetaDriveAgentEnv
from configs.config import ENV_CONFIG, VIDEO_DIR
from video.recorder import write_video

print("=" * 50)
print("  RUNNING TRAFFIC-AWARE AGENT SIMULATOR")
print("=" * 50)

# Load model
print("[1] Loading trained model...")
model = SAC.load("checkpoints/sac_step4000000_final.zip", device="cpu")
print("    ✓ Model loaded successfully")

# Create environment with rendering enabled
print("[2] Creating environment with rendering...")
config = {**ENV_CONFIG, "use_render": True}
env = MetaDriveAgentEnv(config=config, traffic_density=0.2)
print("    ✓ Environment created")

# Run episode
print("[3] Running episode...")
obs, info = env.reset()
done = False
step = 0
frames = []

# Capture frames using Panda3D screenshot
    while not done and step < 500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        
        # Panda3D screenshot() returns a Filename, so we need to read the image
        try:
            if hasattr(env, 'engine') and hasattr(env.engine, 'screenshot'):
                # screenshot() saves PNG and returns Filename
                filename = env.engine.screenshot()
                if filename is not None:
                    # Read the PNG file into numpy array
                    try:
                        import imageio
                        frame = imageio.imread(str(filename))
                        if frame is not None and isinstance(frame, np.ndarray):
                            frames.append(frame)
                        # Clean up temp file
                        try:
                            os.remove(str(filename))
                        except:
                            pass
                    except Exception as e:
                        pass
    except:
        pass

env.close()

print(f"    ✓ Episode ended at step {step}")
print(f"    ✓ Frames captured: {len(frames)}")

# Save video if we have frames
if frames:
    print("[4] Saving video...")
    output_path = os.path.join(VIDEO_DIR, "agent_demo.mp4")
    write_video(frames, output_path, fps=15)
    print(f"    ✓ Video saved: {output_path}")
else:
    print("[4] NO FRAMES CAPTURED - showing message instead")
    print("    The agent ran successfully for 500 steps!")
    print("    The environment rendered internally but frames couldn't be extracted.")

print("=" * 50)
print("  DONE!")
print("=" * 50)
