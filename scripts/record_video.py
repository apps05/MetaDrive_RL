import os
import sys
import numpy as np
from stable_baselines3 import SAC

# MetaDrive/Panda3D work more reliably in headless mode on this machine.
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("PYGLET_HEADLESS", "true")
os.environ.setdefault("DISPLAY", ":99")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.metadrive_env import MetaDriveAgentEnv
from configs.config import ENV_CONFIG, VIDEO_DIR
from video.recorder import write_video

import argparse

def main():
    parser = argparse.ArgumentParser(description="Record 3D video of trained Traffic-Aware 4M-step agent")
    parser.add_argument("--checkpoint", type=str, 
                        default="checkpoints/sac_step4000000_final.zip",
                        help="Path to the model checkpoint ZIP file")
    parser.add_argument("--out", type=str,
                        default="final_agent_evaluation.mp4",
                        help="Output video filename (saved to videos/ directory)")
    args = parser.parse_args()

    print("============================================")
    print("  Traffic-Aware Agent Video Evaluation      ")
    print("============================================")

    # 1. Setup paths
    checkpoint_path = args.checkpoint
    video_path = os.path.join(VIDEO_DIR, args.out)

    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found at {checkpoint_path}")
        return

    # 2. Load Model
    print(f"[Info] Loading model from: {checkpoint_path}")
    model = SAC.load(checkpoint_path, device="cpu")

    from metadrive.component.sensors.rgb_camera import RGBCamera
    import gymnasium as gym
    from metadrive.obs.state_obs import LidarStateObservation

    # 3. Setup Environment for Recording
    print("[Info] Initializing MetaDrive Environment...")
    rec_config = {
        **ENV_CONFIG,
        "map": "OXOXO",  # CONTINUOUS CONNECTED MAP: Roundabout → Intersection → Roundabout → Intersection → Roundabout
        "use_render": True,
        "image_observation": False,
        "traffic_density": 0.0,  # Clean test
        "horizon": 1500,  # Longer episodes
    }
    env = MetaDriveAgentEnv(config=rec_config, traffic_density=0.0)

    # 4. Run Evaluation Episode
    print("[Info] Running evaluation episode (Max 1500 steps on continuous map)...")
    obs, info = env.reset()
    done = False
    step = 0
    frames = []

    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio

    try:
        while not done and step < 1500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            # Capture every 10 steps using engine screenshot
            if step % 10 == 0:
                try:
                    if hasattr(env, 'engine') and hasattr(env.engine, 'screenshot'):
                        filename = env.engine.screenshot()
                        if filename is not None:
                            try:
                                frame = imageio.imread(str(filename))
                                if frame is not None and isinstance(frame, np.ndarray):
                                    frames.append(frame)
                            except:
                                pass
                            # Clean up temp file
                            try:
                                os.remove(str(filename))
                            except:
                                pass
                except Exception as e:
                    pass
    except KeyboardInterrupt:
        print("\n[Info] Interrupted early. Saving frames collected so far...")

    env.close()

    # 5. Save Video
    if frames:
        print(f"\n[Info] Episode completed in {step} steps. Saving {len(frames)} frames to MP4...")
        write_video(frames, video_path, fps=15)
        print("============================================")
        print(f"  SUCCESS! Video saved to: {video_path}")
        print(f"  Steps: {step} | Frames: {len(frames)}")
        print(f"  Map: OXOXO (Continuous - Roundabout → Intersection → ...")
        print("============================================")
    else:
        print(f"[Warning] Episode ran for {step} steps but no frames captured.")
        print("[Info] Note: Agent was running on CONTINUOUS CONNECTED MAP")
        print("       Map: OXOXO (Roundabout → Intersection → Roundabout...)")

if __name__ == "__main__":
    main()
