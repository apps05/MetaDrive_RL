"""
Headless Video Recorder for Anti-Gravity MetaDrive Agent

Captures simulation frames using MetaDrive's top-down renderer
(which works in headless mode) and assembles them into MP4 files.

Triggered every 30 minutes of wall-clock time during training.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from datetime import datetime
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import VIDEO_DIR, TRAIN_CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# Backend selection (OpenCV preferred, imageio fallback)
# ─────────────────────────────────────────────────────────────────────────────

_VIDEO_BACKEND = None

try:
    import cv2
    _VIDEO_BACKEND = "cv2"
except ImportError:
    pass

if _VIDEO_BACKEND is None:
    try:
        import imageio
        _VIDEO_BACKEND = "imageio"
    except ImportError:
        pass

if _VIDEO_BACKEND is None:
    print("[VideoRecorder] WARNING: No video backend found (cv2 / imageio). "
          "Install opencv-python-headless or imageio[ffmpeg].")


# ─────────────────────────────────────────────────────────────────────────────
# Video writer helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_video_cv2(frames: list, path: str, fps: int = 15):
    import cv2
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        if frame.ndim == 2:                # grayscale → BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:          # RGBA → BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] == 3:          # RGB → BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()


def _write_video_imageio(frames: list, path: str, fps: int = 15):
    import imageio
    if not frames:
        return
    # imageio expects RGB uint8
    rgb_frames = []
    for f in frames:
        if f.ndim == 2:
            f = np.stack([f, f, f], axis=-1)
        elif f.shape[2] == 4:
            f = f[:, :, :3]
        rgb_frames.append(f.astype(np.uint8))
    imageio.mimwrite(path, rgb_frames, fps=fps, macro_block_size=None)


def write_video(frames: list, path: str, fps: int = 15):
    """Write a list of numpy frames to an MP4 file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if _VIDEO_BACKEND == "cv2":
        _write_video_cv2(frames, path, fps)
    elif _VIDEO_BACKEND == "imageio":
        _write_video_imageio(frames, path, fps)
    else:
        print(f"[VideoRecorder] Skipping video write — no backend available.")
        return
    size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
    print(f"[VideoRecorder] Saved: {path} ({size_mb:.2f} MB, {len(frames)} frames @ {fps}fps)")


# ─────────────────────────────────────────────────────────────────────────────
# Episode Recorder
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeRecorder:
    """
    Records a single evaluation episode using a dedicated environment.

    Instantiate a fresh AntiGravityMetaDriveEnv with top-down rendering
    enabled and run the trained policy for `max_steps` steps, collecting
    frames along the way.
    """

    def __init__(
        self,
        model,
        traffic_density: float = 0.0,
        max_steps: int = 500,
        fps: int = 15,
    ):
        self.model = model
        self.traffic_density = traffic_density
        self.max_steps = max_steps
        self.fps = fps

    def record(self, global_step: int) -> Optional[str]:
        """
        Run one episode and save to video.

        Returns the path to the saved MP4, or None on failure.
        """
        try:
            from env.antigravity_env import AntiGravityMetaDriveEnv
            from configs.config import ENV_CONFIG
        except ImportError as e:
            print(f"[VideoRecorder] Import error: {e}")
            return None

        # Build a recording-capable env config
        rec_config = {
            **ENV_CONFIG,
            "use_render": False,
            "image_observation": False,
            "traffic_density": self.traffic_density,
        }

        try:
            rec_env = AntiGravityMetaDriveEnv(config=rec_config, traffic_density=self.traffic_density)
        except Exception as e:
            print(f"[VideoRecorder] Could not create recording env: {e}")
            return None

        frames = []
        obs, _ = rec_env.reset()
        done = False
        step = 0

        while not done and step < self.max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = rec_env.step(action)
            done = terminated or truncated
            step += 1

            # Capture top-down frame
            try:
                frame = rec_env.render(mode="top_down_rgb_array")
                if frame is not None and isinstance(frame, np.ndarray):
                    frames.append(frame)
            except Exception:
                # Fallback: generate a simple placeholder frame
                placeholder = np.zeros((200, 400, 3), dtype=np.uint8)
                import cv2 as _cv2
                _cv2.putText(
                    placeholder,
                    f"Step {step} | R={reward:.2f}",
                    (20, 100),
                    _cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                frames.append(placeholder)

        rec_env.close()

        if not frames:
            print("[VideoRecorder] No frames captured — skipping video write.")
            return None

        # Save video
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(VIDEO_DIR, f"episode_step{global_step:07d}_{ts}.mp4")
        write_video(frames, path, fps=self.fps)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Timer-based periodic recorder
# ─────────────────────────────────────────────────────────────────────────────

class PeriodicVideoRecorder:
    """
    Fires every `interval_minutes` of wall-clock time to record and save a
    video of the current policy.

    Thread-safe: uses a threading.Timer and a shared state reference.
    """

    def __init__(
        self,
        model_ref,            # mutable reference: {"model": sac_model}
        curriculum_ref,       # mutable reference: {"density": float}
        interval_minutes: float = 30.0,
        max_steps: int = 500,
        fps: int = 15,
    ):
        self._model_ref = model_ref
        self._curriculum_ref = curriculum_ref
        self._interval_sec = interval_minutes * 60.0
        self._max_steps = max_steps
        self._fps = fps
        self._timer: Optional[threading.Timer] = None
        self._global_step_ref: dict = {"step": 0}
        self._stopped = False

    def start(self):
        """Start the periodic recording timer."""
        self._stopped = False
        self._schedule_next()
        print(
            f"[PeriodicVideoRecorder] Started — will record every "
            f"{self._interval_sec / 60:.1f} minutes."
        )

    def stop(self):
        """Stop the timer."""
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()
        print("[PeriodicVideoRecorder] Stopped.")

    def update_step(self, step: int):
        """Update the global step counter (called from training loop)."""
        self._global_step_ref["step"] = step

    def _schedule_next(self):
        if self._stopped:
            return
        self._timer = threading.Timer(self._interval_sec, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self):
        """Record an episode and save the video."""
        if self._stopped:
            return

        model = self._model_ref.get("model")
        density = self._curriculum_ref.get("density", 0.0)
        step = self._global_step_ref["step"]

        if model is None:
            print("[PeriodicVideoRecorder] Model not yet available — skipping.")
        else:
            print(f"\n[PeriodicVideoRecorder] Recording episode at step {step:,} ...")
            recorder = EpisodeRecorder(
                model=model,
                traffic_density=density,
                max_steps=self._max_steps,
                fps=self._fps,
            )
            path = recorder.record(step)
            if path:
                print(f"[PeriodicVideoRecorder] ✓ Video saved: {path}")
            else:
                print("[PeriodicVideoRecorder] ✗ Recording failed.")

        # Schedule next recording
        self._schedule_next()


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[VideoRecorder] Backend:", _VIDEO_BACKEND)
    # Create a dummy frame sequence and write a test video
    frames = [np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8) for _ in range(30)]
    test_path = os.path.join(VIDEO_DIR, "test_video.mp4")
    write_video(frames, test_path)
    print("[VideoRecorder] Smoke test complete.")
