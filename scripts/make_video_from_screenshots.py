#!/usr/bin/env python3
"""Assemble screenshots saved in videos/ into an MP4.
Usage: python scripts/make_video_from_screenshots.py --pattern sac_agent_step2800000_*.png --out demo.mp4 --fps 15
"""
import os
import glob
import argparse

try:
    import imageio.v2 as imageio
except Exception:
    import imageio

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import VIDEO_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--pattern', type=str, required=True)
parser.add_argument('--out', type=str, default='demo.mp4')
parser.add_argument('--fps', type=int, default=15)
args = parser.parse_args()

pattern = os.path.join(VIDEO_DIR, args.pattern)
files = sorted(glob.glob(pattern))
if not files:
    print('[Error] No files found for pattern:', pattern)
    raise SystemExit(1)

frames = []
for f in files:
    img = imageio.imread(f)
    frames.append(img)

out_path = os.path.join(VIDEO_DIR, args.out)
print(f"Writing {len(frames)} frames to {out_path} @ {args.fps} fps")
imageio.mimwrite(out_path, frames, fps=args.fps)
print('Done:', out_path)
