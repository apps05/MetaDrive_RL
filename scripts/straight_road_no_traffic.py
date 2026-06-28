#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_demo_utils import run_demo


if __name__ == "__main__":
    run_demo(
        title="STRAIGHT ROAD - NO TRAFFIC",
        checkpoint_path="checkpoints/sac_antigravity_step3000000.zip",
        map_name="S",
        traffic_density=0.0,
        output_name="STRAIGHT_ROAD_NO_TRAFFIC.mp4",
        max_steps=1500,
    )