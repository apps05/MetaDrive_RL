#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_demo_utils import run_demo


if __name__ == "__main__":
    run_demo(
        title="STRAIGHT ROAD - WITH TRAFFIC",
        checkpoint_path="checkpoints/sac_step4000000_final.zip",
        map_name="S",
        traffic_density=0.1,
        output_name="STRAIGHT_ROAD_WITH_TRAFFIC.mp4",
        max_steps=2000,
    )