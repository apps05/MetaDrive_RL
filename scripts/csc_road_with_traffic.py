#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from road_demo_utils import run_demo


if __name__ == "__main__":
    run_demo(
        title="CSC ROAD - WITH TRAFFIC",
        checkpoint_path="checkpoints/sac_agent_step3000000.zip",
        map_name="CSC",
        traffic_density=0.1,
        output_name="CSC_ROAD_WITH_TRAFFIC.mp4",
        max_steps=2000,
    )