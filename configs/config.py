"""
MetaDrive Traffic-Aware Agent — Centralized Configuration
All hyperparameters and paths defined here.

Training Setup
──────────────
Agent trained on multiple maps and traffic conditions using SAC algorithm.
Curriculum learning increases traffic density over 4M training steps.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

for d in [LOG_DIR, VIDEO_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Environment ─────────────────────────────────────────────────────────────
# Agent trained on multiple map types and traffic conditions
TRAINING_MAPS = {
    "OXXO": "Mixed roundabouts & intersections (primary training)",
    "OOO": "Pure roundabouts (turn capability)",
    "CSC": "Complex curves (lane keeping)",
    "S": "Straight road (baseline)",
}

ENV_CONFIG = {
    # Procedural map: O = roundabout, X = intersection
    # Training conducted on multiple map types (see TRAINING_MAPS)
    "map": "OXXO",
    "num_scenarios": 1000,
    "start_seed": 42,

    # Traffic: starts at 0, increases via curriculum over 4M steps
    "traffic_density": 0.0,
    "random_traffic": True,

    # Observation: LiDAR state vector (259-dim)
    "use_render": False,
    "image_observation": False,
    "vehicle_config": {
        "lidar": {
            "num_lasers": 72,
            "distance": 50,
            "num_others": 4,
        },
        "side_detector": {"num_lasers": 20, "distance": 50},
        "lane_line_detector": {"num_lasers": 20, "distance": 20},
    },

    # Physics
    "decision_repeat": 5,       # sim substeps per agent step
    "physics_world_step_size": 1e-2,

    # Episode limits
    "horizon": 1000,

    # Reward shaping (used in custom reward function)
    "success_reward": 10.0,
    "out_of_road_penalty": 5.0,
    "crash_vehicle_penalty": 5.0,
    "crash_object_penalty": 5.0,

    # Termination
    "out_of_road_done": True,
    "crash_vehicle_done": True,
    "crash_object_done": True,
}



# ─── SAC Hyperparameters ─────────────────────────────────────────────────────
SAC_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto",
    "target_update_interval": 1,
    "gradient_steps": 1,
    "learning_starts": 10_000,
    "train_freq": 1,
    "policy_kwargs": {
        "net_arch": [256, 256],
    },
    "verbose": 1,
    "device": "auto",   # "cuda" if available, else "cpu"
}

# ─── Training ─────────────────────────────────────────────────────────────────
# Extended training to 4M steps with 5-phase curriculum for traffic learning
TRAIN_CONFIG = {
    "total_timesteps": 4_000_000,  # Extended from 2M to 4M for better convergence
    "n_envs": 4,                    # 4 parallel environments (SubprocVecEnv)
    "checkpoint_every": 100_000,    # Checkpoint every 100k steps
    "video_interval_minutes": 30,   # Wall-clock minutes between video captures
    "video_episode_steps": 500,     # Steps to record per video
    "tensorboard_log": LOG_DIR,
    "seed": 42,
}

# ─── Curriculum (5-Phase Traffic Learning) ────────────────────────────────────
# Gradual increase in traffic density to enable robust learning
CURRICULUM_CONFIG = {
    "schedule": [
        (0,           0.0),   # Phase 0 (0-500k):   No traffic (baseline driving)
        (500_000,     0.05),  # Phase 1 (500k-1M):  Light traffic (5%)
        (1_000_000,   0.15),  # Phase 2 (1M-1.5M):  Moderate traffic (15%)
        (1_500_000,   0.3),   # Phase 3 (1.5M-2.5M): Dense traffic (30%)
        (2_500_000,   0.45),  # Phase 4 (2.5M-4M):   Heavy traffic (45%) + overtaking
    ]
}

# ─── Reward Function Weights ──────────────────────────────────────────────────
REWARD_CONFIG = {
    # Speed control
    "v_target": 60.0,                 # Target speed in km/h
    "c_speed": 1.0,                   # Speed control coefficient
    
    # Route progress
    "c_route": 1.0,                   # Route progress coefficient (longitudinal)
    
    # Lane keeping
    "c_lat": 0.5,                     # Lane center deviation penalty coefficient
    "c_dir": 0.5,                     # Direction angular deviation penalty coefficient
    
    # Smooth driving
    "c_smooth": 0.2,                  # Smooth driving penalty (steering/throttle delta)
    "c_yaw": 0.5,                     # Yaw rate penalty to prevent slalom
    "c_border": 1.0,                  # Border proximity gradient penalty
    
    # Turn quality & traffic awareness
    "c_turn": 0.8,                    # Turn execution reward (smooth turns)
    "c_traffic_dist": 0.5,            # Traffic distance reward (safe spacing)
    "c_overtake": 1.5,                # Overtaking bonus reward
    "c_lane_change": 0.3,             # Lane change smoothness penalty
    
    # Stability
    "c_stab": 0.5,                    # Stability reward weight
    
    # Penalties
    "collision_penalty": 10.0,        # Non-terminal (allows learning in traffic)
    "off_road_penalty": 5.0,
    "time_penalty": 0.05,             # Per-step cost to encourage reaching destination
}
