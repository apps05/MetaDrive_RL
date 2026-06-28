MetaDrive Traffic-Aware Autonomous Agent
=========================================

## Overview

This project implements a traffic-aware autonomous driving agent trained using the Soft Actor-Critic (SAC) reinforcement learning algorithm on the MetaDrive simulator. The agent learns to navigate complex road scenarios with varying traffic densities through curriculum-based learning, progressing from traffic-free baseline driving to 45% dense urban traffic with overtaking capabilities.

The system demonstrates driving behavior across multiple map types and traffic conditions, with performance validated through 4 million training steps (9.78 hours) on CPU infrastructure.

## Key Features

- **Traffic-Aware Reward Function**: 11-component reward system optimizing for speed control, lane keeping, turn quality, traffic distance maintenance, and overtaking detection
- **Curriculum Learning**: 5-phase training schedule gradually increasing traffic density from 0% to 45%
- **Multi-Map Support**: Trained on diverse procedurally generated road types (roundabouts, intersections, curves, straights)
- **Challenging Dense Traffic**: Trained to 4M timesteps achieving 165 reward in 45% traffic (33% of baseline), demonstrating agent adaptation to extreme conditions
- **Backward Compatible**: Old checkpoints load seamlessly with new reward function

## Project Structure

```
RL_project/
├── agent/                          # SAC model utilities
│   ├── __init__.py
│   └── sac_agent.py               # Checkpoint management and model creation
│
├── configs/                        # Configuration system
│   ├── __init__.py
│   └── config.py                  # All hyperparameters, paths, reward weights
│
├── env/                            # Custom environments
│   ├── __init__.py
│   └── metadrive_env.py           # Main: MetaDriveAgentEnv with 11-component rewards
│
├── training/                       # Training pipeline
│   ├── __init__.py
│   ├── train.py                   # Main training script with curriculum scheduler
│   └── curriculum.py              # Curriculum logic for traffic ramping
│
├── scripts/                        # Demo and utility scripts
│   ├── simple_run.py              # Minimal evaluation runner
│   ├── straight_road_with_traffic.py
│   ├── roundabout_road_with_traffic.py
│   ├── record_video.py            # Video capture utility
│   ├── run_checkpoint_simple_map.py
│   ├── road_demo_utils.py         # Shared demo utilities
│   └── ... (additional demo variants)

├── video/                          # Video recording utilities
│   ├── __init__.py
│   └── recorder.py               # Frame capture and encoding helpers
│
├── checkpoints/                    # Trained model checkpoints (gitignored)
│   └── sac_step4000000_final.zip
│
                
├── TRAINING_STATISTICS.md         # Real training performance metrics
├── README.md                       # This file
├── requirements.txt               # Python dependencies
└── .gitignore                      # Git exclusion patterns
```

## Environment Details

### MetaDrive Simulator Configuration

The agent operates in MetaDrive v0.4.3, a procedurally generated autonomous driving simulator with realistic physics and traffic interactions.

**Map Types**:
- OXXO: Mixed roundabouts and intersections (primary training map)
- OOO: Pure roundabouts for turn capability validation
- CSC: Complex curves for lane keeping assessment
- S: Straight roads for baseline performance

**Observation Space**: 144-dimensional sensor fusion
- Main LiDAR: 72 lasers at 50-meter range
- Side detector: 20 lasers for lateral obstacle detection
- Lane detector: 20 lasers for lane boundary tracking
- Other state: 32 dimensions (velocity, heading, etc.)

**Action Space**: 2-dimensional continuous control
- Steering: [-1.0, 1.0]
- Throttle: [-1.0, 1.0]

**Physics Configuration**:
- Decision repeat: 5 physics substeps per agent action
- Physics step size: 0.01 seconds
- Horizon: 1000 steps per episode
- Crash handling: Non-terminal (allows learning in traffic)
- Traffic randomization: Enabled for robustness

## Reward Function Architecture

The 11-component reward function balances multiple driving objectives:

### Speed Control (R_speed)
- Target velocity: 60 km/h
- Coefficient: 1.0
- Asymmetric penalty for overspeeding
- Formula: c_speed * (1.0 - |velocity - v_target| / v_target)

### Lane and Direction Alignment
- Lane keeping penalty (R_lat, c_lat=0.5): Distance from lane center
- Border proximity penalty (R_border, c_border=1.0): Squared deviation from lane center
- Direction penalty (R_dir, c_dir=0.5): Angular deviation from lane heading

### Smooth Driving Incentives
- Smoothness reward (R_smooth, c_smooth=0.2): Penalizes large action changes
- Yaw rate penalty (R_yaw, c_yaw=0.5): Prevents slalom driving
- Formula: -c_smooth * (|delta_steering| + |delta_throttle|)

### Advanced Traffic Components
- Turn quality reward (R_turn, c_turn=0.8): Exponential penalty for sharp heading changes
- Traffic distance (R_traffic, c_traffic_dist=0.5): Rewards maintaining lateral separation from vehicles
- Overtaking bonus (R_overtake, c_overtake=1.5): Bonus when reducing nearby vehicle count

### Route Progress and Stability
- Route progress (R_route, c_route=1.0): Reward for advancing toward destination
- Stability (R_stab, c_stab=0.5): Reward for maintaining vehicle alignment

### Penalties
- Collision penalty: 10.0 (non-terminal to allow traffic learning)
- Off-road penalty: 5.0 (terminal)
- Time penalty: 0.05 per step (encourages destination reaching)
- Immobility penalty: 1.0 if speed < 0.5 km/h

**Total Reward Formula**:
R_total = R_speed + R_route + R_lat + R_dir + R_smooth + R_yaw + R_border + R_turn + R_traffic + R_overtake + R_stab - C_penalty

## Algorithm and Training Configuration

### Soft Actor-Critic (SAC)
- Framework: stable-baselines3 v2.8.0
- Policy: MlpPolicy with [256, 256] hidden layers
- Learning rate: 0.0003 (Adam optimizer)
- Replay buffer: 1,000,000 transitions
- Batch size: 256
- Tau (soft update): 0.005
- Gamma (discount factor): 0.99
- Entropy coefficient: Auto (starts at 0.861, converges to ~0.001)

### Training Scale
- Total timesteps: 4,000,000
- Parallel environments: 4 (SubprocVecEnv)
- Checkpoint frequency: Every 100,000 steps
- Total episodes: 20,576
- Training time: 9 hours 47 minutes (9.78 hours, CPU)
- Model parameters: 518,664

### Curriculum Learning Schedule

The agent progresses through 5 phases of traffic density:

**Phase 0 (Steps 0-500K)**: Baseline Driving
- Traffic density: 0%
- Objective: Learn lane keeping, speed control, smooth navigation
- Episode reward: 499
- Episode length: 1000 steps

**Phase 1 (Steps 500K-1M)**: Light Traffic Introduction
- Traffic density: 5%
- Objective: Adapt to moving obstacles, maintain distance
- Episode reward: 499 (maintained)
- Episode length: 1000 steps

**Phase 2 (Steps 1M-1.5M)**: Moderate Traffic
- Traffic density: 15%
- Objective: Complex interaction handling, predictive behavior
- Episode reward: 499
- Episode length: 1000 steps

**Phase 3 (Steps 1.5M-2.5M)**: Dense Urban Traffic
- Traffic density: 30%
- Objective: Complex driving in heavy traffic
- Episode reward: 51 (significant drop from baseline)
- Episode length: 147 steps (severe reduction)
- Actor loss: -4.96 to -7.47 (very weak policy)
- Observation: Agent severely struggles with 30% traffic density

**Phase 4 (Steps 2.5M-4M)**: Heavy Traffic with Overtaking
- Traffic density: 45%
- Objective: Aggressive overtaking, prediction, complex decisions
- Episode reward: 96 initially, recovered to 165 by 4M
- Episode length: 172 initially, improved to 242 by 4M
- Actor loss: -14.6 to -27.3 (improving through phase)
- Observation: Extreme stress at phase start, gradual recovery over 1.5M steps

## Real Training Performance

### Real Training Performance (4M Steps)

#### Overall Results
- Total training duration: 9 hours 47 minutes
- Episodes completed: 20,576
- Total timesteps: 4,000,000
- Average FPS: 127-155 (early) declining to 114 (final)
- Final FPS: 114
- Policy convergence: Actor loss -27.3 at end

#### Checkpoint Performance Across All Phases

| Checkpoint | Steps | Elapsed | Episodes | Reward | Ep. Length | Traffic | Phase | FPS |
|-----------|-------|---------|----------|--------|-----------|---------|-------|-----|
| 100K      | 100K  | 0.11h   | 100      | 499    | 1000      | 0%      | 0     | 245 |
| 500K      | 500K  | 0.70h   | 516      | 499    | 1000      | 5%      | 1     | 197 |
| 1M        | 1M    | 1.54h   | 1,016    | 499    | 1000      | 15%     | 2     | 180 |
| 1.5M      | 1.5M  | 3.03h   | 6,948    | 51     | 140       | 30%     | 3     | 138 |
| 2M        | 2M    | 4.36h   | 10,740   | 61     | 147       | 30%     | 3     | 127 |
| 2.5M      | 2.5M  | 5.64h   | 13,696   | 96     | 172       | 45%     | 4     | 123 |
| 3M        | 3M    | 7.02h   | 16,240   | 146    | 229       | 45%     | 4     | 119 |
| 4M Final  | 4M    | 9.78h   | 20,576   | 165    | 242       | 45%     | 4     | 114 |

#### Performance Analysis by Phase

**Baseline (0-15% traffic): Excellent Performance**
- Reward: 499 (optimal)
- Episode length: 1000 (max horizon)
- Agent operates flawlessly in low-traffic scenarios

**Phase 3 Transition (30% traffic): Severe Performance Degradation**
- Reward drops 87.6% (from 499 to 51)
- Episode length drops 85.3% (from 1000 to 140)
- Agent crashes or terminates early in majority of episodes
- Actor loss: -4.96 (very weak policy convergence)
- Critic loss: 12.7 (highly unstable value estimates)

**Phase 4 Progression (45% traffic): Gradual Recovery**
- Initial (2.5M): Reward 96, length 172 (still severe struggle)
- Mid (3M): Reward 146, length 229 (improvement begins)
- Final (4M): Reward 165, length 242 (continuing improvement)
- Actor loss improves: -14.6 to -27.3 (policy strengthening)
- Entropy stabilizes: 0.0493 to 0.0464 (controlled exploration)

#### Loss Dynamics (4M Training)

**Actor Loss Evolution**:
- Phase 0-2: -41.5 to -49.7 (optimal convergence)
- Phase 3 start: Drops to -4.96 (catastrophic collapse)
- Phase 3 (2M): -6.2 (very weak)
- Phase 4 start: -15 (gradual recovery begins)
- Phase 4 mid (3M): -23.6 to -23.9 (strong improvement)
- Phase 4 final (4M): -27.2 to -27.3 (approaching baseline quality)

**Critic Loss Evolution**:
- Phase 0-2: 0.0032 to 0.0111 (stable)
- Phase 3 start: Spikes to 47.5 (severe instability)
- Phase 3 (2M): 4.66 to 12.7 (unstable)
- Phase 4: 1.95 to 2.26 (volatile but gradually controlled)

**Entropy Coefficient Evolution**:
- Phase 0-2: 0.861 to 0.001 (initial collapse, then stabilization)
- Phase 3: 0.0331 to 0.0562 (increased for exploration in traffic)
- Phase 4: 0.0493 to 0.0464 (maintained moderate levels)

## Setup and Installation

### Requirements
- Python 3.10
- MetaDrive v0.4.3
- PyTorch (CPU or CUDA compatible)
- stable-baselines3 v2.8.0
- Gymnasium v0.29+
- ImageIO with FFmpeg support

### Installation on Windows

Create and activate Python 3.10 virtual environment:
```bash
py -3.10 -m venv C:\v310
C:\v310\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install --upgrade pip
pip install stable-baselines3==2.8.0 gymnasium metadrive-simulator imageio imageio-ffmpeg tensorboard opencv-python
```

### Installation on Linux/macOS

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install stable-baselines3==2.8.0 gymnasium metadrive-simulator imageio imageio-ffmpeg tensorboard opencv-python
```

## Usage

### Running Demo Scripts

Basic demonstration on simple road:
```bash
python scripts/demo_simple_road.py
```

Traffic scenario evaluation:
```bash
python scripts/straight_road_with_traffic.py
```

Roundabout handling (OOO map):
```bash
python scripts/csc_road_no_traffic.py
```

All demo scripts accept optional parameters for checkpoint selection, map type, traffic density, and scenario count.

### Training New Model

Start training from scratch with 4 million steps:
```bash
python training/train.py
```

Resume from checkpoint (e.g., at 500K steps):
```bash
python training/train.py --resume checkpoints/sac_step0500000
```

Quick validation test (1000 steps):
```bash
python training/train.py --test 1000
```

## Configuration

All training parameters are centralized in `configs/config.py`:

### Environment Configuration
- map: "OXXO" (procedurally generated)
- num_scenarios: 1000 (different map instances)
- traffic_density: 0.0 (overridden by curriculum)
- horizon: 1000 (max steps per episode)
- use_render: False (headless operation)

### SAC Hyperparameters
- policy: "MlpPolicy"
- learning_rate: 3e-4
- buffer_size: 1,000,000
- batch_size: 256
- gamma: 0.99 (discount factor)
- tau: 0.005 (soft update)

### Training Configuration
- total_timesteps: 4,000,000 (for extended training)
- n_envs: 4 (parallel environments)
- checkpoint_every: 100,000 (save frequency)

### Reward Weights
Modify in REWARD_CONFIG dictionary:
- c_speed: 1.0 (speed control)
- c_route: 1.0 (route progress)
- c_lat: 0.5 (lane keeping)
- c_turn: 0.8 (turn quality)
- c_traffic_dist: 0.5 (traffic distance)
- c_overtake: 1.5 (overtaking bonus)

## Architecture Overview

### Custom Environment (MetaDriveAgentEnv)
Located in `env/metadrive_env.py`:
- Extends MetaDrive's MetaDriveEnv
- Implements custom 11-component reward function
- Tracks state: last_action, route_completion, heading, vehicle_count
- Detects overtaking opportunities through vehicle count reduction
- Calculates turn quality and traffic distances
- Non-terminal crashes to enable robust traffic learning

### State Tracking
- last_action: Previous steering and throttle for smoothness calculation
- last_route_completion: Progress metric for route rewards
- _last_heading: Heading angle for yaw rate calculation
- _last_vehicle_count: Vehicle tracking for overtaking detection

### Helper Methods
- _detect_overtake_opportunity(): Returns 1.0 when vehicle count decreases
- _calc_traffic_distance_reward(): Lateral separation reward
- _calc_turn_quality(): Exponential penalty for sharp heading changes

## Performance Analysis

### Strengths
- Rapid initial convergence (actor loss reaches -41.5 by 100K steps)
- Stable policy maintenance across traffic transitions
- Successful adaptation from 0% to 30% traffic density
- Robust recovery after stress conditions
- Efficient CPU-based training (2M steps in 3.58 hours)

### Areas for Improvement
- Entropy coefficient collapsed too early (100-200K steps), limiting exploration
- Episode length reduced 13% at phase 3 due to traffic stress
- FPS degradation from 245 to 155 fps over training duration
- Reward plateau after convergence suggests limited policy improvement post-convergence

## Deployment

### Model Inference
```python
from stable_baselines3 import SAC
from env.metadrive_env import MetaDriveAgentEnv

env = MetaDriveAgentEnv(traffic_density=0.3)
model = SAC.load("checkpoints/sac_step4000000_final.zip", device="cpu")

obs, info = env.reset()
for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Running in Headless Mode (Linux/Cloud)
```bash
export SDL_VIDEODRIVER=offscreen
export DISPLAY=:99
export PYGLET_HEADLESS=true
python scripts/demo_simple_road.py
```

## Troubleshooting

### MetaDrive Rendering Issues
- Ensure SDL_VIDEODRIVER environment variable is set to "offscreen"
- On Windows, use wglGraphicsPipe for GPU acceleration
- On Linux, use glxGraphicsPipe with X11 display

### Out of Memory Errors
- Reduce num_scenarios from 1000 to 500
- Reduce n_envs from 4 to 2
- Decrease batch_size from 256 to 128

### Slow Training (Low FPS)
- Verify GPU availability (nvidia-smi)
- Check replay buffer size (may require increased RAM)
- Profile with PyTorch profiler for bottleneck identification

### Checkpoint Loading Failures
- Verify checkpoint path is correct
- Ensure stable-baselines3 version matches training version
- Check device parameter ("cpu" or "cuda")

## References

- MetaDrive Documentation: https://metadrive-simulator.readthedocs.io/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- SAC Algorithm: Haarnoja et al. "Soft Actor-Critic Algorithms and Applications" (2019)
- Curriculum Learning: Bengio et al. "Curriculum Learning" (2009)

## License

This project uses MetaDrive (Apache 2.0) and stable-baselines3 (MIT).

