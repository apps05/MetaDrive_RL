#!/usr/bin/env python3
"""Debug script to understand MetaDrive rendering capabilities."""

import os
import sys
import numpy as np

# Headless mode
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("PYGLET_HEADLESS", "true")
os.environ.setdefault("DISPLAY", ":99")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.metadrive_env import MetaDriveAgentEnv
from configs.config import ENV_CONFIG

# Create environment
print("[1] Creating environment...")
config = {
    **ENV_CONFIG,
    "use_render": True,
}
env = MetaDriveAgentEnv(config=config, traffic_density=0.0)

# Reset
print("[2] Resetting environment...")
obs, info = env.reset()

# Take a step
print("[3] Taking one step...")
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# Check environment attributes
print("\n[DEBUG] Environment attributes:")
print(f"  has render method: {hasattr(env, 'render')}")
print(f"  has get_image method: {hasattr(env, 'get_image')}")
print(f"  has vehicle: {hasattr(env, 'vehicle')}")

if hasattr(env, 'vehicle'):
    vehicle = env.vehicle
    print(f"  vehicle has sensors: {hasattr(vehicle, 'sensors')}")
    if hasattr(vehicle, 'sensors'):
        print(f"    sensors type: {type(vehicle.sensors)}")
        print(f"    sensors: {vehicle.sensors}")

print(f"  has agent: {hasattr(env, 'agent')}")
if hasattr(env, 'agent'):
    agent = env.agent
    print(f"  agent has sensors: {hasattr(agent, 'sensors')}")
    if hasattr(agent, 'sensors'):
        print(f"    sensors type: {type(agent.sensors)}")
        print(f"    sensors: {agent.sensors}")

# Try different render modes
print("\n[DEBUG] Testing render modes:")
render_modes = ['rgb_array', 'human', 'ansi', 'top_down_rgb_array']
for mode in render_modes:
    try:
        result = env.render(mode=mode)
        print(f"  mode={mode}: result type={type(result)}, ", end="")
        if result is not None and isinstance(result, np.ndarray):
            print(f"shape={result.shape}, dtype={result.dtype}")
        else:
            print(f"result={result}")
    except Exception as e:
        print(f"  mode={mode}: ERROR - {e}")

# Check if engine has screenshot capability
print("\n[DEBUG] Checking engine for screenshot capability:")
if hasattr(env, 'engine'):
    engine = env.engine
    print(f"  engine type: {type(engine)}")
    print(f"  engine attributes: {[attr for attr in dir(engine) if 'screenshot' in attr.lower() or 'image' in attr.lower() or 'render' in attr.lower()]}")

env.close()
print("\n[SUCCESS] Debug complete.")
