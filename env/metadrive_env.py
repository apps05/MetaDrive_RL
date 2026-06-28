"""
MetaDrive Agent Environment

Custom environment with advanced reward function for:
- Speed control (target 60 km/h)
- Lane keeping and smooth turning
- Traffic awareness and overtaking
- Safety and progress tracking
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import ENV_CONFIG, REWARD_CONFIG

try:
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.component.sensors.lidar import Lidar
    _METADRIVE_AVAILABLE = True
except ImportError:
    _METADRIVE_AVAILABLE = False
    MetaDriveEnv = object


class MetaDriveAgentEnv(MetaDriveEnv):
    """
    MetaDriveEnv subclass with advanced reward shaping.
    
    Features:
    - Speed targeting (60 km/h)
    - Lane keeping rewards
    - Turn quality feedback
    - Traffic interaction rewards
    - Overtaking bonus
    - Curriculum-based traffic density
    """

    def __init__(self, config: dict | None = None, traffic_density: float = 0.0):
        merged = {**ENV_CONFIG, **(config or {})}
        merged["traffic_density"] = traffic_density
        super().__init__(merged)

        # State tracking
        self._prev_dist_to_dest: float = 0.0
        self.last_action = np.array([0.0, 0.0])
        self.last_route_completion = 0.0
        self._last_heading = 0.0
        self._last_vehicle_count = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Traffic and Overtaking Detection
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_overtake_opportunity(self, ego, step_info):
        """Detect if agent is overtaking another vehicle."""
        try:
            nearby_vehicles = step_info.get("nearby_vehicles", 0)
            if nearby_vehicles < self._last_vehicle_count:
                return 1.0  # Successfully reduced vehicle count (overtook)
            self._last_vehicle_count = nearby_vehicles
        except Exception:
            pass
        return 0.0

    def _calc_traffic_distance_reward(self, ego, step_info):
        """Reward maintaining safe distance from traffic."""
        try:
            lane = ego.navigation.current_lane
            lon, lat = lane.local_coordinates(ego.position)
            return abs(lat) * 0.1  # Reward lateral separation
        except Exception:
            return 0.0

    def _calc_turn_quality(self, ego):
        """Reward smooth, controlled turns."""
        try:
            current_heading = ego.heading_theta
            heading_delta = abs((current_heading - self._last_heading + np.pi) % (2 * np.pi) - np.pi)
            turn_quality = math.exp(-heading_delta * 2.0)
            self._last_heading = current_heading
            return turn_quality
        except Exception:
            return 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Reward Function
    # ──────────────────────────────────────────────────────────────────────────

    def reward(self, action, step_info=None):
        """
        Composite reward:
            R_total = R_speed + R_route + R_lat + R_dir + R_smooth + R_yaw 
                    + R_border + R_turn + R_traffic + R_overtake + R_stab - C_penalty
        """
        if step_info is None:
            step_info = {}
            
        ego = self.vehicle
        reward_val = 0.0
        terminated = False
        truncated = False
        info = {}
        rc = REWARD_CONFIG

        # 1. Speed Control
        v = ego.speed_km_h
        v_target = rc["v_target"]
        r_speed = rc["c_speed"] * (1.0 - abs(v - v_target) / v_target)
        if v > v_target:
            r_speed -= 0.3 * (v - v_target)

        # 2. Route Progress
        try:
            route_completion = step_info.get("route_completion", self.last_route_completion)
            progress = route_completion - self.last_route_completion
            r_route = rc["c_route"] * progress * 100.0
            self.last_route_completion = route_completion
        except Exception:
            r_route = 0.0

        # 3. Lane Keeping
        r_lat = 0.0
        r_border = 0.0
        try:
            lane = ego.navigation.current_lane
            lon, lat = lane.local_coordinates(ego.position)
            w = lane.width
            r_lat = -rc["c_lat"] * abs(lat) / max(w, 1e-3)
            r_border = -rc["c_border"] * (abs(lat) / max(w, 1e-3))**2
        except Exception:
            pass

        # 4. Direction Alignment
        r_dir = 0.0
        try:
            lane = ego.navigation.current_lane
            lane_heading = lane.heading_theta_at(lon)
            theta_diff = (ego.heading_theta - lane_heading + np.pi) % (2 * np.pi) - np.pi
            r_dir = -rc["c_dir"] * abs(theta_diff)
        except Exception:
            pass

        # 5. Smooth Driving
        r_smooth = -rc["c_smooth"] * (abs(action[0] - self.last_action[0]) + abs(action[1] - self.last_action[1]))

        # 6. Yaw Rate Penalty
        current_heading = ego.heading_theta
        yaw_rate = (current_heading - self._last_heading + np.pi) % (2 * np.pi) - np.pi
        r_yaw = -rc["c_yaw"] * abs(yaw_rate) * 10.0
        self._last_heading = current_heading

        # 7. Turn Quality
        r_turn = rc["c_turn"] * self._calc_turn_quality(ego)

        # 8. Traffic Distance
        r_traffic = rc["c_traffic_dist"] * self._calc_traffic_distance_reward(ego, step_info)

        # 9. Overtaking Bonus
        r_overtake = rc["c_overtake"] * self._detect_overtake_opportunity(ego, step_info)

        # 10. Stability
        r_stab = rc["c_stab"] * (1.0 - min(abs(ego.heading_theta - self._last_heading), 0.5))

        # 11. Penalties & Termination
        c_penalty = 0.0

        if ego.crash_vehicle or ego.crash_object:
            c_penalty += rc["collision_penalty"]
            info["crash"] = True

        if step_info.get("out_of_road", False):
            c_penalty += rc["off_road_penalty"]
            terminated = True
            info["out_of_road"] = True

        c_penalty += rc["time_penalty"]

        # Immobility penalty
        if v < 0.5:
            c_penalty += 1.0

        # Success bonus
        if step_info.get("arrive_dest", False):
            reward_val += ENV_CONFIG["success_reward"]
            truncated = True
            info["success"] = True

        # Composite total
        reward_val += r_speed + r_route + r_lat + r_dir + r_smooth + r_yaw + r_border + r_turn + r_traffic + r_overtake + r_stab - c_penalty
        
        info.update({
            "r_speed": r_speed,
            "r_route": r_route,
            "r_lat": r_lat,
            "r_dir": r_dir,
            "r_smooth": r_smooth,
            "r_yaw": r_yaw,
            "r_border": r_border,
            "r_turn": r_turn,
            "r_traffic": r_traffic,
            "r_overtake": r_overtake,
            "r_stab": r_stab,
            "c_penalty": c_penalty,
        })

        return reward_val, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # Step Override
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, action):
        obs, reward_meta, terminated, truncated, info = super().step(action)
        custom_reward, c_term, c_trunc, c_info = self.reward(action, info)
        info.update(c_info)
        terminated = terminated or c_term
        truncated = truncated or c_trunc
        self.last_action = action
        return obs, custom_reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.last_action = np.array([0.0, 0.0])
        self.last_route_completion = info.get("route_completion", 0.0)
        self._last_heading = 0.0
        self._last_vehicle_count = 0
        try:
            ego = self.vehicle
            self._prev_dist_to_dest = ego.navigation.get_checkpoints_num() * 10.0
        except Exception:
            self._prev_dist_to_dest = 0.0
        return obs, info

    # ──────────────────────────────────────────────────────────────────────────
    # Curriculum: Traffic Density Update
    # ──────────────────────────────────────────────────────────────────────────

    def set_traffic_density(self, density: float):
        """Update traffic density at runtime."""
        self.config["traffic_density"] = float(density)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
    os.environ.setdefault("DISPLAY", ":99")

    print("[TEST] Creating MetaDriveAgentEnv ...")
    env = MetaDriveAgentEnv(traffic_density=0.0)
    obs, info = env.reset()
    print(f"  Observation shape : {obs.shape}  (expected ~259)")
    print(f"  Action space      : {env.action_space}")

    total_reward = 0.0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print(f"  20 steps OK — total reward: {total_reward:.4f}")
    print("[TEST] PASSED ✓")
