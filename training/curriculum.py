"""
Curriculum Scheduler

Adjusts traffic_density in all parallel environments at predefined
step thresholds according to CURRICULUM_CONFIG.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import CURRICULUM_CONFIG


class CurriculumScheduler:
    """
    Monitors global timesteps and updates traffic density accordingly.

    Usage
    ─────
        scheduler = CurriculumScheduler(vec_env)
        # Inside training loop:
        scheduler.update(current_step)
    """

    def __init__(self, vec_env=None):
        self.vec_env = vec_env
        self.schedule = sorted(CURRICULUM_CONFIG["schedule"], key=lambda x: x[0])
        self._current_density = self.schedule[0][1]
        self._current_phase = 0
        print(f"[Curriculum] Initialized | Phase 0 | traffic_density={self._current_density:.2f}")

    def update(self, global_step: int):
        """Check if a new curriculum phase should be activated."""
        new_phase = 0
        new_density = self.schedule[0][1]

        for phase_idx, (threshold, density) in enumerate(self.schedule):
            if global_step >= threshold:
                new_phase = phase_idx
                new_density = density

        if new_phase != self._current_phase:
            self._current_phase = new_phase
            self._current_density = new_density
            print(
                f"[Curriculum] Phase {new_phase} activated at step {global_step:,} "
                f"| traffic_density → {new_density:.2f}"
            )
            self._apply_density(new_density)

    def _apply_density(self, density: float):
        """Push new traffic density to all environments."""
        if self.vec_env is None:
            return
        try:
            # For SubprocVecEnv: call env method via env_method
            self.vec_env.env_method("set_traffic_density", density)
        except Exception as e:
            print(f"[Curriculum] Warning: could not update traffic density: {e}")

    def get_current_density(self) -> float:
        return self._current_density

    def get_current_phase(self) -> int:
        return self._current_phase
