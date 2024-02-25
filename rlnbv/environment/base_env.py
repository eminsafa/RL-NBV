from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BaseEnv(gym.Env):
    robot = None
    sim = None
    task = None

    def __init__(self) -> None:
        observation, _ = self.reset()

        observation_shape = observation.shape
        self.observation_space = spaces.Box(low=np.array([-0.0, 0.0]), high=np.array([1.0, .5]), shape=observation_shape, dtype=np.float32)

        self.action_space = self.robot.action_space
        self.compute_reward = self.task.compute_reward
        # self._saved_goal = dict()

    def _get_obs(self) -> np.ndarray:
        return np.array([self.task.goal[0], self.task.goal[3]])

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        return NotImplemented

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        return NotImplemented

    def close(self):
        return NotImplemented

    def render(self) -> Optional[np.ndarray]:
        return NotImplemented
