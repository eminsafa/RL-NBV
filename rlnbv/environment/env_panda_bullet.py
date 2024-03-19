from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)

import numpy as np
from gymnasium.utils import seeding

from rlnbv.environment import BaseEnv
from rlnbv.simulation.bullet import BulletSim
from rlnbv.robot.bullet_panda_robot import BulletPanda
# from rlnbv.task.reach_task import Reach
from rlnbv.task.nbv_task import NBV


class PandaBulletEnv(BaseEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str = "human",
        orientation_task: bool = False,
        distance_threshold: float = 0.05,
        goal_range: float = 0.3,
    ) -> None:
        self.sim = BulletSim(render_mode=render_mode, n_substeps=30, orientation_task=orientation_task)
        self.robot = BulletPanda(self.sim, orientation_task=orientation_task)
        self.task = NBV(
            self.sim,
            self.robot,
            reward_type="dense",
            distance_threshold=distance_threshold,
        )
        super().__init__()

        self.render_width = 700
        self.render_height = 400
        self.render_target_position = np.array([0.0, 0.0, 0.72])
        self.render_distance = 2
        self.render_yaw = 45
        self.render_pitch = -30
        self.render_roll = 0
        with self.sim.no_rendering():
            self.sim.place_camera(
                target_position=self.render_target_position,
                distance=self.render_distance,
                yaw=self.render_yaw,
                pitch=self.render_pitch,
            )
        self.last_success_count = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed, options=options)
        self.task.np_random, seed = seeding.np_random(seed)
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        observation = self._get_obs()
        # info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal()[:3])}
        return observation, {}

    def step(self, action: np.ndarray):
        reward = self.task.set_action(action)
        truncated = True if reward == self.task.min_reward else False
        terminated = True if reward > -17 or reward == self.task.max_reward else False
        # print("terminate")
        self.last_success_count = self.task.last_success_count
        return np.array(self._get_obs(), dtype=np.float32), reward, terminated, truncated, {}

    def close(self) -> None:
        self.sim.close()

    def render(self) -> Optional[np.ndarray]:
        """Render."""
        return self.sim.render(
            width=self.render_width,
            height=self.render_height,
            target_position=self.render_target_position,
            distance=self.render_distance,
            yaw=self.render_yaw,
            pitch=self.render_pitch,
            roll=self.render_roll,
        )
