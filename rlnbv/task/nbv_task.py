from typing import (
    Any,
    Dict,
    Optional,
    TypeVar,
)

import numpy as np
from rlnbv.utils import (
    distance,
    get_azimuth_and_elevation,
    get_azimuth_and_elevation_for_aim,
)
from rlnbv.simulation import Simulation
from rlnbv.robot import Robot

Sim = TypeVar('Sim', bound=Simulation)
Rob = TypeVar('Rob', bound=Robot)


class NBV:

    def __init__(
        self,
        sim: Sim,
        robot: Rob,
        reward_type: Optional[str] = "dense",
        distance_threshold: Optional[float] = 0.1,
    ) -> None:
        self.sim = sim
        self.robot = robot
        self.goal = None

        # Hemisphere
        self.draw = False
        self.h_count = 32
        self.view_threshold = 0.25

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold

        self.goal_range_low = np.array([0.0, 0.0, 0.05, 0.05])  # position + radius
        self.goal_range_high = np.array([0.9, 0.0, 0.05, 0.1])

        self.raw_hemisphere_positions = None
        self.hemisphere_a_and_e = []
        self.get_raw_hemisphere_positions()
        self.hemisphere_poses = None

        self.last_view_array = None

    def create_scene(self) -> None:
        self.sim.create_scene(self.goal[:3], self.goal[3])

    def reset(self) -> None:
        if self.goal is not None:
            self.sim.remove_body("target")
            if self.draw:
                self.sim.remove_hemisphere_poses(self.h_count)
        self.goal = self._sample_goal()

        with self.sim.no_rendering():
            self.create_scene()

        self.reset_hemisphere_poses()
        self.last_view_array = self.get_view_array()

        if self.draw:
            self.draw_hemisphere()
        self.sim.set_base_pose("target", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))

    def reset_hemisphere_poses(self):
        self.hemisphere_poses = []
        for i in range(self.h_count):
            raw_position = self.raw_hemisphere_positions[i]
            orientation = get_azimuth_and_elevation(raw_position)
            a_orientation = get_azimuth_and_elevation_for_aim(raw_position)
            self.hemisphere_a_and_e.append(a_orientation)
            position = [raw_position[0] + self.goal[0], raw_position[1], raw_position[2]]
            orientation = list(self.sim.physics_client.getQuaternionFromEuler(orientation))
            self.hemisphere_poses.append(position + orientation)

    def get_raw_hemisphere_positions(self):
        self.raw_hemisphere_positions = []
        with open("/Users/eminsafatok/dev/RL-NBV/rlnbv/simulation/bullet/tot_sort.txt", "r") as fin_sphere:
            for i in range(self.h_count):
                raw_position = [float(val) for val in fin_sphere.readline().split()]
                self.raw_hemisphere_positions.append(raw_position)

    def draw_hemisphere(self):
        for idx, i in enumerate(self.hemisphere_poses):
            self.sim.draw_box_and_line(i[:3], i[3:], idx)

    def set_goal(self, goal: np.ndarray):
        self.goal = goal

    def set_action(self, action: np.ndarray) -> int:
        new_position = np.array([
            min(0.8999, max(self.goal[0] + action[0], 0.001)),
            self.goal[1],
            self.goal[2],
        ])
        self.goal = new_position
        self.reset_hemisphere_poses()

        new_view_array = self.get_view_array()
        success_count = self.compare_view_arrays(self.last_view_array, new_view_array)
        self.last_view_array = new_view_array

        return success_count - 32

    def get_view_array(self):
        success_counter = 0
        view_array = []
        for idx, i in enumerate(self.hemisphere_poses):
            position = i[:3]
            self.robot.set_joint_neutral()
            any(self.sim.step() for _ in range(10))

            self.robot.set_joint_neutral()
            orientation = i[3:]
            o = self.sim.physics_client.getEulerFromQuaternion(orientation)
            orientation = self.sim.physics_client.getQuaternionFromEuler([o[0], o[1] + 1.75, o[2]])
            ik = self.robot.calculate_inverse_kinematics(position, orientation)
            self.robot.set_joint_angles(ik)
            for _ in range(100):
                if np.sum(self.robot.get_ee_velocity()) > 0.05:
                    self.sim.step()
                else:
                    break
            position_distance = distance(self.robot.get_ee_position(), np.array(position))
            if position_distance < self.view_threshold:
                success_counter += 1
                view_array.append(True)
            else:
                view_array.append(False)
        return view_array

    def compare_view_arrays(self, last_view_array, new_view_array):
        count = 0
        for old, new in zip(last_view_array, new_view_array):
            if not old and new:
                count += 1

        return count

    @staticmethod
    def get_obs() -> np.ndarray:
        return np.array([])  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.robot.get_ee_position())
        return ee_position

    def _sample_goal(self) -> np.ndarray:
        return np.random.uniform(self.goal_range_low, self.goal_range_high)

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        result = np.array(d < self.distance_threshold, dtype=bool)
        return result

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)

    def get_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            return self.goal.copy()

    def get_reachable_view_poses(self):
        pass

