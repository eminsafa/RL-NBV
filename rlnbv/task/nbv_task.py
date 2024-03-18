from math import sqrt
from random import choice as random_choice
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
        self.max_reward = 0
        self.min_reward = -50

        # Google Colab Check
        try:
            import google.colab
            self.google_colab = True
        except:
            self.google_colab = False

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold

        self.goal_range_low = np.array([0.05, 0.0, 0.05, 0.05])  # position + radius
        self.goal_range_high = np.array([0.9, 0.0, 0.05, 0.15])

        self.raw_hemisphere_positions = None  # RAW DATA FROM TOT FILE
        self.pure_hemisphere_poses = []
        self.get_raw_hemisphere_positions()
        self.init_hemisphere_poses()
        self.hemisphere_poses = None

        self.last_view_array = None
        self.radius_options = self.create_radius_list(0.05, 0.25, 0.01)
        self.max_pos_view_count = 30
        self.min_pos_view_count = 20

    def create_scene(self) -> None:
        self.sim.create_scene(self.goal[:3], self.goal[3])

    def reset(self) -> None:
        if self.goal is not None:
            self.sim.remove_body("target")
            if self.draw:
                self.sim.remove_hemisphere_poses(self.h_count)
        self.goal = self._sample_goal()
        self.goal[3] = random_choice(self.radius_options)

        with self.sim.no_rendering():
            self.create_scene()

        self.reset_hemisphere_poses(self.goal[3])
        self.move_hemisphere_poses(self.goal[0])
        
        if self.draw:
            self.draw_hemisphere()
            
        self.last_view_array, _ = self.get_view_array()

        self.sim.set_base_pose("target", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))
        # print(f"position: {self.goal[:3]}\narray: {self.last_view_array}")

    def init_hemisphere_poses(self):
        """
        Initialize hemisphere
        :return:
        """
        self.pure_hemisphere_poses = []
        for i in range(self.h_count):
            raw_position = self.raw_hemisphere_positions[i]
            orientation = get_azimuth_and_elevation(raw_position)
            position = [raw_position[0], raw_position[1], raw_position[2]]
            orientation = list(self.sim.physics_client.getQuaternionFromEuler(orientation))
            self.pure_hemisphere_poses.append(position + orientation)

    def get_raw_hemisphere_positions(self):
        self.raw_hemisphere_positions = []
        if self.google_colab:
            path = "/content/RL-NBV/rlnbv/task/tot_sort.txt"
        else:
            path = "/home/furkanduman/dev/RL-NBV/rlnbv/task/tot_sort.txt"
        with open(path, "r") as fin_sphere:
            for i in range(self.h_count):
                raw_position = [float(val) for val in fin_sphere.readline().split()]
                self.raw_hemisphere_positions.append(raw_position)

    def draw_hemisphere(self):
        for idx, i in enumerate(self.hemisphere_poses):
            self.sim.draw_box_and_line(i[:3], i[3:], idx)

    def set_goal(self, goal: np.ndarray):
        self.goal = goal

    def set_action(self, action: np.ndarray) -> int:
        # print(f"Action: {action}")
        if not 0.0 < self.goal[0] + action[0] < 0.90:
            return self.min_reward

        self.goal = np.array([
            self.goal[0] + action[0],
            self.goal[1],
            self.goal[2],
            self.goal[3],
        ])
        self.move_hemisphere_poses(action[0])
        self.sim.set_base_pose("target", self.goal[:3], np.array([0.0, 0.0, 0.0, 1.0]))

        if self.draw:
            self.sim.remove_hemisphere_poses(self.h_count)
            self.draw_hemisphere()

        new_view_array, count = self.get_view_array()
        # new_view_array, count = [True for i in range(32)], 32
        success_count = self.compare_view_arrays(self.last_view_array, new_view_array)
        self.last_view_array = new_view_array

        # print(f"Radius {self.goal[3]} - Threshold {self.map_range(self.goal[3])}")
        if count >= self.map_range(self.goal[3]):
            return self.max_reward

        return success_count - 32

    def reset_hemisphere_poses(self, radius: float):
        # self.hemisphere_a_and_e = self.pure_hemisphere_a_and_e
        self.hemisphere_poses = []
        for pure_pose in self.pure_hemisphere_poses:
            self.hemisphere_poses.append(
                self.get_pose_with_radius(pure_pose, radius)
            )

    def move_hemisphere_poses(self, replacement):
        temp_hemisphere_poses = []
        for pose in self.hemisphere_poses:
            temp_pose = pose.copy()
            temp_pose[0] = pose[0] + replacement
            temp_hemisphere_poses.append(temp_pose)
        self.hemisphere_poses = temp_hemisphere_poses

    def get_view_array(self):
        view_array = []
        for idx, i in enumerate(self.hemisphere_poses):
            position = i[:3]
            self.robot.set_joint_neutral()
            any(self.sim.step() for _ in range(2))
            orientation = i[3:]
            o = self.sim.physics_client.getEulerFromQuaternion(orientation)
            orientation = self.sim.physics_client.getQuaternionFromEuler([o[0], o[1] + 1.75, o[2]])
            ik = self.robot.calculate_inverse_kinematics(position, orientation)
            self.robot.set_joint_angles(ik)
            for _ in range(15):
                if np.sum(self.robot.get_ee_velocity()) > 0.05:
                    self.sim.step()
                else:
                    break
            position_distance = distance(self.robot.get_ee_position(), np.array(position))
            if position_distance < self.view_threshold:
                view_array.append(True)
            else:
                view_array.append(False)
        count = sum(1 for item in view_array if item)
        # print("View count: {}".format(count))
        return view_array, count

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

    def get_pose_with_radius(self, pose, radius):
        px, py, pz = pose[:3]
        radius = (radius * 1.5) + 0.25
        px_new = radius * (px / sqrt(pow(px, 2) + pow(py, 2) + pow(pz, 2)))
        py_new = radius * (py / sqrt(pow(px, 2) + pow(py, 2) + pow(pz, 2)))
        pz_new = radius * (pz / sqrt(pow(px, 2) + pow(py, 2) + pow(pz, 2)))
        new_position = [px_new, py_new, pz_new]
        return new_position + pose[3:]

    def map_range(self, value):
        in_range = self.radius_options[-1] - self.radius_options[0]
        out_range = self.min_pos_view_count - self.max_pos_view_count
        value_scaled = float(value - self.radius_options[0]) / float(in_range)
        return self.max_pos_view_count + (value_scaled * out_range)

    def create_radius_list(self, min_radius, max_radius, step):
        return [round(min_radius + i * step, 2) for i in range(int((max_radius - min_radius) / step) + 1)]
