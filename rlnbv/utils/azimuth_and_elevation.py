import math

import numpy as np
from numpy.linalg import norm
from math import atan2, asin, degrees


def get_azimuth_and_elevation(position):
    direction_vector = np.array([0, 0, 0]) - np.array(position)
    azimuth = degrees(atan2(direction_vector[1], direction_vector[0]))
    elevation = degrees(asin(direction_vector[2] / np.linalg.norm(direction_vector)))
    return np.radians([0.0, -elevation, azimuth])


def get_azimuth_and_elevation_for_aim(position):
    direction_vector = np.array(position) - np.array([0, 0, 0])
    x, y, z = position
    yaw = math.atan2(y, x)
    pitch = math.atan2(-z, math.sqrt(x**2 + y**2))
    return np.array([0.0, pitch+1.57, yaw])
    # return vector_to_quaternion(position[0], position[1], position[2])


def vector_to_quaternion(x, y, z):
    v = np.array([x, y, z])
    v_norm = norm(v)
    if v_norm == 0:
        return None  # Can't compute a direction for a zero vector
    v_normalized = v / v_norm
    target = np.array([-1, 1, 0])
    axis = np.cross(v_normalized, target)
    angle = np.arccos(np.dot(v_normalized, target))
    w = np.cos(angle / 2)
    axis_normalized = axis / norm(axis) if norm(axis) != 0 else (0, 0, 0)
    xyz = np.sin(angle / 2) * axis_normalized

    return np.array([xyz[0], xyz[1], xyz[2], w])
