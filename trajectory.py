from spline import *
from velocity_profile import generate_velocity_profile
import numpy as np

def compute_trajectory(spline_trajectory, actuator_limits, resolution, sampling_time):
    ''' spline_trajectory is a SplineTrajectory
        actuator_limits is a function that returns (vel, acc) limits for a position
        resolution is the distance between points in actuator space where acceleration can change
    '''
    points = spline_trajectory.get_sample_points(resolution)
    actuator_vel_limits, actuator_acc_limits = zip([actuator_limits(p.position) for p in points])

    velocity_limit = [np.linalg.norm(p.direction * min(limit / p.direction))
                      for p, limit in zip(points, actuator_vel_limits)]


def limit_in_direction(v, limit_max, limit_min=None):
    if limit_min is None:
        limit_min = -limit_max
    factor = min(i for i in np.concatenate((limit_max / v, limit_min / v)) if i > 0)
    return v * factor

def max_acceleration_along_spline(point, velocity, acc_limits):
    """ translate acc_limits by - point.centripetal_acceleration * velocity
        find longest vector in direction of point.direction within acc_limits
    """
    centripetal_acceleration = np.array(point.centripetal_acceleration * velocity)
    direction = np.array(point.direction)
    limits_translated = acc_limits - centripetal_acceleration
    return np.linalg.norm(limit_in_direction(direction, limits_translated[0], limits_translated[1]))
