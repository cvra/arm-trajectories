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
