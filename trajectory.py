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

def generate_velocity_profile(points,
                              sampling_distance,
                              actuator_acc_limit,
                              v_start=0,
                              v_end=0,
                              velocity_limit=0):
    """TODO: Docstring for generate_velocity_profile.

    :points: array of spline points
    :sampling_distance: constant distance between points
    :actuator_acc_limit: returns min/max acceleration for each actuator in function of pos & vel
    :v_start: velocity at first point
    :v_end: velocity at last point
    :velocity_limit: overall maximal velocity (unlimited when zero)
    :returns: TODO

    """

    def required_acceleration(v1, v2, distance):
        ''' calculates the requierd constant acceleration between two points with given
            velocities and distances in between. '''
        return 0.5 * (v2 - v1) * (v1 + v2) / distance

    def velocity_after_distance(v, distance, acceleration):
        ''' calculates the velocity after accelerating constantly on a certain distance. '''
        return np.sqrt(2.0 * distance * acceleration + v**2)

    nb_samples = len(points)
    velocity_limits = np.ones(nb_samples) * velocity_limit
    velocity_limits[-1] = v_end

    for i in reversed(range(1, nb_samples-1)):
        if velocity_limit == 0:
            decceleration_vel_limit_to_break_limit = 0
        else:
            decceleration_vel_limit_to_break_limit = required_acceleration(velocity_limits[i+1],
                                                                           velocity_limits[i],
                                                                           sampling_distance)

        velocity = points[i+1].direction / np.linalg.norm(points[i+1].direction) * velocity_limits[i+1]
        acc_limits = actuator_acc_limit(points[i+1].position, -velocity)
        decceleration_limit = max_acceleration_along_spline(points[i+1].position,
                                                            velocity_limits[i+1],
                                                            acc_limits)

        if decceleration_vel_limit_to_break_limit > decceleration_limit:
            velocity_limits[i] = velocity_after_distance(velocity_limits[i+1],
                                                         sampling_distance,
                                                         decceleration_limit)
        else:
            pass #velocity is velocity_limit

    velocity_limits[0] = v_start
    acceleration = np.zeros(nb_samples)

    for i in range(1, nb_samples):
        if velocity_limit == 0:
            acceleration_vel_to_break_limit = 0
        else:
            acceleration_vel_to_break_limit = required_acceleration(velocity_limits[i-1],
                                                                    velocity_limits[i],
                                                                    sampling_distance)

        velocity = points[i-1].direction / np.linalg.norm(points[i-1].direction) * velocity_limits[i-1]
        acc_limits = actuator_acc_limit(points[i-1].position, velocity)
        acceleration_limit = max_acceleration_along_spline(points[i-1].position,
                                                            velocity_limits[i-1],
                                                            acc_limits)

        if acceleration_vel_to_break_limit > acceleration_limit:
            velocity_limits[i] = velocity_after_distance(velocity_limits[i-1],
                                                         sampling_distance,
                                                         acceleration_limit)
            acceleration[i-1] = acceleration_limit
        else:
            acceleration[i-1] = acceleration_vel_to_break_limit

    #TODO resample in time

    return velocity_limits, acceleration
