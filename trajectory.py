from spline import *
from velocity_profile import generate_velocity_profile
import numpy as np

def compute_trajectory(spline_trajectory,
                       actuator_limits,
                       resolution,
                       sampling_time,
                       v_start=0,
                       v_end=0,
                       v_limit=0):
    ''' spline_trajectory is a SplineTrajectory
        actuator_limits is a function that returns (min_acc, max_acc) limits for
        (position, velocity)
        resolution is the distance between points in actuator space where acceleration can change
    '''
    points = spline_trajectory.get_sample_points(resolution)

    vel, acc = generate_velocity_profile(points,
                                         resolution,
                                         actuator_limits,
                                         v_start,
                                         v_end,
                                         v_limit)

    trajectory_points = resample_in_time(resolution, vel, acc, sampling_time)

    # evaluate spline at each point to get position, direction, and centripetal acceleration

    return trajectory_points


def limit_in_direction(v, limit_max, limit_min=None):
#TODO handle division by zero
    if limit_min is None:
        limit_min = -limit_max
    if np.linalg.norm(v) > 0:
        factor = min(i for i in np.concatenate((limit_max / v, limit_min / v)) if i > 0)
        return v * factor
    else:
        return limit_max

def velocity_after_distance(v, distance, acceleration):
    ''' calculates the velocity after accelerating constantly on a certain distance. '''
    return np.sqrt(2.0 * distance * acceleration + v**2)


def max_acceleration_along_spline(point, speed, acc_limits, sampling_distance):
    """ translate acc_limits by - point.centripetal_acceleration * speed
        find longest vector in direction of point.direction within acc_limits
    """
    centripetal_acceleration = np.array(point.centripetal_acceleration * speed)
    direction = np.array(point.direction)
    limits_translated = acc_limits - centripetal_acceleration
    max_acceleration = np.linalg.norm(limit_in_direction(direction,
                                      limits_translated[0],
                                      limits_translated[1]))
    if np.vdot(limit_in_direction(point.centripetal_acceleration, acc_limits[0], acc_limits[1])
            - point.centripetal_acceleration
            * velocity_after_distance(speed, sampling_distance, max_acceleration),
            point.centripetal_acceleration) < 0:
        max_acceleration = 0

    return max_acceleration


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

    nb_samples = len(points)
    velocity_limits = np.ones(nb_samples) * velocity_limit
    velocity_limits[-1] = v_end

    for i in reversed(range(1, nb_samples-1)):
        if velocity_limit == 0:
            decceleration_vel_limit_to_break_limit = 0
        else:
            decceleration_vel_limit_to_break_limit = \
                                    required_acceleration(velocity_limits[i+1],
                                                          velocity_limits[i],
                                                          sampling_distance)

        velocity = points[i+1].direction / np.linalg.norm(points[i+1].direction) \
                   * velocity_limits[i+1]
        acc_limits = actuator_acc_limit(points[i+1].position, -velocity)
        decceleration_limit = max_acceleration_along_spline(points[i+1],
                                                            velocity_limits[i+1],
                                                            acc_limits,
                                                            sampling_distance)

        if decceleration_vel_limit_to_break_limit > decceleration_limit:
            velocity_limits[i] = velocity_after_distance(velocity_limits[i+1],
                                                         sampling_distance,
                                                         decceleration_limit)
        else:
            pass #velocity is velocity_limit

    velocity_limits[0] = v_start
    acceleration = np.zeros(nb_samples)

    for i in range(1, nb_samples):
        acceleration_vel_to_break_limit = required_acceleration(velocity_limits[i-1],
                                                                velocity_limits[i],
                                                                sampling_distance)

        velocity = points[i-1].direction / np.linalg.norm(points[i-1].direction) \
                   * velocity_limits[i-1]
        acc_limits = actuator_acc_limit(points[i-1].position, velocity)
        acceleration_limit = max_acceleration_along_spline(points[i-1],
                                                            velocity_limits[i-1],
                                                            acc_limits,
                                                            sampling_distance)

        if acceleration_vel_to_break_limit > acceleration_limit:
            velocity_limits[i] = velocity_after_distance(velocity_limits[i-1],
                                                         sampling_distance,
                                                         acceleration_limit)
            acceleration[i-1] = acceleration_limit
        else:
            acceleration[i-1] = acceleration_vel_to_break_limit

    return velocity_limits, acceleration

def time_for_distance(distance, v, a):
    if a != 0:
        return (np.sqrt(v**2 + 2 * distance * a) - v) / a
    else:
        return distance / v

def list_of_sums(l):
    sums = [0]
    accumulator = 0

    for i in range(len(l)):
        accumulator += l[i]
        sums.append(accumulator)
    return sums

def resample_in_time(sampling_distance, velocities, accelerations, sampling_time):
    """ naive implementation with not always the correct feed-forward velocity
        and acceleration.
    """

    def integrate_position(pos, vel, acc, delta_t):
        delta_distance = vel * delta_t + 0.5 * acc * delta_t**2
        return pos + delta_distance

    def integrate_velocity(vel, acc, delta_t):
        return vel + acc * delta_t


    positions = np.arange(0, len(velocities) * sampling_distance, sampling_distance)
    resampled_points = []

    delta_times = [time_for_distance(sampling_distance, v, a)
                    for v, a in zip(velocities[0:-1], accelerations)]

    times = list_of_sums(delta_times)

    nb_resampled_pts = int(round(times[-1] / sampling_time)) + 1 # add one for the first point

    distance_sampled_index = 0

    for resampled_index in range(nb_resampled_pts - 1): # remove one for the last point
        while (distance_sampled_index + 1 < len(times) and
               resampled_index * sampling_time >= times[distance_sampled_index+1]):
            distance_sampled_index += 1

        # distance_sampled_index points to the point before the point at resampled_index
        delta_t = resampled_index * sampling_time - times[distance_sampled_index]
        resampled_pos = integrate_position(positions[distance_sampled_index],
                                           velocities[distance_sampled_index],
                                           accelerations[distance_sampled_index],
                                           delta_t)
        resampled_vel = integrate_velocity(velocities[distance_sampled_index],
                                           accelerations[distance_sampled_index],
                                           delta_t)
        resampled_points.append((resampled_pos,
                                resampled_vel,
                                accelerations[distance_sampled_index]))

    # append the last point
    resampled_points.append((positions[-1],
                            velocities[-1],
                            accelerations[-1]))

    return resampled_points
