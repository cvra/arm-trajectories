import numpy as np

def generate_velocity_profile(sampling_distance,
                              velocity_limit,
                              acceleration_limit,
                              decceleration_limit=None,
                              v_start=0,
                              v_end=0):

    if decceleration_limit is None:
        decceleration_limit = acceleration_limit

    def required_acceleration(v1, v2, distance):
        ''' calculates the requierd constant acceleration between two points with given
            velocities and distances in between. '''
        return 0.5 * (v2 - v1) * (v1 + v2) / distance

    def velocity_after_distance(v, distance, acceleration):
        ''' calculates the velocity after accelerating constantly on a certain distance. '''
        return np.sqrt(2.0 * distance * acceleration + v**2)

    nb_samples = len(velocity_limit)
    velocity_break_limit = np.zeros(nb_samples)
    velocity_break_limit[-1] = v_end

    for i in reversed(range(1, nb_samples-1)):
        decceleration_vel_limit_to_break_limit = required_acceleration(velocity_break_limit[i+1],
                                                                       velocity_limit[i],
                                                                       sampling_distance)
        if decceleration_vel_limit_to_break_limit > decceleration_limit[i]:
            velocity_break_limit[i] = velocity_after_distance(velocity_break_limit[i+1],
                                                              sampling_distance,
                                                              decceleration_limit[i])
        else:
            velocity_break_limit[i] = velocity_limit[i]

    velocity = np.zeros(nb_samples)
    velocity[0] = v_start
    acceleration = np.zeros(nb_samples)

    for i in range(1, nb_samples):
        acceleration_vel_to_break_limit = required_acceleration(velocity[i-1],
                                                                velocity_break_limit[i],
                                                                sampling_distance)
        acceleration[i] = min(acceleration_limit[i], acceleration_vel_to_break_limit)
        velocity[i] = velocity_after_distance(velocity[i-1],
                                              sampling_distance,
                                              acceleration[i])

    return velocity, acceleration
