''' README
    Forward and inverse kinematics for a SCARA robotic arm.
    `pos` is either `[x, y, z]` in cartesian space or `[alpha, beta, z]` in
    actuator space. With `alpha` being the angle of the upper arm to the
    `x` axis and `beta` the angle between upper and lower arm.
    The parameters `a`, `b`, and `c` are length of upper arm, length of lower
    arm, and a scaling factor for the `z` axis respectively.
    `c` maps from cartesian space to the actuator space, so its unit is usually
    `2pi / meter` or the inverse of the pitch of the screw times `2pi`.
    The `limits` is in the form of `[[alpha_min, alpha_max], [beta_min, ...], ...]`.
    Solution with a positive `beta` will always be the fist choice when it
    is valid.
'''
import numpy as np

def forward_kinematics(pos, a, b, c):   # c is in 2pi*rad/m
    alpha = pos[0]
    beta = pos[1]
    z = pos[2]

    x = a * np.cos(alpha) + b * np.cos(alpha + beta)
    y = a * np.sin(alpha) + b * np.sin(alpha + beta)

    return (x, y, c*z)

def inside_limits(v, limits):
    return np.all([dim >= lim[0] and dim <= lim[1] for (dim, lim) in zip(v, limits)])

# limits is an array of [min, max]
def inverse_kinematics(pos, a, b, c, limits):
    x = pos[0]
    y = pos[1]
    z = pos[2]/c

    dist_dist = x**2 + y**2
    dist = np.sqrt(dist_dist)

    # if the position is too far away, go to furthest possible point in the same direction
    if dist > a + b:
        raise ValueError("Position {} out of reach".format(pos))
        x /= dist + np.sqrt(a**2 + b**2)
        y /= dist + np.sqrt(a**2 + b**2)
        dist_dist = x**2 + y**2
        dist = np.sqrt(dist_dist)

    alpha = np.arctan2(y, x) - np.arccos((dist_dist + a**2 - b**2) / (2*a*dist))
    beta = np.arccos((dist_dist - a**2 - b**2) / (2 * a * b))

    if not inside_limits((alpha, beta, z), limits):
        alpha = 2 * np.arctan2(y, x) - alpha
        beta *= -1
        if not inside_limits((alpha, beta, z), limits):
            raise ValueError("Solution for {} violates limits".format(pos))

    return (alpha, beta, z)
