import numpy as np

def forward_kinematics(pos, a, b):
    alpha = pos[0]
    beta = pos[1]
    #z = pos[2]

    x = a * np.cos(alpha) + b * np.cos(alpha + beta)
    y = a * np.sin(alpha) + b * np.sin(alpha + beta)

    return (x, y)

def inside_limits(v, limits):
    return np.all([dim >= lim[0] and dim <= lim[1] for (dim, lim) in zip(v, limits)])

def inverse_kinematics(pos, a, b, limits):
    x = pos[0]
    y = pos[1]
    #z = pos[2]

    dist_dist = x**2 + y**2
    dist = np.sqrt(dist_dist)

    # if the position is too far away, go to furthest possible point in the same direction
    if dist > a + b:
        print("Too long.")
        x /= dist + np.sqrt(a**2 + b**2)
        y /= dist + np.sqrt(a**2 + b**2)
        dist_dist = x**2 + y**2
        dist = np.sqrt(dist_dist)

    alpha = np.arctan2(y, x) - np.arccos((dist_dist + a**2 - b**2) / (2*a*dist))
    beta = np.arccos((dist_dist - a**2 - b**2) / (2 * a * b))

    if not inside_limits((alpha, beta), limits):
        alpha = 2 * np.arctan2(y, x) - alpha
        beta *= -1
        if not inside_limits((alpha, beta), limits):
            print(pos, "impossible")

    return (alpha, beta)
