import sympy as sp
import numpy as np
import scipy as sc
from scipy.integrate import quad
from scipy.optimize import newton

def _parameter_generator():
    a, b, c, d = sp.symbols('a b c d')
    t = sp.symbols('t')

    x = a*t**3 + b*t**2 + c*t + d
    x_dot = sp.diff(x, t)

    z1, z2 = sp.symbols('z_1 z_2')
    z1_dot, z2_dot = sp.symbols('\dot{z_1} \dot{z_2}')

    lin_system = (z1 - x.subs(t, 0),
                  z1_dot - x_dot.subs(t, 0),
                  z2 - x.subs(t, 1),
                  z2_dot - x_dot.subs(t, 1))
    result = sp.solvers.solve(lin_system, a, b, c, d)

    return sp.lambdify((z1, z2, z1_dot, z2_dot),
                       [result[a], result[b], result[c], result[d]],
                       "numpy")

def _spline_generator(parameters):
    return lambda t: (parameters[0]*t**3 + parameters[1]*t**2 +
                      parameters[2]*t + parameters[3])

def _spline_dot_generator(parameters):
    return lambda t: (3 * parameters[0]*t**2 + 2 * parameters[1]*t +
                      parameters[2])

def _spline_dot_dot_generator(parameters):
    return lambda t: (6 * parameters[0]*t + 2 * parameters[1])

generate_spline = _parameter_generator()


class SplineTrajectorySegment():
    def __init__(self, start_pt, end_pt, start_dir, end_dir, roundness=0.1):
        self.start_point = np.array(start_pt)
        self.end_point = np.array(end_pt)
        self.start_dir = np.array(start_dir) / np.linalg.norm(start_dir)
        self.end_dir = np.array(end_dir) / np.linalg.norm(end_dir)

        distance = np.linalg.norm(self.end_point - self.start_point)

        self.start_dir *= roundness * distance
        self.end_dir *= roundness * distance

        self.parameters = [generate_spline(*conditions) for conditions in
                           zip(start_pt, end_pt, self.start_dir, self.end_dir)]

        self._spline = [_spline_generator(p) for p in self.parameters]
        self._spline_dot = [_spline_dot_generator(p) for p in self.parameters]
        self._spline_dot_dot = [_spline_dot_dot_generator(p)
                                for p in self.parameters]
        self.spline_length = self.section_length(0, 1)

    def spline(self, t):
        return [s(t) for s in self._spline]

    def spline_dot(self, t):
        return [s(t) for s in self._spline_dot]

    def section_length(self, a, b):
        ''' Evaluates the length of the spline from a to b. '''
        return quad(lambda t: np.linalg.norm(self.spline_dot(t)), a, b)[0]

    def parametrization_at_distance(self, s):
        return newton(lambda t: self.section_length(0, t) - s, 0.5)


class SplineTrajectoryPoint():
    def __init__(self, spline, distance):
        self.spline = spline
        self.distance = distance
        self.parametrization = spline.parametrization_at_distance(distance)
        self.position = self.spline.spline(self.parametrization)
        self.direction = self.spline.spline_dot(self.parametrization)

    def __repr__(self):
        return str(self.position)


class SplineTrajectory():
    def __init__(self, points, start_dir=None, end_dir=None, roundness=0.1):
        if start_dir is None:
            start_dir = points[1] - points[0]

        if end_dir is None:
            end_dir = points[-1] - points[-2]

        directions = [self.direction_at_point(*pts) for pts
                      in zip(points, points[1:], points[2:])]

        directions = [start_dir] + directions + [end_dir]

        self.splines = [SplineTrajectorySegment(start, end, start_dir, end_dir, roundness)
                        for start, end, start_dir, end_dir
                        in zip(points, points[1:], directions, directions[1:])]

    @staticmethod
    def direction_at_point(point_before, point, point_after):
        v1 = point - point_before
        v2 = point_after - point

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        direction = (v1 / v1_norm**2 + v2 / v2_norm**2)

        return direction / np.linalg.norm(direction)

    def sample_at_distance(self, distance):
        traj_distance = 0
        for spline in self.splines:
            if distance <= traj_distance + spline.spline_length:
                return SplineTrajectoryPoint(spline, distance - traj_distance)
            traj_distance += spline.spline_length

        return SplineTrajectoryPoint(self.splines[-1],
                                     self.splines[-1].spline_length)

    def get_sample_points(self, resolution):
        traj_length = sum([s.spline_length for s in self.splines])
        n = round(traj_length / resolution)
        return [self.sample_at_distance(d) for d in np.linspace(0, traj_length, n)]
