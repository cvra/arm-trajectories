import spline
import unittest
import numpy.testing as npt
import numpy as np

def normalize(v):
    return np.array(v) / np.linalg.norm(v)

class SplineTest(unittest.TestCase):

    def test_start_and_end_postions_1D(self):
        trajectory = spline.SplineTrajectorySegment([0], [1], [1], [1])
        npt.assert_almost_equal([0], trajectory.spline(0))
        npt.assert_almost_equal([1], trajectory.spline(1))

    def test_start_and_end_postions_3D(self):
        trajectory = spline.SplineTrajectorySegment([0, 1, 2], [-1, -2, -3], [1, 1, 1], [1, 1, 1])
        npt.assert_almost_equal([0, 1, 2], trajectory.spline(0))
        npt.assert_almost_equal([-1, -2, -3], trajectory.spline(1))

    def test_start_and_end_derivatives_3D(self):
        d1 = [1, 1, 1]
        d2 = [1, 2, 3]
        trajectory = spline.SplineTrajectorySegment([0, 1, 2], [-1, -2, -3], d1, d2)
        npt.assert_almost_equal(normalize(d1), normalize(trajectory.spline_dot(0)))
        npt.assert_almost_equal(normalize(d2), normalize(trajectory.spline_dot(1)))

    def test_section_length(self):
        trajectory = spline.SplineTrajectorySegment([0, 1, 2], [-1, -2, -3], [1, 1, 1], [1, 1, 1])
        sa = trajectory.section_length(0, 0.5)
        sb = trajectory.section_length(0.5, 1)
        sab = trajectory.section_length(0, 1)
        self.assertAlmostEqual(sa + sb, sab)

    def test_parametrization_at_distance(self):
        trajectory = spline.SplineTrajectorySegment([0, 1, 2], [-1, -2, -3], [1, 1, 1], [1, 1, 1])
        r = trajectory.parametrization_at_distance(trajectory.spline_length / 3)
        length = trajectory.section_length(0, r)
        self.assertAlmostEqual(trajectory.spline_length / 3, length)


class SplineTrajectoryTest(unittest.TestCase):
    def test_direciton_at_point(self):
        p1 = [-1, -1]
        p2 = [0, 0]
        p3 = [1, -1]
        expected_dir = np.array([1, 0])
        computed_dir = spline.SplineTrajectory.direction_at_point(p1, p2, p3)

        npt.assert_almost_equal(expected_dir, computed_dir)

    def test_spline_segments(self):
        p1 = [0, 0]
        p2 = [21, 42]
        p3 = [88, 88]

        pts = [p1, p2, p3]

        p1_dir = np.array(p2) - np.array(p1)
        p2_dir = spline.SplineTrajectory.direction_at_point(p1, p2, p3)
        p3_dir = np.array(p3) - np.array(p2)

        trajectory = spline.SplineTrajectory(pts)

        npt.assert_almost_equal(p1, trajectory.splines[0].spline(0))
        npt.assert_almost_equal(normalize(p1_dir), normalize(trajectory.splines[0].spline_dot(0)))
        npt.assert_almost_equal(p2, trajectory.splines[0].spline(1))
        npt.assert_almost_equal(p2, trajectory.splines[1].spline(0))
        npt.assert_almost_equal(normalize(p2_dir), normalize(trajectory.splines[0].spline_dot(1)))
        npt.assert_almost_equal(normalize(p2_dir), normalize(trajectory.splines[1].spline_dot(0)))
        npt.assert_almost_equal(p3, trajectory.splines[1].spline(1))
        npt.assert_almost_equal(normalize(p3_dir), normalize(trajectory.splines[1].spline_dot(1)))
