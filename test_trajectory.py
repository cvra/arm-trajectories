import unittest
import spline
import trajectory
import numpy as np
import numpy.testing as npt

class TestLimitInDirection(unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_trivial_case(self):
        direction = np.array([1, 0])
        limit = np.array([5, 5])

        npt.assert_almost_equal([5, 0], trajectory.limit_in_direction(direction, limit))

    def test_negative_direction(self):
        direction = np.array([-1, 0])
        limit = np.array([5, 5])

        npt.assert_almost_equal([-5, 0], trajectory.limit_in_direction(direction, limit))

    def test_with_limit_min(self):
        direction = np.array([-2, 0])
        limit_max = np.array([5, 5])
        limit_min = np.array([-3, -3])

        npt.assert_almost_equal([-3, 0],
                trajectory.limit_in_direction(direction, limit_max, limit_min))

    def test_with_limit_min_2(self):
        direction = np.array([-1, 1])
        limit_max = np.array([5, 5])
        limit_min = np.array([-3, -3])

        npt.assert_almost_equal([-3, 3],
                trajectory.limit_in_direction(direction, limit_max, limit_min))

    def test_commutativity_of_min_and_max(self):
        direction = np.array([-1, 1])
        limit_max = np.array([5, 5])
        limit_min = np.array([-3, -3])

        npt.assert_almost_equal([-3, 3],
                trajectory.limit_in_direction(direction, limit_min, limit_max))


class TestMaxAccelerationAlongSpline(unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        p1 = [0, 0]
        p2 = [1, 0]
        p3 = [1, 1]
        pts = [p1, p2, p3]

        self.trajectory = spline.SplineTrajectory(pts, [1, 0], [0, 1], roundness=0.5)

    def tearDown(self):
        pass

    def test_trivial_case(self):
        point = self.trajectory.sample_at_distance(0)
        velocity = 0
        acc_limits = ([-1, -1], [1, 1])

        max_acceleration = trajectory.max_acceleration_along_spline(point,
                                                                    velocity,
                                                                    acc_limits)
        self.assertEqual(1, max_acceleration)

    def test_with_independent_centripetal_acceleration(self):
        point = self.trajectory.sample_at_distance(0)
        velocity = 10
        acc_limits = ([-1, -1], [1, 1])

        max_acceleration = trajectory.max_acceleration_along_spline(point,
                                                                    velocity,
                                                                    acc_limits)
        self.assertAlmostEqual(1, max_acceleration)

    def test_with_centripetal_acceleration(self):
        point = self.trajectory.sample_at_distance(1.01211)
        velocity = 1
        acc_limits = ([-1, -1], [1, 1])
        print(point.direction)

        max_acceleration = trajectory.max_acceleration_along_spline(point,
                                                                    velocity,
                                                                    acc_limits)
        self.assertGreater(1, max_acceleration)

