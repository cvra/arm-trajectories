import unittest
from arm_trajectories import spline
from arm_trajectories import trajectory
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
        limit_max = np.array([3, 3])
        limit_min = np.array([-5, -5])

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
                                                                    acc_limits,
                                                                    1)
        self.assertEqual(1, max_acceleration)

    def test_with_independent_centripetal_acceleration(self):
        point = self.trajectory.sample_at_distance(0)
        velocity = 10
        acc_limits = ([-1, -1], [1, 1])

        max_acceleration = trajectory.max_acceleration_along_spline(point,
                                                                    velocity,
                                                                    acc_limits,
                                                                    1)
        #self.assertAlmostEqual(1, max_acceleration)

    def test_with_centripetal_acceleration(self):
        point = self.trajectory.sample_at_distance(1.01211)
        velocity = 1
        acc_limits = ([-1, -1], [1, 1])
        print(point.direction)

        max_acceleration = trajectory.max_acceleration_along_spline(point,
                                                                    velocity,
                                                                    acc_limits,
                                                                    1)
        self.assertGreater(1, max_acceleration)

class TestResampleInTime(unittest.TestCase):

    """Test case docstring."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_time_for_distance(self):
        distance = 0.5
        velocity = 0
        acceleration = 1

        self.assertAlmostEqual(1, trajectory.time_for_distance(distance, velocity, acceleration))

    def test_time_for_distance_decceleration(self):
        distance = 0.5
        velocity = 1
        acceleration = -1

        self.assertAlmostEqual(1, trajectory.time_for_distance(distance, velocity, acceleration))


    def test_time_for_distance_without_accelreation(self):
        distance = 1
        velocity = 1
        acceleration = 0

        self.assertAlmostEqual(1, trajectory.time_for_distance(distance, velocity, acceleration))

    def test_list_of_sums(self):
        l = [1, 2, 3]
        self.assertEqual([0, 1, 3, 6], trajectory.list_of_sums(l))

    def test_two_point_trajectory(self):
        sampling_distance = 1/2
        velocities = [0, 1]
        accelerations = [1, 0]
        sampling_time = 1

        self.assertEqual([(0, 0, 1), (0.5, 1, 0)],
                         trajectory.resample_in_time(sampling_distance,
                                                     velocities,
                                                     accelerations,
                                                     sampling_time))

    def test_ramp_trajectory(self):
        sampling_distance = 1/2
        velocities =    [0, 1, 1, 1, 1, 1, 0]
        accelerations = [1, 0, 0, 0, 0, -1, 0]
        sampling_time = 1

        resampled_points = [(0, 0, 1), (0.5, 1, 0), (1.5, 1, 0), (2.5, 1, -1), (3, 0, 0)]
        self.assertEqual(resampled_points,
                         trajectory.resample_in_time(sampling_distance,
                                                     velocities,
                                                     accelerations,
                                                     sampling_time))
