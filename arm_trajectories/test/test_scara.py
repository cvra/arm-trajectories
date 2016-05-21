from arm_trajectories import scara
import unittest


class TestScara(unittest.TestCase):

    def test_invariant(self):
        a = 2
        b = 1
        c = 1
        pos = [2, 0.5, 0]
        joints = scara.inverse_kinematics(pos, a, b, c, [[-1, 1], [-2, 2], [-1, 1]])
        pos_res = scara.forward_kinematics(joints, a, b, c)
        self.assertAlmostEqual(pos[0], pos_res[0])
        self.assertAlmostEqual(pos[1], pos_res[1])
        self.assertAlmostEqual(pos[2], pos_res[2])

    def test_out_of_reach_exception(self):
        a = 2
        b = 1
        c = 1
        pos = [4, 0, 0]
        limits = [[-1, 1], [-2, 2], [-1, 1]]
        with self.assertRaises(ValueError):
            scara.inverse_kinematics(pos, a, b, c, limits)

    def test_limit_exception(self):
        a = 2
        b = 1
        c = 1
        pos = [2, 1, 0]
        limits = [[-0.1, 0.1], [-2, 1], [-1, 1]]
        with self.assertRaises(ValueError):
            print(scara.inverse_kinematics(pos, a, b, c, limits))
