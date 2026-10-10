"""Expression must respect target limits and never address leg actuators."""

import numpy as np
import torch
from types import SimpleNamespace
import unittest

from bdxr_classroom.simulation.head_motion import (
    HeadMotion,
    HeadMotionConfig,
    JOINT_NAMES,
)


class RobotFixture:
    def __init__(self):
        self.data = SimpleNamespace(
            soft_joint_pos_limits=torch.tensor([[[-0.5, 0.5]] * 14])
        )
        self.targets = torch.zeros(1, 14)
        self.writes = 0

    def find_joints(self, names, preserve_order):
        return [10, 11, 12, 13], JOINT_NAMES

    def set_joint_position_target(self, values, joint_ids):
        self.writes += 1
        self.targets[:, joint_ids] = values


class HeadMotionTests(unittest.TestCase):
    def make_head(self, **options):
        robot = RobotFixture()
        robot.targets[:, :10] = 0.25
        env = SimpleNamespace(scene={"robot": robot}, device="cpu", num_envs=1)
        return HeadMotion(env, HeadMotionConfig(**options)), robot

    def test_only_head_targets_change_and_motion_is_bounded(self):
        head, robot = self.make_head(enabled=True)
        positions = []
        for index in range(1500):
            speed = 0.3 if 500 <= index < 1000 else 0
            head.step(0.02, speed, 0)
            positions.append(head.position.copy())
            self.assertTrue(
                np.all(np.abs(head.velocity) <= head.config.max_velocity + 1e-6)
            )
            self.assertTrue(
                np.all(
                    np.abs(head.last_acceleration)
                    <= head.config.max_acceleration + 1e-6
                )
            )
            self.assertTrue(np.all(head.position >= head.lower))
            self.assertTrue(np.all(head.position <= head.upper))
        torch.testing.assert_close(robot.targets[:, :10], torch.full((1, 10), 0.25))
        self.assertGreater(np.ptp(np.array(positions)[:500, 2]), 0.03)
        self.assertLess(abs(positions[999][2]), 0.001)

    def test_neutral_and_reproducible_reset(self):
        head, robot = self.make_head()
        for _ in range(50):
            np.testing.assert_allclose(head.step(0.02, 0.3, 0), np.zeros(4))
        self.assertEqual(robot.writes, 0)
        head, robot = self.make_head(enabled=True)
        for _ in range(50):
            head.step(0.02, 0.3, 0, "hop")
        self.assertEqual(robot.writes, 0)
        head, _ = self.make_head(enabled=True, seed=17)
        first = [head.step(0.02, 0.3, 0) for _ in range(100)]
        head.reset()
        second = [head.step(0.02, 0.3, 0) for _ in range(100)]
        np.testing.assert_allclose(first, second)
        head.time = 3
        np.testing.assert_array_equal(head.desired(0.3, 0, "hop"), np.zeros(4))
