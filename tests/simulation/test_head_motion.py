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
    def test_visible_walk_bounce_idle_glances_and_smooth_handover(self):
        head, _ = self.make_head(enabled=True, level_gaze_in_render=True)
        moving, idle, poses = [], [], []
        for index in range(1400):
            speed = 0.3 if index < 500 else 0
            mode = "hop" if index >= 1300 else "walk"
            head.step(0.02, speed, 0, mode)
            pose = head.display_offsets()
            poses.append(list(pose.values()))
            if 100 < index < 500:
                moving.append(pose)
            if 650 < index < 1300:
                idle.append(pose)
        self.assertGreater(np.ptp([p["neck"] for p in moving]), 0.25)
        self.assertLess(max(abs(p["yaw"]) for p in moving), 0.001)
        self.assertGreater(np.ptp([p["yaw"] for p in idle]), 0.35)
        self.assertLess(np.abs(np.diff(np.array(poses), axis=0)).max(), 0.04)

    def test_render_option_preserves_physical_target_sequence(self):
        ordinary, _ = self.make_head(enabled=True)
        visual, _ = self.make_head(enabled=True, level_gaze_in_render=True)
        for mode, speed, yaw in [("walk", .3, 0), ("walk", 0, 0), ("hop", .3, 0)]:
            for _ in range(300):
                np.testing.assert_array_equal(
                    ordinary.step(.02, speed, yaw, mode),
                    visual.step(.02, speed, yaw, mode),
                )
        cosmetic, robot = self.make_head(level_gaze_in_render=True)
        cosmetic.step(.02, .3, 0)
        self.assertEqual(robot.writes, 0)

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
