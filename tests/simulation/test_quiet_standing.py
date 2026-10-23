"""Quiet-standing feedback must not reward freezing during commanded motion."""

import types
import unittest

import torch

from bdxr_classroom.simulation.behaviors import quiet_standing_reward


class QuietStandingTests(unittest.TestCase):
    def test_idle_support_and_motion_are_separate(self):
        commands = torch.tensor([[0., 0., 0.], [.3, 0., 0.], [0., 0., .4], [0., 0., 0.]])
        robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_vel=torch.zeros(4, 10)))
        robot.data.joint_vel[3] = 2
        env = types.SimpleNamespace(
            num_envs=4, _quiet_leg_ids=list(range(10)),
            command_manager=types.SimpleNamespace(get_command=lambda _: commands),
            scene={
                "robot": robot,
                "feet_ground_contact": types.SimpleNamespace(
                    data=types.SimpleNamespace(found=torch.tensor([[1, 1], [1, 1], [1, 1], [1, 0]]))
                ),
            },
        )
        reward = quiet_standing_reward(env)
        self.assertEqual(reward[0], 1)
        self.assertEqual(reward[1], 0)
        self.assertEqual(reward[2], 0)
        self.assertLess(reward[3], .01)
        robot.data.joint_vel[0] = .1
        self.assertGreater(quiet_standing_reward(env)[0], .95)
