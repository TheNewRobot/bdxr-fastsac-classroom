"""Quiet-standing feedback must not reward freezing during commanded motion."""

import types
import unittest

import torch

from bdxr_classroom.simulation.behaviors import (
    contact_spin_cost, idle_motion_cost, quiet_standing_reward,
)


class QuietStandingTests(unittest.TestCase):
    def test_idle_velocity_excludes_head_and_commanded_motion(self):
        command = torch.tensor([[0., 0., 0.], [.3, 0., 0.], [0., 0., .4]])
        velocity = torch.ones(3, 13)
        velocity[:, 10:] = 100
        env = types.SimpleNamespace(
            _quiet_leg_ids=list(range(10)),
            command_manager=types.SimpleNamespace(get_command=lambda _: command),
            scene={"robot": types.SimpleNamespace(
                data=types.SimpleNamespace(joint_vel=velocity)
            )},
        )
        torch.testing.assert_close(idle_motion_cost(env), torch.tensor([1., 0., 0.]))
        velocity[0, :10] = .1
        torch.testing.assert_close(idle_motion_cost(env), torch.tensor([.1, 0., 0.]))

    def test_spin_cost_applies_only_to_planted_feet(self):
        env = types.SimpleNamespace(
            num_envs=2, _spin_foot_ids=[0, 1],
            scene={
                "robot": types.SimpleNamespace(data=types.SimpleNamespace(
                    site_ang_vel_w=torch.tensor([[[0., 0., .4], [0., 0., .4]],
                                                 [[0., 0., .4], [0., 0., .4]]])
                )),
                "feet_ground_contact": types.SimpleNamespace(
                    data=types.SimpleNamespace(found=torch.tensor([[1, 1], [0, 0]]))
                ),
            },
        )
        torch.testing.assert_close(contact_spin_cost(env), torch.tensor([.32, 0.]))
        env.scene["robot"].data.site_ang_vel_w[:, :, 2] = 10
        torch.testing.assert_close(
            contact_spin_cost(env, max_rate=1.0), torch.tensor([2., 0.])
        )
        torch.testing.assert_close(contact_spin_cost(env), torch.tensor([200., 0.]))
        env.scene["feet_ground_contact"].data.found[:] = 1
        env.command_manager = types.SimpleNamespace(
            get_command=lambda _: torch.tensor([[.3, 0., 0.], [0., 0., .4]])
        )
        torch.testing.assert_close(
            contact_spin_cost(env, max_rate=1., turn_only=True),
            torch.tensor([0., 2.]),
        )

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
