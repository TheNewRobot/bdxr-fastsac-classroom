"""History-based reward state must reset independently for each episode."""

from types import SimpleNamespace
import unittest
import torch
from bdxr_classroom.behaviors import yaw_bias_tracking


class RewardHistory(unittest.TestCase):
    def test_reset_clears_only_reset_environment(self):
        velocity = torch.tensor([[0., 0., .1], [0., 0., .1]])
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=velocity))},
            command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(2, 3)),
            episode_length_buf=torch.tensor([100, 100]), step_dt=.02)
        for _ in range(50):
            yaw_bias_tracking(env)
        velocity.zero_()
        env.episode_length_buf[0] = 1
        reward = yaw_bias_tracking(env)
        self.assertEqual(float(reward[0]), 1)
        self.assertLess(float(reward[1]), .001)
        self.assertEqual(float(env._yaw_bias_error[0]), 0)
        self.assertGreater(float(env._yaw_bias_error[1]), .08)
