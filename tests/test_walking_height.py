import types
import unittest
from unittest.mock import patch

import torch

from bdxr_classroom.behaviors import walking_height_reward
from bdxr_classroom.environment import make_env


class WalkingHeightTests(unittest.TestCase):
    def test_target_height_is_relative_to_ground(self):
        position = torch.tensor([[0., 0., .23], [0., 0., .28], [0., 0., .33]])
        env = types.SimpleNamespace(scene={
            "robot": types.SimpleNamespace(data=types.SimpleNamespace(root_link_pos_w=position))
        })
        with patch("bdxr_classroom.behaviors.ground_height",
                   return_value=torch.tensor([0., 0., .10])):
            reward = walking_height_reward(env, .23)
        torch.testing.assert_close(reward[[0, 2]], torch.ones(2))
        self.assertLess(float(reward[1]), .2)

    def test_height_reward_does_not_add_actor_observations(self):
        env = make_env(4, walking_height=.23, yaw_tracking="imu")
        try:
            obs, _ = env.reset(seed=7)
            self.assertEqual(obs["actor"].shape, (4, 47))
            self.assertNotIn("walking_height", env.cfg.observations["actor"].terms)
            self.assertNotIn("base_lin_vel", env.cfg.observations["actor"].terms)
            self.assertEqual(env.cfg.rewards["walking_height"].params["target_height"], .23)
            for _ in range(3):
                _, reward, _, _, _ = env.step(torch.zeros(4, 10, device=env.device))
                self.assertTrue(torch.isfinite(reward).all())
        finally:
            env.close()
