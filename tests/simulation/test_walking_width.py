import math
import types
import unittest

import torch

from bdxr_classroom.simulation.behaviors import foot_spacing, foot_spacing_reward
from bdxr_classroom.simulation.environment import make_env


class WalkingWidthTests(unittest.TestCase):
    def test_width_is_in_body_frame_and_ignores_forward_offset(self):
        # Same foot placement with a 90-degree world yaw.
        positions = torch.tensor(
            [[[0.1, 0.1, 0.0], [-0.1, -0.1, 0.0]], [[-0.1, 0.1, 0.0], [0.1, -0.1, 0.0]]]
        )
        quats = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]]
        )
        robot = types.SimpleNamespace(
            data=types.SimpleNamespace(site_pos_w=positions, root_link_quat_w=quats),
            find_sites=lambda names: ([0, 1], names),
        )
        env = types.SimpleNamespace(num_envs=2, scene={"robot": robot})
        torch.testing.assert_close(foot_spacing(env), torch.full((2,), 0.2))
        torch.testing.assert_close(foot_spacing_reward(env, 0.2), torch.ones(2))

    def test_width_reward_preserves_actor_contract(self):
        env = make_env(4, walking_width=0.2, yaw_tracking="imu")
        try:
            obs, _ = env.reset(seed=7)
            self.assertEqual(obs["actor"].shape, (4, 47))
            self.assertNotIn("walking_width", env.cfg.observations["actor"].terms)
            self.assertNotIn("base_lin_vel", env.cfg.observations["actor"].terms)
            for _ in range(3):
                _, reward, _, _, _ = env.step(torch.zeros(4, 10, device=env.device))
                self.assertTrue(torch.isfinite(reward).all())
        finally:
            env.close()
