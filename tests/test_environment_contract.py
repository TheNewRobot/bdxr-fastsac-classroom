"""Exercise real GPU resets and the classroom robot's action mapping."""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
import unittest
import torch
from bdxr_classroom.environment import make_env, step_and_reset


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class EnvironmentContract(unittest.TestCase):
    def test_full_robot_and_terminal_observation(self):
        env = make_env(4)
        try:
            env.cfg.episode_length_s = 0.04
            obs, _ = env.reset()
            self.assertEqual(env.action_manager.total_action_dim, 10)
            self.assertEqual(obs["actor"].shape, (4, 47))
            self.assertEqual(obs["critic"].shape, (4, 62))
            self.assertIn("Head_Pitch", env.scene["robot"].joint_names)
            self.assertGreater(float(env.sim.mj_model.body_mass.sum()), 19)
            actions = torch.full((4, 10), 0.1, device="cuda")
            result = step_and_reset(env, actions)
            result = step_and_reset(env, actions)
            post_reset, final, _, terminated, truncated, lengths, _ = result
            self.assertTrue(truncated.all())
            self.assertTrue((lengths == 2).all())
            self.assertTrue((env.episode_length_buf == 0).all())
            self.assertFalse(torch.equal(post_reset["actor"], final["actor"]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
