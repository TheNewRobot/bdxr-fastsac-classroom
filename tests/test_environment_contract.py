"""Exercise real GPU resets and the classroom robot's action mapping."""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
import unittest
import torch
from bdxr_classroom.environment import make_env, step_and_reset


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class EnvironmentContract(unittest.TestCase):
    def test_command_evaluation_continues_after_one_trial_falls(self):
        from bdxr_classroom.command_evaluate import run_case
        env = make_env(2, command_mode="balanced")

        class TestController:
            injected = False

            def act(self, obs, deterministic=False):
                if not self.injected:
                    # Explicit test-only fall in world 1; world 0 stays upright.
                    state = torch.tensor(
                        [[0, 0, .1, .7071068, .7071068, 0, 0, 0, 0, 0, 0, 0, 0]], device="cuda")
                    env.scene["robot"].write_root_state_to_sim(
                        state, env_ids=torch.tensor([1], device="cuda"))
                    self.injected = True
                return torch.zeros(2, 10, device="cuda")

        try:
            reports, rows = run_case(env, TestController(), "stop", .2)
            self.assertFalse(reports[0]["fell"])
            self.assertAlmostEqual(reports[0]["duration_s"], .2)
            self.assertTrue(reports[1]["fell"])
            self.assertLess(reports[1]["duration_s"], .2)
            self.assertEqual(sum(row["fell"] for row in rows if row["episode"] == 1), 1)
        finally:
            env.close()

    def test_mixed_terrain_reward_height_matches_physics(self):
        import mujoco
        import numpy as np
        from bdxr_classroom.training_terrain import ground_height
        env = make_env(1, task_version="hop", training_terrain_height=.035,
                       training_terrain_style="mixed")
        try:
            model = env.sim.mj_model
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            for x in (.65, .66, .7, 1.25):
                distance = mujoco.mj_ray(
                    model, data, np.array([x, 0., 1.]), np.array([0., 0., -1.]),
                    None, True, -1, np.array([-1], dtype=np.int32))
                predicted = ground_height(
                    env, torch.tensor([[x, 0., 0.]], device="cuda")).item()
                self.assertAlmostEqual(predicted, 1 - distance, places=6)
        finally:
            env.close()

    def test_practice_terrain_preserves_actor_contract(self):
        from bdxr_classroom.training_terrain import ground_height
        env = make_env(4, task_version="hop", training_terrain_height=.015)
        try:
            obs, _ = env.reset()
            self.assertEqual(obs["actor"].shape, (4, 49))
            self.assertEqual(obs["critic"].shape, (4, 64))
            self.assertNotIn("foot_height", env.cfg.observations["actor"].terms)
            self.assertIn("foot_height", env.cfg.observations["critic"].terms)
            self.assertEqual(float(env.scene.env_origins.abs().max()), 0)
            points = torch.tensor([[[0., 0., 0.], [.65, 0., 0.],
                                    [1.25, 0., 0.], [.65, 5., 0.]]], device="cuda")
            torch.testing.assert_close(
                ground_height(env, points),
                torch.tensor([[0, .015, .012, 0]], device="cuda"))
            result = step_and_reset(env, torch.zeros((4, 10), device="cuda"))
            self.assertTrue(torch.isfinite(result[2]).all())
        finally:
            env.close()

    def test_balanced_command_coverage(self):
        env = make_env(128, command_mode="balanced", seed=123)
        try:
            env.reset(seed=123)
            commands = env.command_manager.get_command("twist")
            x, yaw = commands[:, 0], commands[:, 2]
            self.assertTrue(((x == 0) & (yaw == 0)).any())
            self.assertTrue(((x != 0) & (yaw == 0)).any())
            self.assertTrue(((x == 0) & (yaw != 0)).any())
            self.assertTrue(((x != 0) & (yaw != 0)).any())
            self.assertTrue((x < 0).any())
            self.assertTrue((yaw < 0).any() and (yaw > 0).any())
            self.assertTrue((commands[:, 1] == 0).all())
        finally:
            env.close()

    def test_command_inputs_and_privileged_critic_separation(self):
        from bdxr_classroom.command_evaluate import set_command
        env = make_env(4, command_mode="mixed", yaw_tracking="imu",
                       yaw_bias_weight=2, linear_tracking="average")
        try:
            env.reset(seed=123)
            actor_terms = set(env.cfg.observations["actor"].terms)
            self.assertEqual(actor_terms, {
                "base_ang_vel", "projected_gravity", "joint_pos",
                "joint_vel", "actions", "command",
            })
            self.assertIn("base_lin_vel", env.cfg.observations["critic"].terms)
            self.assertIn("foot_contact_forces", env.cfg.observations["critic"].terms)
            obs = set_command(env, -0.2, -0.4)
            requested = torch.tensor([-0.2, 0, -0.4], device="cuda").expand(4, -1)
            torch.testing.assert_close(obs["actor"][:, -3:], requested)
            self.assertEqual(obs["actor"].shape, (4, 47))
            self.assertEqual(obs["critic"].shape, (4, 62))
            _, _, _, _, _ = env.step(torch.zeros(4, 10, device="cuda"))
            torch.testing.assert_close(env.command_manager.get_command("twist"), requested)
        finally:
            env.close()

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
