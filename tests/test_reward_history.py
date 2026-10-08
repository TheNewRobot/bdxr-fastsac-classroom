"""History-based reward state must reset independently for each episode."""

from types import SimpleNamespace
import math
import unittest
import torch
from bdxr_classroom.behaviors import yaw_bias_tracking, lateral_bias_tracking, average_linear_velocity_tracking, hop_height_tracking, imu_turn_rate


class RewardHistory(unittest.TestCase):
    def test_sideways_drift_reward_preserves_forward_commands_and_resets_per_world(self):
        velocity = torch.tensor([[.3, .05, 0.], [-.2, .05, 0.]])
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=velocity))},
            command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(2, 3)),
            episode_length_buf=torch.tensor([100, 100]), step_dt=.02)
        for _ in range(100):
            reward = lateral_bias_tracking(env)
        self.assertTrue((reward < .02).all())
        velocity[:, 1] = 0
        env.episode_length_buf[0] = 1
        reward = lateral_bias_tracking(env)
        self.assertEqual(float(reward[0]), 1)
        self.assertLess(float(reward[1]), .05)
        env.episode_length_buf[1] = 1
        self.assertTrue(torch.equal(lateral_bias_tracking(env), torch.ones(2)))

    def test_imu_turn_rate_removes_roll_pitch_coupling(self):
        roll, pitch = .3, -.2
        roll_rate, pitch_rate = .8, -.4
        yaw_rates = [0., .4, -.4]
        gravity = [math.sin(pitch), -math.sin(roll) * math.cos(pitch),
                   -math.cos(roll) * math.cos(pitch)]
        gyro = [[roll_rate - math.sin(pitch) * yaw,
                 math.cos(roll) * pitch_rate + math.sin(roll) * math.cos(pitch) * yaw,
                 -math.sin(roll) * pitch_rate + math.cos(roll) * math.cos(pitch) * yaw]
                for yaw in yaw_rates]
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(
            projected_gravity_b=torch.tensor([gravity] * 3),
            root_link_ang_vel_b=torch.tensor(gyro)))})
        torch.testing.assert_close(imu_turn_rate(env), torch.tensor(yaw_rates), atol=1e-6, rtol=1e-6)

    def test_flight_height_bonus_requires_both_feet_to_rise(self):
        positions = torch.zeros(1, 2, 3)
        robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=positions),
                                find_sites=lambda _: ([0, 1], None))
        env = SimpleNamespace(scene={"robot": robot}, episode_length_buf=torch.tensor([5]), step_dt=.02)
        self.assertEqual(float(hop_height_tracking(env, .1, flight_only=True)[0]), 0)
        env.episode_length_buf[0] = 24
        positions[..., 2] = .1
        self.assertGreater(float(hop_height_tracking(env, .1, flight_only=True)[0]), 3.9)
        positions[0, 1, 2] = 0
        self.assertLess(float(hop_height_tracking(env, .1, flight_only=True)[0]), .01)

    def test_average_speed_rewards_command_tracking_and_resets(self):
        velocity = torch.tensor([[.3, 0., 0.], [-.3, 0., 0.]])
        env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=velocity))},
            command_manager=SimpleNamespace(
                get_command=lambda _: torch.tensor([[.3, 0., 0.], [.3, 0., 0.]])),
            episode_length_buf=torch.tensor([100, 100]), step_dt=.02)
        for _ in range(100):
            reward = average_linear_velocity_tracking(env)
        self.assertEqual(float(reward[0]), 1)
        self.assertLess(float(reward[1]), .025)
        env.episode_length_buf[1] = 1
        velocity[1, 0] = .3
        reward = average_linear_velocity_tracking(env)
        self.assertEqual(float(reward[1]), 1)

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
