"""Command practice frequency must preserve the actor interface and old defaults."""

import unittest

import torch

from bdxr_classroom.simulation.environment import make_env


class CommandIntervalTests(unittest.TestCase):
    def test_invalid_interval_is_rejected_before_simulator_creation(self):
        for interval in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "Command interval"):
                make_env(command_interval=interval)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_command_timer_restarts_at_requested_frequency(self):
        env = make_env(
            3, command_mode="balanced", command_interval=2,
            yaw_tracking="imu",
        )
        try:
            observations, _ = env.reset(seed=7)
            self.assertEqual(observations["actor"].shape, (3, 47))
            term = env.command_manager.get_term("twist")
            torch.testing.assert_close(term.time_left, torch.full_like(term.time_left, 2))
            term.time_left.fill_(env.step_dt / 2)
            # Exercise the actual command timer; no physics rollout is needed.
            env.command_manager.compute(env.step_dt)
            torch.testing.assert_close(term.time_left, torch.full_like(term.time_left, 2))
        finally:
            env.close()

        original = make_env(1, command_mode="balanced", yaw_tracking="imu")
        try:
            self.assertEqual(
                original.cfg.commands["twist"].resampling_time_range, (3.0, 6.0)
            )
        finally:
            original.close()
