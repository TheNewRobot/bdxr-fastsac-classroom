"""Real-simulator reset/finish tests using explicit test-only state injection."""

import os
import tempfile
import unittest
import torch
from bdxr_classroom.game import GameSession


@unittest.skipUnless(os.environ.get("BDXR_TEST_WALK"), "Set BDXR_TEST_WALK to a compatible checkpoint")
class SimulatorGameRules(unittest.TestCase):
    def test_physical_fall_reset_and_supported_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            game = GameSession(os.environ["BDXR_TEST_WALK"], render=False, record=False,
                               run_root=directory, operator="injected_state_contract_test")
            try:
                game.command(0, 0)
                for _ in range(20):
                    game.step()
                elapsed = game.score.elapsed
                robot = game.env.scene["robot"]
                # Tip the physical body to force the environment's fall termination.
                state = torch.tensor([[0, 0, .10, .7071068, .7071068, 0, 0, 0, 0, 0, 0, 0, 0]], device="cuda")
                robot.write_root_state_to_sim(state)
                for _ in range(30):
                    game.step()
                    if game.score.falls:
                        break
                self.assertEqual(game.score.falls, 1)
                self.assertGreater(game.score.elapsed, elapsed)
                self.assertTrue(abs(float(robot.data.root_link_pos_w[0, 0])) < .1)
                self.assertFalse(game.score.finished)
                # Verify the scoring trigger, not traversal: place the robot on the finish platform.
                state = torch.tensor([[12, 0, .455, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device="cuda")
                robot.write_root_state_to_sim(state)
                for _ in range(200):
                    game.step()
                    if game.score.finished:
                        break
                self.assertTrue(game.score.finished)
                self.assertEqual(game.score.falls, 1)
                finished_time = game.score.elapsed
                game.step()
                self.assertEqual(game.score.elapsed, finished_time)
                game.new_game()
                self.assertEqual(game.score.elapsed, 0)
                self.assertEqual(game.score.falls, 0)
                self.assertFalse(game.score.finished)
            finally:
                game.close()


if __name__ == "__main__":
    unittest.main()
