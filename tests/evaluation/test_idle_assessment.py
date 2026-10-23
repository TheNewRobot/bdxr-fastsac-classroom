"""Parallel evaluations must expose individual command-response failures."""

import unittest

from scripts.evaluation.validate_idle import stationary_start_indices


class IdleAssessmentTests(unittest.TestCase):
    def test_two_moving_starts_do_not_hide_one_stall(self):
        row = {
            "command": [-.15, 0],
            "speed": [.0004, -.144, -.141],
            "yaw": [.006, -.002, -.005],
        }
        self.assertEqual(stationary_start_indices(row), [0])
        row["speed"][0] = -.14
        self.assertEqual(stationary_start_indices(row), [])
        row["command"] = [0, .4]
        self.assertEqual(stationary_start_indices(row), [0, 1, 2])
        row["command"] = [0, 0]
        self.assertEqual(stationary_start_indices(row), [])
