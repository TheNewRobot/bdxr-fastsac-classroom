"""The scoring contract is independent of policy quality."""

import unittest
from bdxr_classroom.game import Score


class GameRules(unittest.TestCase):
    def test_fall_preserves_elapsed_time_and_accumulates_count(self):
        score = Score()
        score.advance(10, False, False)
        self.assertEqual(score.advance(.02, True, False), "fall")
        self.assertEqual(score.falls, 1)
        self.assertAlmostEqual(score.elapsed, 10.02)
        score.advance(4, False, False)
        score.advance(.02, True, False)
        self.assertEqual(score.falls, 2)
        self.assertAlmostEqual(score.elapsed, 14.04)

    def test_finish_requires_sustained_support_and_stops_clock(self):
        score = Score()
        score.advance(.3, False, True)
        self.assertFalse(score.finished)
        score.advance(.1, False, False)
        self.assertEqual(score.supported_finish_time, 0)
        score.advance(.2, False, True)
        self.assertEqual(score.advance(.2, False, True), "finish")
        elapsed = score.elapsed
        score.advance(10, True, False)
        self.assertEqual(score.elapsed, elapsed)
        self.assertEqual(score.falls, 0)


if __name__ == "__main__":
    unittest.main()
