import unittest

from bdxr_classroom.game.course_progress import CourseProgress


def feet(x, y=0):
    return [[x - .05, y - .10, .1], [x + .05, y + .10, .1]]


class CourseProgressTests(unittest.TestCase):
    def test_bypass_and_finish_teleport_do_not_count(self):
        progress = CourseProgress()
        progress.update(feet(12))
        self.assertFalse(progress.complete)
        for x in [1.8, 2, 2.2]:
            progress.update(feet(x, .8))
        self.assertEqual(progress.completed, 0)

    def test_return_to_missed_challenge_and_complete_route(self):
        progress = CourseProgress()
        for x, y in [(1.8, 0), (2, .8), (2.2, 0)]:
            progress.update(feet(x, y))
        self.assertEqual(progress.completed, 0)
        for front, back, _ in progress.checkpoints:
            for x in [front - .2, front, (front + back) / 2, back + .2]:
                progress.update(feet(x))
        self.assertTrue(progress.complete)
