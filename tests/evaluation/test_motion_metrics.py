import unittest
from bdxr_classroom.evaluation.motion_metrics import count_hops


class HopCounting(unittest.TestCase):
    def samples(self, start, states):
        return [{"time": start + i * .02, "foot_contacts": [not airborne] * 2,
                 "foot_heights": [.04 if airborne else 0] * 2, "upright": 1}
                for i, airborne in enumerate(states)]

    def test_initial_drop_is_not_a_hop(self):
        self.assertEqual(count_hops(self.samples(0, [True] * 5 + [False] * 20), .02), (0, 0))

    def test_rebound_requires_a_settled_landing(self):
        rows = self.samples(1, [True] * 4 + [False] * 2 + [True] * 3 + [False] * 15)
        self.assertEqual(count_hops(rows, .02), (2, 1))
