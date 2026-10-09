"""Check that reported flight measurements describe physical sample intervals."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "measure_gait", Path(__file__).resolve().parents[1] / "scripts/measure_gait.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GaitMeasurements(unittest.TestCase):
    def test_clear_flight_and_small_rebound_are_separate(self):
        samples = []
        for i in range(22):
            flying = 2 <= i < 8 or 15 <= i < 17
            clearance = .06 if 2 <= i < 8 else (.008 if flying else 0)
            samples.append({
                "episode": 0, "time": 1 + i * .02,
                "foot_contacts": [not flying, not flying],
                "sole_clearance_m": [clearance, clearance],
                "sole_tilt_rad": [math.radians(10)] * 2,
                "fell": False, "speed": .3, "total_foot_contact_force": 0 if flying else 90,
            })
        report = module.summarize(samples)["trials"][0]
        self.assertEqual(len(report["complete_contact_free_intervals"]), 2)
        self.assertEqual(len(report["clear_flight_intervals"]), 1)
        self.assertAlmostEqual(report["mean_clear_flight_duration_s"], .12)
        self.assertAlmostEqual(report["mean_clear_flight_peak_m"], .06)
        self.assertAlmostEqual(report["mean_clear_flight_tilt_deg"], 10)
        self.assertEqual(report["per_foot_landings"], [2, 2])

    def test_swing_tilt_excludes_the_other_stance_foot(self):
        samples = [{
            "episode": 0, "time": 1 + i * .02,
            "foot_contacts": [False, True],
            "sole_clearance_m": [.04, 0],
            "sole_tilt_rad": [math.radians(30), 0],
            "fell": False, "speed": .3,
        } for i in range(10)]
        report = module.summarize(samples)["trials"][0]
        self.assertAlmostEqual(report["swing_sole_tilt_deg"], 30)
        self.assertAlmostEqual(report["mean_sole_tilt_deg"], 15)
        self.assertIsNone(report["flight_sole_tilt_deg"])
