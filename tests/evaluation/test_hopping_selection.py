"""A checkpoint screen must reject falls, standing, and incomplete evidence."""

import copy
import json
from pathlib import Path
import unittest

from scripts.evaluation.check_hopping_results import check, choose


class HoppingSelection(unittest.TestCase):
    def setUp(self):
        self.targets = json.loads(
            (
                Path(__file__).resolve().parents[2] / "configs/hopping_targets.json"
            ).read_text()
        )
        self.report = {
            "checkpoint": "earlier.pt",
            "terrain_style": "mixed",
            "terrain_height_m": 0.015,
            "foot_height_measurement": "Sampled collision-sole clearance",
            "cases": {
                case: [
                    dict(
                        fell=False,
                        duration_s=15.0,
                        landed_hops=10,
                        peak_simultaneous_foot_clearance=0.04,
                        mean_speed=0.0 if case == "stop" else 0.1,
                    )
                    for _ in range(5)
                ]
                for case in ("slow", "forward", "stop")
            },
        }

    def test_failing_final_never_overrides_passing_earlier_checkpoint(self):
        final = copy.deepcopy(self.report)
        final["checkpoint"] = "final.pt"
        final["cases"]["forward"][4]["fell"] = True
        result = choose([final, self.report], self.targets)
        self.assertEqual(result["selected_checkpoint"], "earlier.pt")
        self.assertFalse(result["checks"][0]["passed"])
        self.assertIsNone(choose([final], self.targets)["selected_checkpoint"])

    def test_standing_wrong_direction_and_missing_measurements_fail(self):
        for field, value in (
            ("landed_hops", 0),
            ("mean_speed", -0.1),
            ("peak_simultaneous_foot_clearance", 0.01),
            ("duration_s", 8.0),
            ("mean_speed", float("nan")),
        ):
            with self.subTest(field=field, value=value):
                report = copy.deepcopy(self.report)
                report["cases"]["forward"][0][field] = value
                self.assertFalse(check(report, self.targets)["passed"])
        report = copy.deepcopy(self.report)
        del report["cases"]["slow"][0]["mean_speed"]
        self.assertFalse(check(report, self.targets)["passed"])

    def test_marker_height_flat_ground_and_short_suites_do_not_qualify(self):
        mutations = [
            lambda r: r.update(foot_height_measurement="Foot reference-site height"),
            lambda r: r.update(terrain_height_m=0),
            lambda r: r["cases"].pop("stop"),
            lambda r: r["cases"]["forward"].pop(),
            lambda r: r.pop("checkpoint"),
        ]
        for mutate in mutations:
            report = copy.deepcopy(self.report)
            mutate(report)
            self.assertFalse(check(report, self.targets)["passed"])
        self.assertTrue(check(self.report, self.targets)["passed"])

    def test_rerunning_does_not_erase_a_failure(self):
        failure = copy.deepcopy(self.report)
        failure["cases"]["forward"][0]["fell"] = True
        result = choose([failure, self.report], self.targets)
        self.assertIsNone(result["selected_checkpoint"])
        self.assertEqual(result["candidates"][0]["evaluation_count"], 2)

    def test_old_weights_cannot_qualify_a_replaced_checkpoint(self):
        old = dict(self.report, checkpoint_sha256="old-weights")
        current = {"earlier.pt": "new-weights"}
        self.assertIsNone(choose([old], self.targets, current)["selected_checkpoint"])
        new = dict(self.report, checkpoint_sha256="new-weights")
        old_failure = copy.deepcopy(old)
        old_failure["cases"]["stop"][0]["fell"] = True
        result = choose([old_failure, new], self.targets, current)
        self.assertEqual(result["selected_checkpoint"], "earlier.pt")
        self.assertEqual(result["candidates"][0]["evaluation_count"], 1)

    def test_moving_a_backup_does_not_forget_its_failures(self):
        moved = dict(self.report, checkpoint_sha256="same-weights")
        moved["cases"] = copy.deepcopy(self.report["cases"])
        moved["cases"]["forward"][0]["fell"] = True
        result = choose(
            [moved], self.targets, {"new-location/earlier.pt": "same-weights"}
        )
        self.assertIsNone(result["selected_checkpoint"])
        self.assertEqual(result["candidates"][0]["evaluation_count"], 1)
