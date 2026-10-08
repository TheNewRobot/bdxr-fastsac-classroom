"""A repeated success must not erase a failed handover for the same weights."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "pair_screen", Path(__file__).resolve().parents[1] / "scripts/validate_policy_pair.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PairHistory(unittest.TestCase):
    def test_failure_persists_for_exact_pair_after_relocation(self):
        good = dict(walking_sha256="walk", hopping_sha256="hop", weights_unchanged=True,
                    tests={"flat": {"passed": True}, "on_step": {"passed": True}})
        bad = dict(good, walking="/old/location.pt",
                   tests={"flat": {"passed": True}, "on_step": {"passed": False}})
        self.assertTrue(module.history_verdict([good], "walk", "hop")["passed"])
        self.assertFalse(module.history_verdict([bad, good], "walk", "hop")["passed"])
        self.assertFalse(module.history_verdict([good], "different", "hop")["passed"])

    def test_partial_or_changed_weight_test_cannot_qualify(self):
        incomplete = dict(walking_sha256="walk", hopping_sha256="hop",
                          weights_unchanged=True, tests={"flat": {"passed": True}})
        self.assertFalse(module.history_verdict([incomplete], "walk", "hop")["passed"])
        changed = dict(incomplete, weights_unchanged=False,
                       tests={"flat": {"passed": True}, "on_step": {"passed": True}})
        self.assertFalse(module.history_verdict([changed], "walk", "hop")["passed"])

    def test_infrastructure_error_does_not_poison_later_real_measurement(self):
        error = dict(walking_sha256="walk", hopping_sha256="hop",
                     valid_evaluation=False, tests={})
        good = dict(walking_sha256="walk", hopping_sha256="hop", weights_unchanged=True,
                    valid_evaluation=True,
                    tests={"flat": {"passed": True}, "on_step": {"passed": True}})
        verdict = module.history_verdict([error, good], "walk", "hop")
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["evaluation_count"], 1)

    def test_real_failure_survives_an_error_in_the_other_scope(self):
        mixed = dict(walking_sha256="walk", hopping_sha256="hop",
                     weights_unchanged=True, valid_evaluation=False,
                     tests={"flat": {"passed": False, "valid_evaluation": True},
                            "on_step": {"passed": False, "valid_evaluation": False}})
        good = dict(walking_sha256="walk", hopping_sha256="hop", weights_unchanged=True,
                    tests={"flat": {"passed": True}, "on_step": {"passed": True}})
        verdict = module.history_verdict([mixed, good], "walk", "hop")
        self.assertFalse(verdict["passed"])
        self.assertTrue(verdict["recorded_failure"])
