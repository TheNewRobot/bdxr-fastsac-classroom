"""The isolated runtime preserves state, reports errors, and releases its process."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

from bdxr_classroom.lesson.simulator import SimulatorCells


class SimulatorCellTests(unittest.TestCase):
    def test_state_errors_and_shutdown(self):
        environment = dict(
            os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src")
        )
        with tempfile.TemporaryDirectory() as work:
            simulator = SimulatorCells(sys.executable, work, environment)
            try:
                simulator.execute("value = 6")
                simulator.execute("assert value * 7 == 42")
                with self.assertRaisesRegex(RuntimeError, "ValueError: test failure"):
                    simulator.execute("raise ValueError('test failure')")
                simulator.execute("assert value == 6")
            finally:
                simulator.close()
            self.assertIsNotNone(simulator.process.poll())
