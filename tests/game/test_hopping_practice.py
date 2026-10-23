"""Single-click control, flat-floor contacts and a separate notebook workflow."""

import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco

from bdxr_classroom.hopping_demo.arena import build_arena
from bdxr_classroom.hopping_demo.session import JumpCycle, clear_hop
from bdxr_classroom.lesson.hop_practice import launch_hop_practice, watch_hop_recording

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


class HopControlTests(unittest.TestCase):
    def test_small_hops_contact_flicker_and_repeated_hops_do_not_pass(self):
        good = dict(fell=False, peak_clearance_m=0.052,
                    longest_flight_s=0.14, flight_events=1)
        self.assertTrue(clear_hop(good))
        for change in (
            {"peak_clearance_m": 0.01}, {"longest_flight_s": 0.02},
            {"flight_events": 2}, {"fell": True},
        ):
            self.assertFalse(clear_hop({**good, **change}))

    def test_one_click_one_phase_cycle_then_stance(self):
        cycle = JumpCycle()
        for _ in range(100):
            cycle.advance(0.02)
        self.assertEqual(cycle.phase, 0)
        self.assertTrue(cycle.request())
        for _ in range(49):
            cycle.advance(0.02)
            self.assertFalse(cycle.request())
        self.assertTrue(cycle.active)
        cycle.advance(0.02)
        self.assertFalse(cycle.active)
        for _ in range(100):
            cycle.advance(0.02)
        self.assertEqual(cycle.phase, 0)

    def test_practice_floor_accepts_full_mesh_collision_bit(self):
        spec = mujoco.MjSpec()
        terrain = spec.worldbody.add_body(name="terrain")
        floor = terrain.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,
                                 size=(0, 0, 0.1), conaffinity=1)
        build_arena(spec)
        self.assertTrue(floor.conaffinity & 2)
        pad = spec.geom("hop_pad")
        self.assertTrue(pad.conaffinity & 2)
        self.assertEqual(pad.contype, 1)
        model = spec.compile()
        self.assertGreater(model.ngeom, 1)

    def test_optional_replay_does_not_contact_or_pause_simulator(self):
        with patch("bdxr_classroom.lesson.hop_practice.game_state") as state:
            self.assertIsNone(watch_hop_recording(None))
            state.assert_not_called()

    def test_hopping_notebook_needs_no_walker(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "notebooks/extensions/hopping.py").read_text()
        for removed in ("WALKING_CHECKPOINT", "select_hopper", "launch_game("):
            self.assertNotIn(removed, source)
        self.assertIn("launch_hop_practice", source)
        self.assertIn("check_and_watch_hops", source)

    def test_busy_port_does_not_start_or_stop_another_application(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with patch("bdxr_classroom.lesson.hop_practice.subprocess.Popen") as start:
                with self.assertRaisesRegex(RuntimeError, "Choose another HOP_PORT"):
                    launch_hop_practice(SimpleNamespace(), "/unused", port=port)
                start.assert_not_called()

    def test_failed_or_stale_check_cannot_launch_the_demo(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "checkpoints").mkdir()
            weights = run / "checkpoints/final.pt"
            weights.write_bytes(b"new weights")
            for check in (
                {"passed": False},
                {"passed": True, "checkpoint": str(weights), "checkpoint_sha256": "old"},
            ):
                with patch("bdxr_classroom.lesson.hop_practice.subprocess.Popen") as start:
                    with self.assertRaisesRegex(ValueError, "these weights"):
                        launch_hop_practice(SimpleNamespace(), run, assessment=check)
                    start.assert_not_called()

    @unittest.skipUnless(sync_playwright, "Playwright is development-only")
    def test_buttons_and_mobile_layout(self):
        root = Path(__file__).resolve().parents[2]
        document = (root / "src/bdxr_classroom/hopping_demo/panel.html").read_text()
        commands = []
        state = dict(ready=True, can_jump=True, recording=False, results=[], message="Ready")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1100, "height": 950})

            def respond(route):
                path = route.request.url.split("hop.test", 1)[-1]
                if path == "/api/state":
                    route.fulfill(json=state)
                elif path == "/api/command":
                    value = json.loads(route.request.post_data)
                    commands.append(value)
                    if value["action"] == "jump":
                        state["can_jump"] = False
                    elif value["action"] == "reset":
                        state["can_jump"] = True
                    elif value["action"] == "record":
                        state["recording"] = value["enabled"]
                    route.fulfill(json={"accepted": True})
                elif path == "/api/frame":
                    route.fulfill(status=204)
                else:
                    route.fulfill(body=document, content_type="text/html")

            page.route("http://hop.test/**", respond)
            page.goto("http://hop.test/")
            page.wait_for_function("!document.getElementById('jump').disabled")
            self.assertEqual(commands, [])
            page.locator("#jump").click()
            page.wait_for_function("document.getElementById('jump').disabled")
            self.assertEqual(commands, [{"action": "jump"}])
            page.locator("#reset").click()
            page.wait_for_function("!document.getElementById('jump').disabled")
            page.locator("#record").check()
            page.wait_for_function("document.getElementById('record').checked")
            self.assertEqual(commands[-1], {"action": "record", "enabled": True})
            self.assertEqual(page.locator('input[type="range"]').count(), 0)
            page.set_viewport_size({"width": 390, "height": 844})
            self.assertTrue(page.evaluate(
                "document.documentElement.scrollWidth <= innerWidth"
            ))
            browser.close()
