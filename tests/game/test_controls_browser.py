"""Use the actual game page to check hold/release, telemetry, and both modes."""

import json
import unittest
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


@unittest.skipUnless(sync_playwright, "Playwright is a development-only dependency")
class BrowserControls(unittest.TestCase):
    def test_controls_and_mobile_layout(self):
        root = Path(__file__).resolve().parents[2]
        document = (root / "src/bdxr_classroom/game/web/game.html").read_text()
        commands = []
        state = dict(
            ready=True, walking_only=True, paused=False, time=2., falls=0,
            course_checkpoints=0, mode="walk", speed=0., yaw=0.,
            actual_speed=.26, actual_yaw=.03, finished=False,
            experiment="test", walking="test/final.pt", hopping="",
            command_limits=dict(speed_min=-.2, speed_max=.45, yaw_max=.6),
        )
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport=dict(width=1200, height=950))

            def respond(route):
                path = route.request.url.split("demo.test", 1)[-1]
                if path == "/api/state":
                    route.fulfill(json=state)
                elif path == "/api/command":
                    command = json.loads(route.request.post_data)
                    commands.append(command)
                    if command.get("action") == "pause":
                        state["paused"] = not state["paused"]
                    route.fulfill(json={"ok": True})
                elif path == "/api/checkpoints":
                    route.fulfill(json=[])
                elif path == "/api/frame":
                    route.fulfill(body=(root / "assets/lesson/course_preview.png").read_bytes(), content_type="image/png")
                else:
                    route.fulfill(body=document, content_type="text/html")

            page.route("http://demo.test/**", respond)
            page.goto("http://demo.test/")
            page.wait_for_function("document.getElementById('actualSpeed').textContent === '0.26'")
            self.assertEqual(page.locator('input[type=range]').count(), 0)
            page.locator('[data-speed=".5"]').click()
            page.locator('[data-key="KeyW"]').hover()
            page.mouse.down()
            page.wait_for_function("document.getElementById('requested').textContent.includes('0.45 m/s')")
            page.mouse.move(10, 10)
            page.mouse.up()
            page.wait_for_function("document.getElementById('requested').textContent.includes('0.00 m/s')")
            page.keyboard.down("w")
            page.keyboard.down("a")
            page.wait_for_function("document.getElementById('requested').textContent.includes('0.45 m/s · 0.40 rad/s')")
            page.evaluate("window.dispatchEvent(new Event('blur'))")
            page.wait_for_function("document.getElementById('requested').textContent.includes('0.00 m/s · 0.00 rad/s')")
            page.keyboard.up("w")
            page.keyboard.up("a")
            page.locator("#pause").click()
            page.wait_for_function("document.getElementById('pause').textContent === 'Resume'")
            page.wait_for_timeout(350)
            self.assertTrue(state["paused"])
            self.assertTrue(any(c.get("speed") == .45 for c in commands))
            drive_commands = [c for c in commands if "speed" in c]
            self.assertEqual(drive_commands[-1]["speed"], 0)
            self.assertTrue(all(c["hold_to_move"] for c in drive_commands))
            state["walking_only"] = False
            page.wait_for_function("!document.getElementById('modeControls').hidden")
            page.locator("#pause").click()
            page.wait_for_function("document.getElementById('pause').textContent === 'Pause'")
            page.locator("#hopMode").click()
            page.wait_for_timeout(100)
            self.assertTrue(any(c.get("mode") == "hop" for c in commands))
            page.locator("#stop").click()
            page.wait_for_timeout(100)
            self.assertEqual([c for c in commands if "speed" in c][-1]["mode"], "walk")
            page.set_viewport_size(dict(width=390, height=844))
            self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
            browser.close()
