"""Record visually directed commands with pauses for inspecting rendered frames.

This is an operator interface, not a navigation controller. It exposes images
and the scoreboard, never world position or heading. The normal browser game
remains the student interface.
"""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.game.session import GameSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("walking")
    parser.add_argument("hopping")
    parser.add_argument("--course-config", type=Path)
    parser.add_argument("--seed", type=int, default=263)
    args = parser.parse_args()
    game = GameSession(args.walking, args.hopping, seed=args.seed,
                       course=json.loads(args.course_config.read_text()) if args.course_config else None,
                       operator="manual_visual_commands_with_pauses", name="manual_course_trial")
    preview_count = 0

    def preview():
        nonlocal preview_count
        preview_count += 1
        game.paused = True
        game.set_camera("top")
        path = game.output / "videos" / f"preview_{preview_count:04d}.png"
        Image.fromarray(game.render()).save(path)
        game.set_camera("follow")
        print(json.dumps({"status": game.status(), "preview": str(path.resolve())}), flush=True)

    try:
        preview()
        for line in sys.stdin:
            try:
                command = json.loads(line)
                if command.get("action") == "quit":
                    break
                if command.get("action") == "new":
                    game.new_game()
                    preview()
                    continue
                seconds = float(command.get("seconds", 0))
                if not 0 <= seconds <= 10:
                    raise ValueError("Commands must last between zero and ten simulation seconds")
                with (game.output / "metrics/operator_commands.jsonl").open("a") as log:
                    log.write(json.dumps({"time_before": game.score.elapsed, "command": command,
                                          "operator": "visual inspection between paused commands"}) + "\n")
                game.command(command.get("speed", 0), command.get("yaw", 0),
                             command.get("mode", "walk"))
                old_falls = game.score.falls
                for _ in range(round(seconds / game.env.step_dt)):
                    game.step()
                    if game.score.finished or game.score.falls != old_falls:
                        break
                preview()
            except (ValueError, KeyError, TypeError) as error:
                print(json.dumps({"error": str(error)}), flush=True)
    finally:
        game.close()


if __name__ == "__main__":
    main()
