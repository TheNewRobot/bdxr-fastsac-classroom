"""Physical course attempt with an explicit fixed command schedule, not a planner."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.game import GameSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("walking")
    parser.add_argument("hopping")
    parser.add_argument("--commands", type=Path)
    parser.add_argument("--seed", type=int, default=263)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    commands = json.loads(args.commands.read_text()) if args.commands else [
        {"seconds": 6, "speed": .3, "yaw": 0, "mode": "walk"},
        {"seconds": 4, "speed": .3, "yaw": 0, "mode": "hop"},
        {"seconds": 6, "speed": .3, "yaw": 0, "mode": "walk"},
        {"seconds": 5, "speed": .3, "yaw": 0, "mode": "hop"},
        {"seconds": 6, "speed": .3, "yaw": 0, "mode": "walk"},
        {"seconds": 4, "speed": .3, "yaw": 0, "mode": "hop"},
        {"seconds": 23, "speed": .3, "yaw": 0, "mode": "walk"},
    ]
    game = GameSession(args.walking, args.hopping, seed=args.seed,
                       render=not args.no_video, record=not args.no_video,
                       operator="fixed_scripted_commands_no_feedback", name="course_validation")
    (game.output / "config/command_schedule.json").write_text(json.dumps(commands, indent=2))
    try:
        for command in commands:
            game.command(command["speed"], command["yaw"], command["mode"])
            for _ in range(round(command["seconds"] / game.env.step_dt)):
                game.step()
                if game.score.finished or game.score.falls:
                    break
            print({"time": game.score.elapsed, "position": game.env.scene["robot"].data.root_link_pos_w[0].tolist(),
                   "mode": game.stage, "falls": game.score.falls}, flush=True)
            if game.score.finished or game.score.falls:
                break
        report = dict(game.status(), course_completed=game.score.finished,
                      operator="fixed_scripted_commands_no_feedback")
        (game.output / "evaluation/traversal.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        game.close()
    raise SystemExit(0 if report["course_completed"] else 1)


if __name__ == "__main__":
    main()
