"""Test fixed human-style hop timings; world measurements score, never steer."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.game.session import GameSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("walking")
    parser.add_argument("hopping")
    parser.add_argument("--walk-times", type=float, nargs="+", default=[5.2, 5.6, 6.0, 6.4])
    parser.add_argument("--hop-speed", type=float, default=.15)
    parser.add_argument("--hop-seconds", type=float, default=8)
    parser.add_argument("--obstacle-scale", type=float, default=1)
    args = parser.parse_args()
    results = []
    for walk_seconds in args.walk_times:
        game = GameSession(args.walking, args.hopping, render=False, record=False,
                           course={"obstacle_scale": args.obstacle_scale},
                           operator="fixed_timing_sweep_no_feedback", name="pipe_timing_sweep")
        commands = [
            {"seconds": walk_seconds, "speed": .3, "mode": "walk"},
            {"seconds": args.hop_seconds, "speed": args.hop_speed, "mode": "hop"},
            {"seconds": 2, "speed": .15, "mode": "walk"},
        ]
        (game.output / "config/command_schedule.json").write_text(json.dumps(commands, indent=2))
        try:
            for command in commands:
                game.command(command["speed"], 0, command["mode"])
                for _ in range(round(command["seconds"] / game.env.step_dt)):
                    game.step()
                    if game.score.falls:
                        break
                if game.score.falls:
                    break
            robot = game.env.scene["robot"].data
            feet_beyond_pipe = bool((robot.site_pos_w[0, game.foot_ids, 0] > 2.12).all())
            game.log.flush()
            rows = [json.loads(line) for line in
                    (game.output / "metrics/gameplay.jsonl").read_text().splitlines()]
            crossing = [row for row in rows if 1.85 <= row["position"][0] <= 2.15]
            stayed_in_corridor = bool(crossing) and all(
                abs(foot[1]) < .60 for row in crossing for foot in row["foot_positions"])
            flight_over_pipe = any(
                abs(row["position"][0] - 2) < .06
                and not any(row["foot_contacts"])
                and min(foot[2] for foot in row["foot_positions"]) > .035 * args.obstacle_scale + .015
                for row in crossing)
            cleared = (game.score.falls == 0 and feet_beyond_pipe and stayed_in_corridor
                       and float(-robot.projected_gravity_b[0, 2]) > .9)
            report = {
                "experiment": str(game.output), "walk_seconds": walk_seconds,
                "hop_speed": args.hop_speed, "obstacle_scale": args.obstacle_scale,
                "hop_seconds": args.hop_seconds, "final_mode": game.stage,
                "falls": game.score.falls,
                "position": robot.root_link_pos_w[0].tolist(),
                "upright": float(-robot.projected_gravity_b[0, 2]),
                "pipe_cleared": cleared, "hopped_pipe": cleared and flight_over_pipe,
                "stayed_in_corridor": stayed_in_corridor, "flight_over_pipe": flight_over_pipe,
                "full_course_completed": False,
                "scope": "One pipe and final walk handover; fixed commands, no position feedback",
            }
            (game.output / "evaluation/pipe_timing.json").write_text(json.dumps(report, indent=2))
            results.append(report)
            print(json.dumps(report), flush=True)
        finally:
            game.close()
    summary = Path(results[0]["experiment"]) / "evaluation/timing_sweep.json"
    summary.write_text(json.dumps(results, indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
