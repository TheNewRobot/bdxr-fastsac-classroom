"""Repeated physical handovers using a declared, open-loop command schedule."""

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
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--speed", type=float, default=0.15)
    args = parser.parse_args()
    reports = []
    for seed in range(args.seeds):
        game = GameSession(args.walking, args.hopping, seed=2000 + seed, flat=True,
                           render=seed == 0, record=seed == 0,
                           operator="scripted_transition_validation", name="transition_validation")
        visited = []
        try:
            for _ in range(args.cycles):
                for mode, seconds in [("walk", 2), ("hop", 3), ("walk", 2)]:
                    game.command(args.speed, 0, mode)
                    stages = []
                    for _ in range(round(seconds / game.env.step_dt)):
                        stages.append(game.step()["mode"])
                    visited.append({"requested": mode, "reached": mode in stages})
            report = {"seed": 2000 + seed, "cycles": args.cycles,
                      "falls": game.score.falls, "all_modes_reached": all(v["reached"] for v in visited),
                      "duration_s": game.score.elapsed, "experiment": str(game.output)}
            (game.output / "evaluation/transitions.json").write_text(json.dumps(report, indent=2))
            reports.append(report)
            print(report, flush=True)
        finally:
            game.close()
    result = {"operator": "scripted open-loop commands; not an autonomous navigation policy",
              "reports": reports, "passed": all(r["falls"] == 0 and r["all_modes_reached"] for r in reports)}
    path = Path(reports[0]["experiment"]) / "evaluation/transition_suite.json"
    path.write_text(json.dumps(result, indent=2))
    print(path)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
