"""Repeated physical handovers using a declared, open-loop command schedule."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.game.session import GameSession
from bdxr_classroom.experiments import create_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("walking")
    parser.add_argument("hopping")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--speed", type=float, default=0.15)
    parser.add_argument("--head-motion-config", type=Path)
    parser.add_argument(
        "--result-file",
        type=Path,
        help="Also save the suite report at this explicit path",
    )
    parser.add_argument(
        "--start-x",
        type=float,
        help="Inject a starting position on the course; not full-traversal evidence",
    )
    parser.add_argument(
        "--start-height",
        type=float,
        default=0,
        help="Raise the injected root pose by the local ground height in metres",
    )
    args = parser.parse_args()
    if args.start_height and args.start_x is None:
        parser.error("--start-height requires --start-x")
    if args.seeds < 1 or args.cycles < 1:
        parser.error("--seeds and --cycles must be positive")
    # Keep per-seed diagnostics beside their suite report, not as unrelated
    # top-level experiments. Pair checks already store reports under the policy.
    if args.result_file:
        trial_root = args.result_file.parent / (args.result_file.stem + "_trials")
    else:
        trial_root = (
            create_experiment("transition_validation_suite") / "evaluation/trials"
        )
    reports = []
    for seed in range(args.seeds):
        game = GameSession(
            args.walking,
            args.hopping,
            seed=2000 + seed,
            head_config=json.loads(args.head_motion_config.read_text())
            if args.head_motion_config
            else None,
            flat=args.start_x is None,
            render=seed == 0,
            record=seed == 0,
            run_root=trial_root,
            operator="scripted_transition_validation",
            name="transition_validation",
        )
        visited = []
        try:
            scope = (
                "Flat-ground handovers"
                if args.start_x is None
                else "Injected course start; partial diagnostic, not a full traversal"
            )
            (game.output / "config/transition_test.json").write_text(
                json.dumps(dict(vars(args), scope=scope), indent=2, default=str)
            )
            if args.start_x is not None:
                robot = game.env.scene["robot"]
                pose = robot.data.root_link_pose_w.clone()
                # Preserve the reset's perturbed orientation and lateral position.
                pose[:, 0] += args.start_x
                pose[:, 2] += args.start_height
                robot.write_root_link_pose_to_sim(pose)
                game.command(0, 0, "walk")
                for _ in range(round(1 / game.env.step_dt)):
                    game.step()
                    if game.score.falls:
                        break
            for _ in range(args.cycles):
                if game.score.falls:
                    break
                for mode, seconds in [("walk", 2), ("hop", 3), ("walk", 2)]:
                    game.command(args.speed, 0, mode)
                    stages = []
                    for _ in range(round(seconds / game.env.step_dt)):
                        stages.append(game.step()["mode"])
                        if game.score.falls:
                            break
                    visited.append({"requested": mode, "reached": mode in stages})
                    if game.score.falls:
                        break
            report = {
                "seed": 2000 + seed,
                "cycles": args.cycles,
                "scope": scope,
                "falls": game.score.falls,
                "all_modes_reached": (
                    len(visited) == args.cycles * 3
                    and all(v["reached"] for v in visited)
                ),
                "duration_s": game.score.elapsed,
                "experiment": str(game.output),
            }
            (game.output / "evaluation/transitions.json").write_text(
                json.dumps(report, indent=2)
            )
            reports.append(report)
            print(report, flush=True)
        finally:
            game.close()
    result = {
        "operator": "scripted open-loop commands; not an autonomous navigation policy",
        "reports": reports,
        "passed": all(r["falls"] == 0 and r["all_modes_reached"] for r in reports),
    }
    path = Path(reports[0]["experiment"]) / "evaluation/transition_suite.json"
    path.write_text(json.dumps(result, indent=2))
    if args.result_file:
        args.result_file.parent.mkdir(parents=True, exist_ok=True)
        args.result_file.write_text(json.dumps(result, indent=2))
    print(path)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
