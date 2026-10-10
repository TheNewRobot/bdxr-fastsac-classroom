"""Isolate a course segment; injected starting states are not full-course evidence."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.game.session import GameSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("walking")
    parser.add_argument("--hopping")
    parser.add_argument("--start-x", type=float, required=True)
    parser.add_argument("--seconds", type=float, default=16)
    parser.add_argument("--speed", type=float, default=.3)
    parser.add_argument("--mode", choices=["walk", "hop"], default="walk")
    parser.add_argument("--seed", type=int, default=263)
    parser.add_argument("--course-config", type=Path)
    parser.add_argument("--run-root", type=Path, default=Path("experiments"),
                        help="Parent folder for this diagnostic's timestamped output")
    args = parser.parse_args()
    if args.mode == "hop" and not args.hopping:
        parser.error("Hop mode requires --hopping")
    game = GameSession(args.walking, args.hopping, seed=args.seed,
                       course=json.loads(args.course_config.read_text()) if args.course_config else None,
                       run_root=args.run_root,
                       operator="injected_start_segment_diagnostic",
                       name="course_segment_diagnostic")
    description = dict(vars(args), scope="Injected start, partial course only; not a full traversal")
    description["course_config"] = str(args.course_config) if args.course_config else None
    description["run_root"] = str(args.run_root)
    (game.output / "config/segment.json").write_text(json.dumps(description, indent=2))
    try:
        state = torch.tensor([[args.start_x, 0, .35, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                             device=game.env.device)
        game.env.scene["robot"].write_root_state_to_sim(state)
        # Settle at the injected start using the policy before issuing motion.
        game.command(0, 0, "walk")
        for _ in range(round(1 / game.env.step_dt)):
            game.step()
            if game.score.falls:
                break
        game.command(args.speed, 0, args.mode)
        for _ in range(round(args.seconds / game.env.step_dt)):
            if game.score.falls or game.score.finished:
                break
            game.step()
        report = dict(description, status=game.status(), full_course_completed=False,
                      final_position=game.env.scene["robot"].data.root_link_pos_w[0].tolist())
        (game.output / "evaluation/segment.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        game.close()


if __name__ == "__main__":
    main()
