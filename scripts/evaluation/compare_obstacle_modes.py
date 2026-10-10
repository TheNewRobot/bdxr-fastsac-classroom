"""Compare walking and hopping on the same middle step with repeated starts.

This is an injected-start diagnostic, not a full-course validation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.experiments import capture_source, create_experiment


def read_report(text):
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            report, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(report, dict) and "final_position" in report:
            return report
    raise ValueError("Segment process did not return its result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("walking", type=Path)
    parser.add_argument("hopping", type=Path)
    parser.add_argument("--step-height", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[282, 283, 284])
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--head-motion-config", type=Path)
    args = parser.parse_args()
    if not 0 < args.step_height < 1 or args.seconds <= 0:
        parser.error("Use a positive step height below 1 m and a positive duration")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    walkers = args.walking.resolve()
    hopper = args.hopping.resolve()
    identities = {
        name: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in [("walking", walkers), ("hopping", hopper)]
    }
    out = create_experiment("obstacle_mode_comparison", ROOT / "experiments")
    config = out / "config/course.json"
    config.write_text(json.dumps({"step_height": args.step_height}, indent=2))
    manifest = {
        "policies": identities,
        "seeds": args.seeds,
        "seconds": args.seconds,
        "step_height_m": args.step_height,
        "start_x_m": 4.0,
        "cases": [["walk", 0.3], ["walk", 0.45], ["hop", 0.3]],
        "scope": "Injected start, fixed commands, middle step only; not full-course evidence.",
        "crossing_rule": "No falls, final x > 5.3 m, and all samples over the step remain within |y| < .6 m.",
        "head_motion": json.loads(args.head_motion_config.read_text())
        if args.head_motion_config
        else {"enabled": False},
    }
    (out / "config/comparison.json").write_text(json.dumps(manifest, indent=2))
    capture_source(out)
    records = []
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MUJOCO_GL="egl")
    for seed in args.seeds:
        for mode, speed in manifest["cases"]:
            log = out / "evaluation" / f"{seed}_{mode}_{speed}.log"
            with log.open("w") as stream:
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/evaluation/diagnose_course_segment.py"),
                        str(walkers),
                        "--hopping",
                        str(hopper),
                        "--start-x",
                        "4",
                        "--seconds",
                        str(args.seconds),
                        "--mode",
                        mode,
                        "--speed",
                        str(speed),
                        "--seed",
                        str(seed),
                        "--course-config",
                        str(config),
                        "--run-root",
                        str(out / "evaluation/trials"),
                        *(
                            ["--head-motion-config", str(args.head_motion_config.resolve())]
                            if args.head_motion_config
                            else []
                        ),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            result = read_report(log.read_text())
            trial = ROOT / result["status"]["experiment"]
            rows = [
                json.loads(line)
                for line in (trial / "metrics/gameplay.jsonl").read_text().splitlines()
            ]
            over_step = [row for row in rows if 4.58 <= row["position"][0] <= 5.02]
            within_width = bool(over_step) and all(
                abs(row["position"][1]) < 0.6 for row in over_step
            )
            crossed = (
                result["status"]["falls"] == 0
                and result["final_position"][0] > 5.3
                and within_width
            )
            record = {
                "seed": seed,
                "mode": mode,
                "speed": speed,
                "crossed": crossed,
                "within_step_width": within_width,
                "result": result,
            }
            records.append(record)
            summary = {
                **manifest,
                "records": records,
                "complete": len(records) == 3 * len(args.seeds),
                "hop_required_for_tested_cases": (
                    len(records) == 3 * len(args.seeds)
                    and all(r["crossed"] == (r["mode"] == "hop") for r in records)
                ),
                "limitation": "Does not prove that all possible walking commands fail. "
                "Inspect crossing videos and validate the full course separately.",
            }
            (out / "evaluation/summary.json").write_text(json.dumps(summary, indent=2))
            print(
                json.dumps(
                    {key: record[key] for key in ["seed", "mode", "speed", "crossed"]}
                ),
                flush=True,
            )
    print(out, flush=True)


if __name__ == "__main__":
    main()
