"""Check the declared full command suite; missing cases are failures."""

import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def check(summary, targets):
    failures = []
    for case in targets["required_cases"]:
        rows = summary["cases"].get(case, [])
        if len(rows) < targets["episodes_per_case"]:
            failures.append(f"{case}: missing evaluations")
            continue
        if any(r["fell"] for r in rows):
            failures.append(f"{case}: fall")
        if any(r["duration_s"] < targets["seconds_per_case"] for r in rows):
            failures.append(f"{case}: incomplete duration")
        for key, limit in [("speed_mae", "speed_mae_max_m_s"), ("yaw_mae", "yaw_rate_mae_max_rad_s")]:
            if any(r[key] > targets[limit] for r in rows):
                failures.append(f"{case}: {key} exceeds {targets[limit]}")
        if case in targets["straight_cases"]:
            drift = [r["max_cross_track_m"] for r in rows]
            if statistics.mean(drift) > targets["mean_max_cross_track_limit_m"]:
                failures.append(f"{case}: mean drift too large")
            if max(drift) > targets["per_episode_max_cross_track_limit_m"]:
                failures.append(f"{case}: worst-episode drift too large")
    return {"passed": not failures, "failures": failures, "targets": targets,
            "checkpoint": summary["checkpoint"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    result = check(json.loads(args.summary.read_text()),
                   json.loads((ROOT / "configs/evaluation_targets.json").read_text()))
    (args.summary.parent / "selection_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
