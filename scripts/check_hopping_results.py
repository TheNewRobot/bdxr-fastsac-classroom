"""Screen hopping checkpoints without confusing standing or foot markers with hops."""

import argparse
import json
import math
from pathlib import Path


def check(summary, targets):
    failures = []
    if not summary.get("checkpoint"):
        failures.append("Missing checkpoint identity")
    if (summary.get("terrain_style") != targets["terrain_style"] or
            summary.get("terrain_height_m") != targets["terrain_height_m"]):
        failures.append("Missing or incorrect physical-terrain evaluation")
    if not summary.get("foot_height_measurement", "").startswith("Sampled collision-sole"):
        failures.append("Sole clearance was not measured")
    for case in targets["required_cases"]:
        rows = summary.get("cases", {}).get(case, [])
        if len(rows) < targets["episodes_per_case"]:
            failures.append(f"{case}: missing trials")
        for index, row in enumerate(rows):
            prefix = f"{case} trial {index}"
            keys = ("duration_s", "landed_hops", "peak_simultaneous_foot_clearance", "mean_speed")
            if not all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in keys):
                failures.append(f"{prefix}: missing or nonfinite measurements")
                continue
            if row.get("fell") is not False or row["duration_s"] < targets["seconds_per_case"] - 1e-6:
                failures.append(f"{prefix}: fall or incomplete trial")
            if row["landed_hops"] < targets["minimum_landed_hops"]:
                failures.append(f"{prefix}: too few landed hops")
            if row["peak_simultaneous_foot_clearance"] < targets["minimum_peak_sole_clearance_m"]:
                failures.append(f"{prefix}: insufficient sole clearance")
            if case != "stop" and row["mean_speed"] < targets["minimum_forward_speed_m_s"]:
                failures.append(f"{prefix}: insufficient forward motion")
            if case == "stop" and abs(row["mean_speed"]) > targets["maximum_stop_speed_m_s"]:
                failures.append(f"{prefix}: excessive motion during stop")
    return {"checkpoint": summary.get("checkpoint"), "passed": not failures,
            "failures": failures, "targets": targets}


def choose(reports, targets):
    """Reports are ordered by preference; never select a failing checkpoint."""
    checks = [check(report, targets) for report in reports]
    selected = next((item["checkpoint"] for item in checks if item["passed"]), None)
    return {"selected_checkpoint": selected, "checks": checks,
            "scope": "Hopping screen only; mode transitions and the full course are separate tests."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    targets = json.loads((Path(__file__).resolve().parents[1] / "configs/hopping_targets.json").read_text())
    result = check(json.loads(args.summary.read_text()), targets)
    args.summary.with_name("hopping_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
