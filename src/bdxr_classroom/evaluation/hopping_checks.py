"""Screen hopping checkpoints without confusing standing or foot markers with hops."""

import math


def check(summary, targets):
    failures = []
    if not summary.get("checkpoint"):
        failures.append("Missing checkpoint identity")
    if (
        summary.get("terrain_style") != targets["terrain_style"]
        or summary.get("terrain_height_m") != targets["terrain_height_m"]
    ):
        failures.append("Missing or incorrect physical-terrain evaluation")
    if not summary.get("foot_height_measurement", "").startswith(
        "Sampled collision-sole"
    ):
        failures.append("Sole clearance was not measured")
    for case in targets["required_cases"]:
        rows = summary.get("cases", {}).get(case, [])
        if len(rows) < targets["episodes_per_case"]:
            failures.append(f"{case}: missing trials")
        for index, row in enumerate(rows):
            prefix = f"{case} trial {index}"
            keys = (
                "duration_s",
                "landed_hops",
                "peak_simultaneous_foot_clearance",
                "mean_speed",
            )
            if not all(
                isinstance(row.get(k), (int, float)) and math.isfinite(row[k])
                for k in keys
            ):
                failures.append(f"{prefix}: missing or nonfinite measurements")
                continue
            if (
                row.get("fell") is not False
                or row["duration_s"] < targets["seconds_per_case"] - 1e-6
            ):
                failures.append(f"{prefix}: fall or incomplete trial")
            if row["landed_hops"] < targets["minimum_landed_hops"]:
                failures.append(f"{prefix}: too few landed hops")
            if (
                row["peak_simultaneous_foot_clearance"]
                < targets["minimum_peak_sole_clearance_m"]
            ):
                failures.append(f"{prefix}: insufficient sole clearance")
            if (
                case != "stop"
                and row["mean_speed"] < targets["minimum_forward_speed_m_s"]
            ):
                failures.append(f"{prefix}: insufficient forward motion")
            if (
                case == "stop"
                and abs(row["mean_speed"]) > targets["maximum_stop_speed_m_s"]
            ):
                failures.append(f"{prefix}: excessive motion during stop")
    return {
        "checkpoint": summary.get("checkpoint"),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "evaluation_directory": summary.get("evaluation_directory"),
        "passed": not failures,
        "failures": failures,
        "targets": targets,
    }


def choose(reports, targets, current_checkpoints=None):
    """Use every recorded test of the exact current weights, in preference order."""
    checks = [check(report, targets) for report in reports]
    identities = (
        list(current_checkpoints.items())
        if current_checkpoints is not None
        else list(
            dict.fromkeys((c["checkpoint"], c["checkpoint_sha256"]) for c in checks)
        )
    )
    candidates = []
    for checkpoint, digest in identities:
        # Hash identity survives moving an exported experiment to a new runtime.
        trials = [
            c
            for c in checks
            if (
                c["checkpoint_sha256"] == digest
                if digest is not None
                else (c["checkpoint"], c["checkpoint_sha256"]) == (checkpoint, None)
            )
        ]
        failures = [failure for trial in trials for failure in trial["failures"]]
        candidates.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_sha256": digest,
                "evaluation_count": len(trials),
                "passed": bool(trials) and all(trial["passed"] for trial in trials),
                "failures": failures
                if trials
                else ["No evaluations of these exact weights"],
            }
        )
    selected = next((item["checkpoint"] for item in candidates if item["passed"]), None)
    return {
        "selected_checkpoint": selected,
        "checks": checks,
        "candidates": candidates,
        "scope": "Hopping screen only; mode transitions and the full course are separate tests.",
    }
