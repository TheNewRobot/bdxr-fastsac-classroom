"""Explain gait differences using measurements from the actual command trials."""

import json
from pathlib import Path
from statistics import mean

from .presentation import table


def compare_gaits(lab, control, variation):
    rows = []
    budgets = []
    for label, run in [("Control", Path(control)), ("Variation", Path(variation))]:
        folder = run / "evaluation/final_commands"
        lab.run("scripts.media.measure_gait", folder)
        trials = json.loads((folder / "gait_measurements.json").read_text())["forward"][
            "trials"
        ]
        steps = json.loads((run / "metrics/timing.json").read_text())["step"]
        budgets.append(steps)
        if not trials:
            print(
                label,
                "ended too early for gait measurements; inspect its failure report.",
            )
            continue
        tilt = [
            row["swing_sole_tilt_deg"]
            for row in trials
            if row["swing_sole_tilt_deg"] is not None
        ]
        rows.append(
            (
                label,
                steps,
                round(mean(r["foot_landings_per_second"] for r in trials), 2),
                round(mean(r["sole_clearance_p95_m"] for r in trials), 3),
                round(mean(tilt), 1) if tilt else "No swing measured",
                round(mean(r["mean_torso_height_m"] for r in trials), 3),
            )
        )
    table(
        rows,
        [
            "Run",
            "Steps",
            "Foot landings/s",
            "Sole clearance p95 (m)",
            "Swing tilt (deg)",
            "Torso height (m)",
        ],
    )
    if len(set(budgets)) != 1:
        print(
            "The time limit produced unequal step counts. Report this confound before attributing a difference to the reward."
        )
