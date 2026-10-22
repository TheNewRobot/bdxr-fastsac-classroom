"""One short classroom assessment; comprehensive checks remain developer tools."""

import json
import time
from pathlib import Path

from bdxr_classroom.evaluation.walking_checks import check, checkpoint_digest
from bdxr_classroom.lesson.presentation import command_table


def assess_walking(lab, run, seconds=10, episodes=3):
    """Measure the final policy in three forward trials, without searching weights.

    The numerical tolerances match the existing walking checks. This shorter,
    forward-only assessment is explicitly not the ten-command validation suite.
    """
    started = time.monotonic()
    run = Path(run)
    weights = run / "checkpoints/final.pt"
    digest = checkpoint_digest(weights)
    output = run / "evaluation" / f"brief_{time.time_ns()}"
    lab.run(
        "bdxr_classroom.evaluation.command_evaluate",
        weights,
        "--cases",
        "forward",
        "--episodes",
        episodes,
        "--seconds",
        seconds,
        "--no-video",
        "--output",
        output,
        *lab.presentation_arguments(),
    )
    summary = json.loads((output / "summary.json").read_text())
    if (
        summary.get("checkpoint_sha256") != digest
        or checkpoint_digest(weights) != digest
        or Path(summary["checkpoint"]).resolve() != weights.resolve()
    ):
        raise RuntimeError("The assessment and checkpoint do not match.")
    targets = json.loads((lab.work / "configs/evaluation_targets.json").read_text())
    targets.update(
        required_cases=["forward"], episodes_per_case=episodes, seconds_per_case=seconds
    )
    targets["description"] = (
        "Short forward-only assessment; original numerical tolerances."
    )
    targets["notes"] = [
        "World measurements are evaluation-only; the actor receives no path feedback.",
        f"{episodes} perturbed starts test one policy, not independent training seeds.",
    ]
    verdict = check(summary, targets)
    verdict.update(
        checkpoint=str(weights),
        checkpoint_sha256=digest,
        evaluation_directory=str(output),
        wall_seconds=time.monotonic() - started,
        scope=f"{episodes} forward trials × {seconds:g}s; not full command validation",
    )
    (output / "brief_check.json").write_text(json.dumps(verdict, indent=2))
    command_table(summary)
    if verdict["passed"]:
        print("Stable forward walking within this short check's tolerances.")
    else:
        trials = summary["cases"]["forward"]
        print(
            "Falls observed."
            if any(t["fell"] for t in trials)
            else "Stayed upright; movement accuracy needs improvement."
        )
        for failure in verdict["failures"]:
            print(" •", failure.removeprefix("forward: "))
    print(f"Assessment time: {verdict['wall_seconds']:.0f}s.")
    print("This check does not establish turning or terrain capability.")
    return verdict
