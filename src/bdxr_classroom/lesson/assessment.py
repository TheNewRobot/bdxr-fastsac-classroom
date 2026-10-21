"""Run full policy checks and bind verdicts to the exact evaluated weights."""

import json
from pathlib import Path

from bdxr_classroom.evaluation.walking_checks import (
    check,
    checkpoint_digest,
    select_checkpoint,
)


def evaluate_walking(lab, run, episodes=5, seconds=15, checkpoint="final"):
    """Check one saved snapshot and retain its exact evidence directory."""
    run = Path(run)
    weights = run / "checkpoints" / f"{checkpoint}.pt"
    digest = checkpoint_digest(weights)
    targets = json.loads((lab.work / "configs/evaluation_targets.json").read_text())
    lab.run(
        "bdxr_classroom.evaluation.command_evaluate",
        weights,
        "--cases",
        *targets["required_cases"],
        "--episodes",
        episodes,
        "--seconds",
        seconds,
        *lab.presentation_arguments(),
    )
    folder = run / "evaluation" / f"{checkpoint}_commands"
    summary = json.loads((folder / "summary.json").read_text())
    if (
        Path(summary["checkpoint"]).resolve() != weights.resolve()
        or summary.get("checkpoint_sha256") != digest
    ):
        raise RuntimeError(
            "Evaluation belongs to different weights; rerun this snapshot."
        )
    verdict = check(summary, targets)
    if checkpoint_digest(weights) != digest:
        raise RuntimeError(
            "Checkpoint changed during evaluation; select a stable saved file."
        )
    verdict.update(
        checkpoint=str(weights),
        checkpoint_sha256=digest,
        evaluation_directory=str(folder),
    )
    (folder / "selection_check.json").write_text(json.dumps(verdict, indent=2))
    return summary, verdict


def selected_walking(reports):
    """Return a verified checkpoint, never silently promote a failed experiment."""
    return select_checkpoint(reports)


def check_recent_walking(lab, run, summary, verdict, limit=2):
    """Try a bounded number of earlier snapshots under the same full criteria.

    The final checkpoint has already been evaluated. Its step is also saved in
    timing.json, so the duplicate step_NNNNNN.pt snapshot is not evaluated again.
    Every attempted snapshot keeps a separate report and its own recordings.
    """
    if verdict["passed"]:
        return summary, verdict
    run = Path(run)
    final_step = json.loads((run / "metrics/timing.json").read_text())["step"]
    snapshots = []
    for path in (run / "checkpoints").glob("step_*.pt"):
        try:
            step = int(path.stem.removeprefix("step_"))
        except ValueError:
            continue
        if step < final_step:
            snapshots.append((step, path.stem))
    for _, checkpoint in sorted(snapshots, reverse=True)[:limit]:
        print(
            f"Checking saved snapshot {checkpoint}; no additional training.", flush=True
        )
        summary, verdict = evaluate_walking(lab, run, checkpoint=checkpoint)
        print(
            "Passed:", verdict["passed"], "Failures:", verdict["failures"], flush=True
        )
        if verdict["passed"]:
            break
    return summary, verdict
