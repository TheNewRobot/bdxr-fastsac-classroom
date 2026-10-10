"""Run full policy checks and bind verdicts to the exact evaluated weights."""

import json
from pathlib import Path

from bdxr_classroom.evaluation.walking_checks import (
    check,
    checkpoint_digest,
    select_checkpoint,
)


def evaluate_walking(lab, run, episodes=5, seconds=15):
    run = Path(run)
    checkpoint = run / "checkpoints/final.pt"
    digest = checkpoint_digest(checkpoint)
    targets = json.loads((lab.work / "configs/evaluation_targets.json").read_text())
    lab.run(
        "bdxr_classroom.evaluation.command_evaluate",
        checkpoint,
        "--cases",
        *targets["required_cases"],
        "--episodes",
        episodes,
        "--seconds",
        seconds,
        *lab.presentation_arguments(),
    )
    folder = run / "evaluation/final_commands"
    summary = json.loads((folder / "summary.json").read_text())
    verdict = check(summary, targets)
    if checkpoint_digest(checkpoint) != digest:
        raise RuntimeError(
            "Checkpoint changed during evaluation; select a stable saved file."
        )
    verdict.update(checkpoint=str(checkpoint), checkpoint_sha256=digest)
    (folder / "selection_check.json").write_text(json.dumps(verdict, indent=2))
    return summary, verdict


def selected_walking(reports):
    """Return a verified checkpoint, never silently promote a failed experiment."""
    return select_checkpoint(reports)
