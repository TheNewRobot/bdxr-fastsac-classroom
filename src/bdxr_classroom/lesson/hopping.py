"""Named curriculum stages and exact-checkpoint hopping selection."""

import json
import time
from pathlib import Path

from bdxr_classroom.evaluation.hopping_checks import choose
from bdxr_classroom.evaluation.walking_checks import checkpoint_digest


def load_curriculum(work):
    return json.loads((Path(work) / "configs/hopping_curriculum.json").read_text())


def train_stage(lab, stage, learning, previous=None):
    """Fine-tune from the preceding stage; the first stage starts from scratch."""
    settings = {**learning, "mirror_replay": False, **stage["learning"]}
    parent = Path(previous) / "checkpoints/final.pt" if previous else None
    return lab.train(stage["name"], stage["environment"], settings, parent=parent)


def validate_pair(lab, run, walking_checkpoint, episodes=5, seconds=15):
    """Retain all screens and require repeated handovers before auto-selection."""
    run = Path(run)
    history = run / "evaluation/hopping_history"
    history.mkdir(exist_ok=True)
    final_step = json.loads((run / "metrics/timing.json").read_text())["step"]
    checkpoints = [run / "checkpoints/final.pt"] + [
        path
        for path in sorted((run / "checkpoints").glob("step_*.pt"), reverse=True)
        if int(path.stem.split("_")[1]) != final_step
    ]
    current = {}
    for checkpoint in checkpoints:
        digest = checkpoint_digest(checkpoint)
        current[str(checkpoint)] = digest
        folder = (
            run
            / "evaluation"
            / f"{checkpoint.stem}_commands_mixed_0.015_{time.time_ns()}"
        )
        lab.run(
            "bdxr_classroom.evaluation.command_evaluate",
            checkpoint,
            "--cases",
            "slow",
            "forward",
            "stop",
            "--episodes",
            episodes,
            "--seconds",
            seconds,
            "--terrain-height",
            0.015,
            "--terrain-style",
            "mixed",
            "--output",
            folder,
            *lab.presentation_arguments(),
        )
        if checkpoint_digest(checkpoint) != digest:
            raise RuntimeError("Checkpoint changed during hopping evaluation.")
        report = json.loads((folder / "summary.json").read_text())
        report.update(checkpoint_sha256=digest, evaluated_at_ns=time.time_ns())
        (history / (folder.name + ".json")).write_text(json.dumps(report, indent=2))
    reports = sorted(
        (json.loads(p.read_text()) for p in history.glob("*.json")),
        key=lambda row: row.get("evaluated_at_ns", 0),
    )
    targets = json.loads((lab.work / "configs/hopping_targets.json").read_text())
    selection = choose(reports, targets, current)
    (run / "evaluation/checkpoint_selection.json").write_text(
        json.dumps(selection, indent=2)
    )
    for candidate in selection["candidates"]:
        print(
            Path(candidate["checkpoint"]).name,
            "passed hopping screen" if candidate["passed"] else candidate["failures"],
        )
        if not candidate["passed"] or not walking_checkpoint:
            continue
        try:
            lab.run(
                "scripts.evaluation.validate_policy_pair",
                walking_checkpoint,
                candidate["checkpoint"],
                "--output",
                run / "evaluation",
                *lab.presentation_arguments(),
            )
        except RuntimeError as error:
            print(error)
            continue
        return candidate["checkpoint"]
    return None
