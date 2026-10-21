"""Run training and recording jobs while keeping notebook cells small.

This module handles subprocesses and paths. Environment and learning settings
remain explicit dictionaries in the lesson; they map directly to CLI arguments.
"""

import json
import time
from pathlib import Path

from .process import run_logged


def command_arguments(settings):
    """Translate named settings without silently changing numeric values."""
    arguments = []
    for name, value in settings.items():
        flag = "--" + name.replace("_", "-")
        if value is None or value is False:
            continue
        arguments.append(flag)
        if value is not True:
            arguments.append(str(value))
    return arguments


class Lesson:
    def __init__(self, work, python, environment):
        self.work = Path(work)
        self.python = str(python)
        self.environment = dict(environment)
        self.runs = []
        self.game_process = None

    def presentation_arguments(self):
        """Apply the internal rendering configuration consistently at inference."""
        path = self.work / "configs/head_motion.json"
        return ["--head-motion-config", str(path)] if path.is_file() else []

    def run(self, module, *arguments):
        """Stream concise progress and retain the complete log, including failures."""
        log_directory = self.work / "artifacts/logs"
        log_directory.mkdir(parents=True, exist_ok=True)
        log_path = log_directory / f"{time.time_ns()}_{module.rsplit('.', 1)[-1]}.log"
        command = [self.python, "-u", "-m", module, *map(str, arguments)]
        budget = 1800
        if "--minutes" in command:
            budget = float(command[command.index("--minutes") + 1]) * 60 + 600
        return run_logged(
            command,
            cwd=self.work,
            env=self.environment,
            log_path=log_path,
            timeout=budget,
            label=module.rsplit(".", 1)[-1].replace("_", " ").capitalize(),
            show=lambda line: (
                line.startswith(
                    (
                        "step=",
                        "Experiment:",
                        "Progress:",
                        "Checkpoint:",
                        "Result:",
                        "Preserved previous evaluation:",
                    )
                )
                or "View run" in line
            ),
        )

    def train(self, name, environment, learning, parent=None):
        """Train from scratch unless an explicit parent requests fine-tuning.

        Fine-tuning restores networks and normalization but starts new replay.
        Exact optimizer/replay continuation remains available through the train
        CLI's --resume option without --finetune.
        """
        pointer = self.work / "artifacts/last_experiment.txt"
        pointer.parent.mkdir(exist_ok=True)
        settings = {
            **environment,
            **learning,
            "name": name,
            "result_file": str(pointer),
        }
        if parent is not None:
            settings.update(resume=str(parent), finetune=True)
        self.run("bdxr_classroom.learning.train", *command_arguments(settings))
        run = Path(pointer.read_text().strip())
        if run not in self.runs:
            self.runs.append(run)
        timing = json.loads((run / "metrics/timing.json").read_text())
        requested = settings.get("steps")
        print(
            f"Training result: {timing['step']:,} steps in "
            f"{timing['session_training_seconds']:.0f}s.",
            flush=True,
        )
        if requested is not None and timing["step"] < requested:
            print(
                f"Stopped before the requested {requested:,} steps. "
                "Check the time limit before comparing this run with the example.",
                flush=True,
            )
        return run

    def record(self, run, checkpoint="final", speed=None):
        """Record actual checkpoint inference and return the generated MP4 path."""
        from bdxr_classroom.evaluation.walking_checks import checkpoint_digest

        run = Path(run)
        weights = run / "checkpoints" / f"{checkpoint}.pt"
        digest = checkpoint_digest(weights)
        suffix = "" if speed is None else f"_speed_{speed:g}"
        output = run / "evaluation" / f"eval_{checkpoint}{suffix}"
        options = [] if speed is None else ["--speed", speed]
        self.run(
            "bdxr_classroom.evaluation.evaluate",
            weights,
            "--episodes",
            1,
            "--output",
            output,
            *options,
            *self.presentation_arguments(),
        )
        report = json.loads((output / "evaluation.json").read_text())
        if (
            report.get("checkpoint_sha256") != digest
            or checkpoint_digest(weights) != digest
        ):
            raise RuntimeError(
                "Video/checkpoint identity mismatch; record stable saved weights."
            )
        video = Path(report["video_path"])
        if not video.is_file():
            raise RuntimeError(f"Recording did not produce its video: {video}")
        print(
            f"Result: video from {weights} (step {report['training_step']}, "
            f"sha256 {digest[:12]})",
            flush=True,
        )
        return video

    def learning_curve(self, run):
        self.run("bdxr_classroom.evaluation.plots", run)
        return Path(run) / "metrics/learning_curves.png"
