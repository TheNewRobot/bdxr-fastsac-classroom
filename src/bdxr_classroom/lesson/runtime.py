"""Run training and recording jobs while keeping notebook cells small.

This module handles subprocesses and paths. Environment and learning settings
remain explicit dictionaries in the lesson; they map directly to CLI arguments.
"""

import json
import subprocess
import time
from pathlib import Path


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
        with log_path.open("w", buffering=1) as log:
            process = subprocess.Popen(
                command,
                cwd=self.work,
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                for line in process.stdout:
                    log.write(line)
                    if (
                        line.startswith(
                            ("step=", "Experiment:", "forward:", "sequence:")
                        )
                        or "View run" in line
                    ):
                        print(line.rstrip())
                returncode = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
        if returncode:
            raise RuntimeError(
                f"Command failed; full log: {log_path}\n{log_path.read_text()[-6000:]}"
            )
        return log_path

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
        return run

    def record(self, run, checkpoint="final", speed=None):
        """Record actual checkpoint inference and return the generated MP4 path."""
        run = Path(run)
        suffix = "" if speed is None else f"_speed_{speed:g}"
        output = run / "evaluation" / f"eval_{checkpoint}{suffix}"
        options = [] if speed is None else ["--speed", speed]
        self.run(
            "bdxr_classroom.evaluation.evaluate",
            run / "checkpoints" / f"{checkpoint}.pt",
            "--episodes",
            1,
            "--output",
            output,
            *options,
            *self.presentation_arguments(),
        )
        return Path(json.loads((output / "evaluation.json").read_text())["video_path"])

    def learning_curve(self, run):
        self.run("bdxr_classroom.evaluation.plots", run)
        return Path(run) / "metrics/learning_curves.png"
