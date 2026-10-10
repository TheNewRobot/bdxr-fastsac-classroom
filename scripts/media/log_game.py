"""Upload a recorded course attempt and its actual outcome to W&B."""

import argparse
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    root = args.experiment
    config = json.loads((root / "config/game.json").read_text())
    score = json.loads((root / "evaluation/score.json").read_text())
    with wandb.init(
        project="bdxr-fastsac-classroom",
        name=root.name,
        job_type="course-evaluation",
        config=config,
    ) as run:
        run.summary.update(
            {
                key: score[key]
                for key in ("time", "falls", "finished", "course_checkpoints")
                if key in score
            }
        )
        for path in sorted((root / "videos").glob("gameplay_*.mp4")):
            run.log({path.stem: wandb.Video(str(path), format="mp4")})
        evidence = wandb.Artifact(
            root.name,
            type="course-evaluation",
            metadata={"operator": config["operator"], "course": config["course"]},
        )
        for directory in ("config", "evaluation", "metrics"):
            for path in (root / directory).glob("*.json*"):
                evidence.add_file(str(path), name=f"{directory}/{path.name}")
        run.log_artifact(evidence)
        (root / "evaluation/wandb.json").write_text(
            json.dumps({"url": run.url, "id": run.id}, indent=2)
        )


if __name__ == "__main__":
    main()
