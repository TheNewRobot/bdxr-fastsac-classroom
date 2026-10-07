"""Upload selected real evaluations to an instructor W&B gallery run."""

import argparse
import json
from pathlib import Path

import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="classroom-gallery")
    p.add_argument("--project", default="bdxr-fastsac-classroom")
    p.add_argument("--upload-checkpoints", action="store_true")
    p.add_argument("evaluations", nargs="+")
    args = p.parse_args()
    with wandb.init(project=args.project, name=args.name, job_type="evaluation") as run:
        table = wandb.Table(columns=["experiment", "target_speed", "mean_speed", "falls",
                                    "landed_hops", "training_step"])
        for directory in args.evaluations:
            directory = Path(directory)
            result = json.loads((directory / "evaluation.json").read_text())
            label = directory.parent.name + "/" + directory.name
            table.add_data(label, result["target_speed"], result["mean_speed"],
                           result["fall_fraction"],
                           sum(r.get("landed_hops", 0) for r in result["episodes"]),
                           result["training_step"])
            run.log({f"{label}/video": wandb.Video(str(directory / "rollout.mp4"), format="mp4"),
                     f"{label}/motion": wandb.Image(str(directory / "motion.png"))})
            curve = directory.parent / "learning_curves.png"
            if curve.exists():
                run.log({f"{label}/learning": wandb.Image(str(curve))})
            if args.upload_checkpoints:
                checkpoint = Path(result["checkpoint"])
                artifact = wandb.Artifact(
                    "bdxr-" + directory.parent.name, type="model",
                    description="Selected FastSAC policy/critics/optimizers. Replay is not included; simulation episodes restart.",
                    metadata={"training_step": result["training_step"],
                              "replay_included": False, "evaluation": result})
                artifact.add_file(str(checkpoint), name="checkpoint.pt")
                artifact.add_file(str(directory / "evaluation.json"), name="evaluation.json")
                artifact.add_file(str(directory.parent / "config.json"), name="config.json")
                run.log_artifact(artifact, aliases=["selected", "classroom"])
        run.log({"evaluation_comparison": table})
        out = Path("artifacts")
        out.mkdir(exist_ok=True)
        (out / f"{args.name}_wandb.json").write_text(json.dumps({"url": run.url, "id": run.id}, indent=2))
        print(run.url)


if __name__ == "__main__":
    main()
