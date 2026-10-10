"""Upload selected real evaluations to an instructor W&B gallery run."""

import argparse
import json
from pathlib import Path
import sys
import statistics

import wandb
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.experiments import experiment_root, config_path
from bdxr_classroom.evaluation.plots import plot_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="classroom-gallery")
    p.add_argument("--project", default="bdxr-fastsac-classroom")
    p.add_argument("--upload-checkpoints", action="store_true")
    p.add_argument("evaluations", nargs="+")
    args = p.parse_args()
    with wandb.init(project=args.project, name=args.name, job_type="evaluation") as run:
        table = wandb.Table(
            columns=[
                "experiment",
                "case",
                "mean_speed",
                "falls",
                "speed_mae",
                "yaw_mae",
                "max_cross_track",
                "landed_hops",
                "peak_foot_clearance_m",
                "foot_height_measurement",
            ]
        )
        for directory in args.evaluations:
            directory = Path(directory)
            command_report = (directory / "summary.json").exists()
            report_file = directory / (
                "summary.json" if command_report else "evaluation.json"
            )
            result = json.loads(report_file.read_text())
            checkpoint = Path(result["checkpoint"]).resolve()
            root = experiment_root(checkpoint)
            config = json.loads(config_path(root).read_text())
            training_step = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )["step"]
            label = root.name + "/" + directory.name
            if command_report:
                for case, rows in result["cases"].items():
                    table.add_data(
                        label,
                        case,
                        statistics.mean(r["mean_speed"] for r in rows),
                        sum(r["fell"] for r in rows),
                        statistics.mean(r["speed_mae"] for r in rows),
                        statistics.mean(r["yaw_mae"] for r in rows),
                        max(r["max_cross_track_m"] for r in rows),
                        sum(r.get("landed_hops", 0) for r in rows),
                        statistics.mean(
                            r.get("peak_simultaneous_foot_clearance", 0) for r in rows
                        ),
                        result.get(
                            "foot_height_measurement",
                            "Legacy foot reference-site height",
                        ),
                    )
                    video = root / "videos" / directory.name / f"{case}.mp4"
                    if video.exists():
                        run.log(
                            {
                                f"{label}/{case}/video": wandb.Video(
                                    str(video), format="mp4"
                                )
                            }
                        )
                    plot = directory / f"{case}.png"
                    if plot.exists():
                        run.log({f"{label}/{case}/motion": wandb.Image(str(plot))})
            else:
                table.add_data(
                    label,
                    "rollout",
                    result["mean_speed"],
                    sum(r["fell"] for r in result["episodes"]),
                    None,
                    None,
                    None,
                    sum(r.get("landed_hops", 0) for r in result["episodes"]),
                    statistics.mean(
                        r.get("steady_peak_min_foot_height", 0)
                        for r in result["episodes"]
                    ),
                    result.get(
                        "foot_height_measurement", "Legacy foot reference-site height"
                    ),
                )
                run.log(
                    {
                        f"{label}/video": wandb.Video(
                            str(result.get("video_path") or directory / "rollout.mp4"),
                            format="mp4",
                        ),
                        f"{label}/motion": wandb.Image(str(directory / "motion.png")),
                    }
                )
            curve = directory / "learning_curves.png"
            plot_run(root, training_step, curve)
            run.log({f"{label}/learning": wandb.Image(str(curve))})
            if args.upload_checkpoints:
                artifact = wandb.Artifact(
                    "bdxr-" + config["name"],
                    type="model",
                    description="Selected FastSAC policy/critics/optimizers. Replay is not included; simulation episodes restart.",
                    metadata={
                        "training_step": training_step,
                        "replay_included": False,
                        "evaluation": result,
                    },
                )
                artifact.add_file(str(checkpoint), name="checkpoint.pt")
                artifact.add_file(str(report_file), name=report_file.name)
                artifact.add_file(str(config_path(root)), name="config.json")
                environment = root / "config/environment.json"
                if environment.exists():
                    artifact.add_file(str(environment), name="environment.json")
                run.log_artifact(artifact, aliases=["selected", "classroom"])
        run.log({"evaluation_comparison": table})
        out = Path("artifacts")
        out.mkdir(exist_ok=True)
        (out / f"{args.name}_wandb.json").write_text(
            json.dumps({"url": run.url, "id": run.id}, indent=2)
        )
        print(run.url)


if __name__ == "__main__":
    main()
