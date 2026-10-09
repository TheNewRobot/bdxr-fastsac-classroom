"""Compose measured policy videos and a compact comparison figure for the lesson."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdxr_classroom.experiments import create_experiment, experiment_root, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", nargs=2, action="append", required=True, metavar=("LABEL", "DIRECTORY"))
    parser.add_argument("--case", default="forward")
    parser.add_argument("--gait-metrics", action="store_true",
                        help="Plot measured step cadence and collision-sole clearance")
    parser.add_argument("--hop-metrics", action="store_true",
                        help="Plot clear-flight duration, bilateral clearance, and sampled landing force")
    parser.add_argument("--playback-speed", type=float, choices=[0.5, 1.0], default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= len(args.evaluation) <= 4:
        parser.error("Provide two to four evaluations")
    if args.gait_metrics and args.hop_metrics:
        parser.error("Choose either gait metrics or hop metrics")
    records, readers = [], []
    for label, directory in args.evaluation:
        directory = Path(directory)
        report = json.loads((directory / "summary.json").read_text())
        trials = report["cases"][args.case]
        if args.gait_metrics or args.hop_metrics:
            measured = json.loads((directory / "gait_measurements.json").read_text())[args.case]["trials"]
            by_episode = {trial["episode"]: trial for trial in measured}
            if set(by_episode) != {trial["episode"] for trial in trials}:
                raise ValueError("Gait measurements must cover every reported trial")
            trials = [dict(trial,
                           foot_landings_per_second=by_episode[trial["episode"]]["foot_landings_per_second"],
                           sole_clearance_p95_m=by_episode[trial["episode"]]["sole_clearance_p95_m"])
                      for trial in trials]
            if args.hop_metrics:
                hop_keys = ["mean_clear_flight_duration_s", "mean_clear_flight_peak_m",
                            "mean_clear_landing_peak_force_n"]
                for trial in trials:
                    for key in hop_keys:
                        value = by_episode[trial["episode"]].get(key)
                        if value is None or not np.isfinite(value):
                            raise ValueError(f"Missing finite {key}; rerun measure_gait.py")
                        trial[key] = value
        root = experiment_root(report["checkpoint"])
        video = root / "videos" / directory.name / f"{args.case}.mp4"
        records.append({"label": label, "evaluation": str(directory.resolve()),
                        "checkpoint": report["checkpoint"], "trials": trials})
        readers.append(imageio.get_reader(video))
    output = create_experiment("policy_comparison")
    write_manifest(output, {"name": "policy_comparison", "hardware": "CPU media composition", "seed": 1000},
                   {"kind": "Recorded evaluation comparison; no new simulation",
                    "evaluations": [r["evaluation"] for r in records]})
    (output / "config/comparison.json").write_text(json.dumps(
        {"case": args.case, "destination": str(args.output), "sources": records,
         "playback_speed": args.playback_speed,
         "metrics": "hop" if args.hop_metrics else "gait" if args.gait_metrics else "walking"}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#266052", "#727d7b", "#317bba", "#bc7825"]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    counts = [reader.count_frames() for reader in readers]
    fps = readers[0].get_meta_data()["fps"]
    if any(abs(reader.get_meta_data()["fps"] - fps) > .01 for reader in readers):
        raise ValueError("Comparison recordings must have the same frame rate")
    columns = 3 if len(readers) == 3 else 2
    height = 640 if len(readers) == 4 else 320
    try:
        with imageio.get_writer(args.output, fps=fps * args.playback_speed, codec="libx264", quality=7) as writer:
            last_frames = [None] * len(readers)
            for frame_index in range(max(counts)):
                canvas = Image.new("RGB", (480 * columns, height), "#edf1e9")
                draw = ImageDraw.Draw(canvas)
                for index, (reader, record) in enumerate(zip(readers, records)):
                    ended = frame_index >= counts[index]
                    if not ended:
                        last_frames[index] = Image.fromarray(reader.get_data(frame_index)).resize((480, 320))
                    x, y = index % columns * 480, index // columns * 320
                    canvas.paste(last_frames[index], (x, y))
                    draw.rectangle((x, y, x + 480, y + 31), fill=colors[index])
                    falls = sum(t["fell"] for t in record["trials"])
                    title = f"{record['label']} | {args.case} | falls {falls}/{len(record['trials'])}"
                    if args.playback_speed != 1:
                        title += f" | {args.playback_speed:g}x"
                    draw.text((x + 9, y + 5), title, font=font, fill="white")
                    if ended:
                        draw.text((x + 9, y + 290), "Recorded trial ended early", font=font, fill="red")
                writer.append_data(np.asarray(canvas))
    finally:
        for reader in readers:
            reader.close()
    metrics = [("action_change_rms", "Action-change RMS", "normalized actions"),
               ("swing_height_p95", "Foot swing (95th percentile)", "m"),
               ("max_cross_track_m", "Maximum lateral deviation", "m")]
    if args.gait_metrics:
        metrics[:2] = [
            ("foot_landings_per_second", "Step cadence", "foot landings per second"),
            ("sole_clearance_p95_m", "Sole clearance (95th percentile)", "m")]
    if args.hop_metrics:
        metrics = [
            ("mean_clear_flight_duration_s", "Clear-flight duration", "s"),
            ("mean_clear_flight_peak_m", "Bilateral peak sole clearance", "m"),
            ("mean_clear_landing_peak_force_n", "Sampled landing peak", "N; first 80 ms after contact"),
        ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for axis, (key, title, units) in zip(axes, metrics):
        groups = [[trial[key] for trial in record["trials"]] for record in records]
        axis.bar(range(len(records)), [statistics.mean(g) for g in groups],
                 yerr=[statistics.pstdev(g) for g in groups], color=colors[:len(records)], capsize=4)
        axis.set_xticks(range(len(records)), [r["label"] for r in records], rotation=20, ha="right")
        axis.set(title=title, ylabel=units)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{args.case.capitalize()} command · variation across starting states, not training seeds")
    fig.savefig(args.output.with_suffix(".png"), dpi=150)
    plt.close(fig)
    (output / "evaluation/outputs.json").write_text(json.dumps(
        {"video": str(args.output), "figure": str(args.output.with_suffix(".png"))}, indent=2))
    print(f"Comparison: {args.output}\nProvenance: {output}")


if __name__ == "__main__":
    main()
