"""Plot measured training logs without a W&B account."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bdxr_classroom.experiments import metrics_path


def plot_command_motion(rows, output, case="forward", yaw_key="imu_yaw_rate"):
    """Show tracking against time without flattening nearly straight XY paths."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), layout="constrained")
    fields = [
        ("cross_track", "Lateral deviation (m)", None),
        ("speed", "Forward speed (m/s)", "command_speed"),
        (yaw_key, "Turning rate (rad/s)", "command_yaw"),
    ]
    episodes = sorted({row["episode"] for row in rows})
    for axis, (field, title, command) in zip(axes, fields):
        for episode in episodes:
            data = [row for row in rows if row["episode"] == episode]
            axis.plot(
                [row["time"] for row in data],
                [row[field] for row in data],
                alpha=0.7,
                linewidth=1.2,
            )
        if episodes:
            first = [row for row in rows if row["episode"] == episodes[0]]
            if command:
                axis.plot(
                    [row["time"] for row in first],
                    [row[command] for row in first],
                    "k--",
                    linewidth=1.4,
                    label="Command",
                )
                axis.legend(fontsize=8)
            else:
                axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="Time (s)")
        axis.grid(alpha=0.2)
    fig.suptitle(f"{case} · measured command tracking", fontsize=12)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return Path(output)


def plot_run(run_dir, through_step=None, output=None):
    run_dir = Path(run_dir)
    rows = [json.loads(s) for s in metrics_path(run_dir).read_text().splitlines()]
    if through_step is not None:
        rows = [r for r in rows if r["step"] <= through_step]
    if not rows:
        raise ValueError("No training measurements have been logged yet")
    x = np.array([r["transitions"] for r in rows]) / 1e6
    has_tracking = "velocity_error" in rows[-1]
    keys = (
        ["episode_return", "episode_seconds", "velocity_error", "yaw_error"]
        if has_tracking
        else ["episode_return", "episode_seconds", "forward_speed"]
    )
    titles = (
        [
            "Episode reward",
            "Episode duration (s)",
            "Planar tracking MAE (m/s)",
            "Yaw tracking MAE (rad/s)",
        ]
        if has_tracking
        else ["Episode reward", "Episode duration (s)", "Mean forward speed (m/s)"]
    )
    # Keep four learning curves readable at ordinary notebook display widths.
    if len(keys) == 4:
        fig, axs = plt.subplots(2, 2, figsize=(8, 5), constrained_layout=True)
        axs = axs.ravel()
    else:
        fig, axs = plt.subplots(1, len(keys), figsize=(14, 3.5), constrained_layout=True)
    for ax, key, title in zip(
        axs,
        keys,
        titles,
    ):
        ax.plot(x, [r.get(key, np.nan) for r in rows], color="#247a56", linewidth=2)
        ax.set(xlabel="Simulated transitions (millions)", title=title)
        ax.grid(alpha=0.2)
    if not has_tracking:
        axs[2].axhline(
            rows[-1]["target_speed"], color="#d58b22", linestyle="--", label="Target"
        )
        axs[2].legend()
    destination = (
        Path(output) if output else metrics_path(run_dir).parent / "learning_curves.png"
    )
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--through-step", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(plot_run(args.run_dir, args.through_step, args.output))
