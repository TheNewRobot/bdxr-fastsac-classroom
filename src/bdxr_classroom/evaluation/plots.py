"""Plot measured training logs without a W&B account."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from bdxr_classroom.experiments import metrics_path


def plot_run(run_dir, through_step=None, output=None):
    run_dir = Path(run_dir)
    rows = [json.loads(s) for s in metrics_path(run_dir).read_text().splitlines()]
    if through_step is not None:
        rows = [r for r in rows if r["step"] <= through_step]
    if not rows:
        raise ValueError("No training measurements have been logged yet")
    x = np.array([r["transitions"] for r in rows]) / 1e6
    has_tracking = "velocity_error" in rows[-1]
    keys = ["episode_return", "episode_seconds", "velocity_error", "yaw_error"] if has_tracking else [
        "episode_return", "episode_seconds", "forward_speed"]
    titles = ["Episode reward", "Episode duration (s)", "Planar tracking MAE (m/s)", "Yaw tracking MAE (rad/s)"] if has_tracking else [
        "Episode reward", "Episode duration (s)", "Mean forward speed (m/s)"]
    fig, axs = plt.subplots(1, len(keys), figsize=(14, 3.5), constrained_layout=True)
    for ax, key, title in zip(
        axs, keys, titles,
    ):
        ax.plot(x, [r.get(key, np.nan) for r in rows], color="#247a56", linewidth=2)
        ax.set(xlabel="Simulated transitions (millions)", title=title)
        ax.grid(alpha=0.2)
    if not has_tracking:
        axs[2].axhline(rows[-1]["target_speed"], color="#d58b22", linestyle="--", label="Target")
        axs[2].legend()
    destination = Path(output) if output else metrics_path(run_dir).parent / "learning_curves.png"
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
