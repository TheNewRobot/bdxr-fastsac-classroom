"""Summarize measured gait changes from command-evaluation trajectories."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def summarize(rows):
    trials = []
    excluded = []
    for episode in sorted({r["episode"] for r in rows}):
        samples = [r for r in rows if r["episode"] == episode and r["time"] >= 1]
        if len(samples) < 2:
            excluded.append({"episode": episode, "reason": "Fewer than two samples after the initial second"})
            continue
        dt = float(np.median(np.diff([r["time"] for r in samples])))
        contacts = np.array([r["foot_contacts"] for r in samples], dtype=bool)
        clearance = np.array([r["sole_clearance_m"] for r in samples])
        tilt = np.array([r["sole_tilt_rad"] for r in samples])
        flight = ~contacts.any(axis=1)
        intervals = []
        start = None
        for i, flying in enumerate(flight):
            if flying and start is None:
                start = i
            if not flying and start is not None:
                # Ignore flights truncated by the measurement boundary.
                if start > 0:
                    segment = slice(start, i)
                    intervals.append({
                        "duration_s": (i - start) * dt,
                        "both_feet_peak_clearance_m": float(clearance[segment].min(axis=1).max()),
                        "mean_sole_tilt_deg": float(np.degrees(tilt[segment]).mean()),
                    })
                start = None
        foot_landings = []
        for foot in range(2):
            air_samples = 0
            landings = 0
            for contact in contacts[:, foot]:
                if not contact:
                    air_samples += 1
                else:
                    landings += int(air_samples >= math.ceil(.04 / dt - 1e-6))
                    air_samples = 0
            foot_landings.append(landings)
        duration = len(samples) * dt
        clear_intervals = [v for v in intervals
                           if v["duration_s"] >= .079 and v["both_feet_peak_clearance_m"] >= .025]
        trials.append({
            "episode": episode,
            "fell": any(r["fell"] for r in samples),
            "duration_s": duration,
            "mean_speed_m_s": float(np.mean([r["speed"] for r in samples])),
            "mean_torso_height_m": float(np.mean([r["torso_height"] for r in samples]))
                if all("torso_height" in r for r in samples) else None,
            "foot_landings_per_second": sum(foot_landings) / duration,
            "per_foot_landings": foot_landings,
            "sole_clearance_p95_m": float(np.percentile(clearance.max(axis=1), 95)),
            "mean_sole_tilt_deg": float(np.degrees(tilt).mean()),
            "swing_sole_tilt_deg": float(np.degrees(tilt[~contacts]).mean())
                if (~contacts).any() else None,
            "flight_fraction": float(flight.mean()),
            "contact_force_p95": float(np.percentile(
                [r["total_foot_contact_force"] for r in samples], 95))
                if all("total_foot_contact_force" in r for r in samples) else None,
            "complete_contact_free_intervals": intervals,
            "clear_flight_intervals": clear_intervals,
            "mean_clear_flight_duration_s": float(np.mean([v["duration_s"] for v in clear_intervals])) if clear_intervals else 0,
            "mean_clear_flight_peak_m": float(np.mean([v["both_feet_peak_clearance_m"] for v in clear_intervals])) if clear_intervals else 0,
            "mean_clear_flight_tilt_deg": float(np.mean([v["mean_sole_tilt_deg"] for v in clear_intervals])) if clear_intervals else None,
            "mean_flight_duration_s": float(np.mean([v["duration_s"] for v in intervals])) if intervals else 0,
            "mean_flight_peak_clearance_m": float(np.mean([v["both_feet_peak_clearance_m"] for v in intervals])) if intervals else 0,
            "flight_sole_tilt_deg": float(np.degrees(tilt[flight]).mean()) if flight.any() else None,
        })
    return {"trials": trials, "excluded_episodes": excluded, "scope": (
        "Actual contact transitions, sampled collision-sole clearance and sole-plane tilt. "
        "Swing tilt uses each foot only while that foot has no contact, excluding stance samples. "
        "Clear-flight subset requires >=80 ms and >=25 mm simultaneous peak clearance; all intervals are retained. "
        "Contact-free intervals alone do not establish stable landed hops. "
        "Use command acceptance and landed-hop checks separately.")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    output = {}
    for path in sorted(args.evaluation.glob("*_motion.json")):
        output[path.stem.removesuffix("_motion")] = summarize(json.loads(path.read_text()))
    if not output:
        raise ValueError("No motion trajectories found")
    (args.evaluation / "gait_measurements.json").write_text(json.dumps(output, indent=2) + "\n")
    for case, report in output.items():
        values = report["trials"]
        keys = ["foot_landings_per_second", "sole_clearance_p95_m", "mean_torso_height_m",
                "mean_flight_duration_s", "mean_flight_peak_clearance_m", "flight_sole_tilt_deg"]
        print(case, {key: round(float(np.mean([v[key] for v in values if v[key] is not None])), 4)
                     for key in keys if any(v[key] is not None for v in values)})


if __name__ == "__main__":
    main()
