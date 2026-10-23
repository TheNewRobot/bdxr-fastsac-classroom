"""Choose between the two student runs using their existing walking checks."""

import json
from pathlib import Path

from bdxr_classroom.evaluation.walking_checks import checkpoint_digest


def choose_walking_policy(runs, assessments, preference="auto"):
    """Summarize exact-checkpoint results without training or evaluating again.

    Auto prefers the baseline if both pass. Missing or stale evidence never
    qualifies a policy; a manual choice must pass the same short assessment.
    """
    if preference not in {"auto", "baseline", "variation"}:
        raise ValueError("Choose auto, baseline, or variation.")
    passing = {}
    print("Existing short walking checks (not full game validation):")
    for name in ("baseline", "variation"):
        run, verdict = runs.get(name), assessments.get(name)
        if run is None or not verdict:
            print(f"  {name.title()}: NOT CHECKED — run its short walking check.")
            continue
        checkpoint = Path(run) / "checkpoints/final.pt"
        if (
            not checkpoint.is_file()
            or Path(verdict.get("checkpoint", "")).resolve() != checkpoint.resolve()
            or verdict.get("checkpoint_sha256") != checkpoint_digest(checkpoint)
        ):
            print(f"  {name.title()}: CHECK OUTDATED — reassess these weights.")
            continue
        summary_path = Path(verdict["evaluation_directory"]) / "summary.json"
        if not summary_path.is_file():
            print(f"  {name.title()}: CHECK MISSING — rerun its short check.")
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("checkpoint_sha256") != verdict["checkpoint_sha256"]:
            print(f"  {name.title()}: CHECK OUTDATED — reassess these weights.")
            continue
        trials = summary["cases"]["forward"]
        speed = sum(trial["mean_speed"] for trial in trials) / len(trials)
        passed = verdict.get("passed") is True
        print(
            f"  {name.title()}: {'PASS' if passed else 'FAIL'}"
            f" — speed {speed:.3f} m/s (requested 0.30 m/s)"
        )
        if passed:
            passing[name] = str(checkpoint)
        else:
            print("    " + "; ".join(verdict.get("failures", [])))
    if not passing:
        print(
            "Neither policy has a current passing check. The game will not start. "
            "For a failed run, try a fresh training run and repeat its short check; "
            "keep the unsuccessful result in your comparison."
        )
        return None
    selected = next(iter(passing)) if preference == "auto" else preference
    if selected not in passing:
        print(
            f"{selected.title()} has no current passing check. Choose auto or "
            f"{', '.join(passing)}; the game will not start with this selection."
        )
        return None
    print(f"Selected: {selected.title()} — passed the short walking check.")
    print("Turning and obstacle capability are not established by this check.")
    return passing[selected]
