"""Screen a walking/hopping pair before loading it into the classroom game."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    from bdxr_classroom.experiments import resolve_experiment_path
    with resolve_experiment_path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def history_verdict(records, walking_hash, hopping_hash):
    matching = [record for record in records
                if record.get("walking_sha256") == walking_hash
                and record.get("hopping_sha256") == hopping_hash
                and record.get("weights_unchanged") is True]
    completed = False
    failure = False
    evaluation_count = 0
    for record in matching:
        valid = {
            scope: test for scope, test in record.get("tests", {}).items()
            if test.get("valid_evaluation", record.get("valid_evaluation", True))
            and isinstance(test.get("passed"), bool)
        }
        evaluation_count += bool(valid)
        failure |= any(not test["passed"] for test in valid.values())
        completed |= all(valid.get(scope, {}).get("passed") is True
                         for scope in ("flat", "on_step"))
    return {"passed": completed and not failure, "evaluation_count": evaluation_count,
            "recorded_failure": failure,
            "walking_sha256": walking_hash, "hopping_sha256": hopping_hash,
            "scope": "Repeated flat/on-step handovers; full course traversal remains separate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("walking", type=Path)
    parser.add_argument("hopping", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    history = args.output / "pair_history"
    history.mkdir(exist_ok=True)
    stamp = str(time.time_ns())
    walking_hash, hopping_hash = digest(args.walking), digest(args.hopping)
    records = [json.loads(path.read_text()) for path in history.glob("*_pair.json")]
    previous = history_verdict(records, walking_hash, hopping_hash)
    if previous["recorded_failure"]:
        (args.output / "pair_selection.json").write_text(json.dumps(previous, indent=2))
        print("These exact weights have a recorded handover failure; retained in pair_history.")
        raise SystemExit(1)
    record = {"walking": str(args.walking), "hopping": str(args.hopping),
              "walking_sha256": walking_hash, "hopping_sha256": hopping_hash,
              "evaluated_at_ns": int(stamp), "tests": {}}
    validator = Path(__file__).with_name("validate_transitions.py")
    for scope, options in (
        ("flat", []),
        ("on_step", ["--start-x", "4.8", "--start-height", "0.02", "--speed", "0"]),
    ):
        report = history / f"{stamp}_{scope}.json"
        log = history / f"{stamp}_{scope}.log"
        with log.open("w") as stream:
            process = subprocess.run(
                [sys.executable, str(validator), str(args.walking), str(args.hopping),
                 "--result-file", str(report), *options],
                stdout=stream, stderr=subprocess.STDOUT)
        result = json.loads(report.read_text()) if report.exists() else {}
        passed = process.returncode == 0 and result.get("passed") is True
        valid = isinstance(result.get("passed"), bool) and isinstance(result.get("reports"), list)
        record["tests"][scope] = {"passed": passed, "valid_evaluation": valid, "report": str(report),
                                 "log": str(log), "returncode": process.returncode}
        label = ("Passed" if passed else "Needs work") if valid else "Evaluation error; inspect log"
        print(f"{scope}: {label}; {report}", flush=True)
    record["valid_evaluation"] = all(test["valid_evaluation"] for test in record["tests"].values())
    record["weights_unchanged"] = (
        digest(args.walking) == walking_hash and digest(args.hopping) == hopping_hash)
    (history / f"{stamp}_pair.json").write_text(json.dumps(record, indent=2))
    records = [json.loads(path.read_text()) for path in history.glob("*_pair.json")]
    verdict = history_verdict(records, walking_hash, hopping_hash)
    (args.output / "pair_selection.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)
    raise SystemExit(0 if verdict["passed"] else (1 if record["valid_evaluation"] else 2))


if __name__ == "__main__":
    main()
