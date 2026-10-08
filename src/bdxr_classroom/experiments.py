"""Timestamped local artifacts, separate from the curated teaching assets."""

import dataclasses
import datetime as dt
import json
import re
import subprocess
import functools
import time
import zipfile
from pathlib import Path


def experiment_root(path):
    path = Path(path)
    if path.is_file() or path.suffix == ".pt":
        path = path.parent
    return path.parent if path.name == "checkpoints" else path


def create_experiment(name, root="experiments"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Experiment names may contain letters, numbers, underscores and hyphens")
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S_%f")
    path = Path(root) / f"{timestamp}_{name}"
    path.mkdir(parents=True, exist_ok=False)
    for folder in ["config", "checkpoints", "metrics", "videos", "evaluation"]:
        (path / folder).mkdir()
    return path


def jsonable(value):
    """Record resolved dataclasses and named callables without executing them."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, functools.partial):
        return {"callable": jsonable(value.func), "args": jsonable(value.args),
                "keywords": jsonable(value.keywords)}
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)


def write_manifest(out, config, env_config):
    source = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source, stderr=subprocess.DEVNULL, text=True).strip())
    except subprocess.CalledProcessError:
        revision, dirty = None, None
    (out / "config/environment.json").write_text(json.dumps(jsonable(env_config), indent=2))
    pins = source / "configs/upstream_revisions.json"
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "name": config["name"], "source_revision": revision, "source_dirty": dirty,
        "parent_checkpoint": config.get("resume"), "hardware": config["hardware"],
        "seed": config["seed"], "upstream": json.loads(pins.read_text()) if pins.exists() else {},
        "layout_version": 2,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    capture_source(out)


def capture_source(out):
    """Preserve the exact Python source even when an experiment starts mid-edit."""
    source = Path(__file__).resolve().parents[2]
    path = out / "config" / f"source_{time.time_ns()}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for pattern in ["src/**/*.py", "src/**/*.html", "configs/*.json", "requirements*.txt"]:
            for file in source.glob(pattern):
                archive.write(file, file.relative_to(source))
    return path


def config_path(root):
    root = experiment_root(root)
    return root / "config/training.json" if (root / "config").exists() else root / "config.json"


def metrics_path(root):
    root = experiment_root(root)
    return root / "metrics/training.jsonl" if (root / "metrics").exists() else root / "metrics.jsonl"
