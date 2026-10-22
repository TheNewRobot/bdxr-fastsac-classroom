"""Atomic downloadable backups of training and gameplay evidence."""

import zipfile
from pathlib import Path


def export_results(work, runs, include_replay=True):
    """Include all curriculum stages and game/transition evidence exactly once.

    A failed archive write leaves the previous completed backup untouched.
    W&B's local cache is excluded; actual metrics/configurations stay included.
    """
    work = Path(work).resolve()
    filename = (
        "my_bdxr_training_backup.zip" if include_replay else "my_bdxr_results.zip"
    )
    destination = work / filename
    temporary = destination.with_suffix(".tmp.zip")
    folders = [Path(run).resolve() for run in runs if run is not None]
    for pattern in (
        "*_command_walking_*recovery",
        "*_desert_droid_game",
        "*_transition_validation*",
    ):
        folders.extend((work / "experiments").rglob(pattern))
    folders = list(
        dict.fromkeys(folder.resolve() for folder in folders if folder.is_dir())
    )
    folder_set = set(folders)
    folders = [
        folder
        for folder in folders
        if not any(parent in folder_set for parent in folder.parents)
    ]
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        # Include enough setup evidence to diagnose a different Colab runtime.
        for name in (
            ".lesson_release.json",
            "runtime_info.json",
            "setup.log",
            "environment_settings.json",
            "learning_settings.json",
        ):
            file = work / name
            if file.is_file():
                archive.write(file, file.relative_to(work))
        logs = work / "artifacts/logs"
        if logs.exists():
            for file in logs.glob("*.log"):
                archive.write(file, file.relative_to(work))
        for folder in folders:
            print("Saving", folder.name, flush=True)
            for file in folder.rglob("*"):
                if (
                    file.is_file()
                    and "wandb" not in file.parts
                    and (include_replay or file.name != "resume.pt")
                ):
                    archive.write(file, file.relative_to(work))
    temporary.replace(destination)
    return destination
