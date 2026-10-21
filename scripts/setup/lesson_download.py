"""Standard-library bootstrap, included verbatim in the notebook's setup cell.

Download one immutable release archive without querying Colab's browser. Files
are verified and staged before installation; student outputs are never removed.
"""

import hashlib
import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


def install_lesson(url, sha256, work, timeout=120, attempts=2):
    """Install a verified archive, with bounded requests and visible progress."""
    if timeout <= 0 or attempts < 1:
        raise ValueError("Download timeout and attempts must be positive.")
    work = Path(work)
    marker = work / ".lesson_release.json"
    if marker.is_file():
        try:
            installed = json.loads(marker.read_text())
        except (ValueError, OSError):
            installed = {}
        if installed.get("sha256") not in (None, sha256):
            raise RuntimeError(
                "This folder contains a different lesson release. "
                "Choose a new WORK folder to preserve your existing experiments."
            )
        if installed.get("files") and all(
            (work / name).is_file() for name in installed["files"]
        ):
            print(f"Lesson files already ready: {work}", flush=True)
            return
    work.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bdxr-download-", dir=work.parent) as tmp:
        archive = Path(tmp) / "lesson.zip"
        for attempt in range(1, attempts + 1):
            start = time.monotonic()
            print(
                f"Downloading lesson files (attempt {attempt}/{attempts})…", flush=True
            )
            digest = hashlib.sha256()
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "BDXR-Lesson"}
                )
                with urllib.request.urlopen(
                    request, timeout=min(20, timeout)
                ) as response:
                    with archive.open("wb") as target:
                        received, reported = 0, -1
                        while chunk := response.read(256 * 1024):
                            target.write(chunk)
                            digest.update(chunk)
                            received += len(chunk)
                            megabytes = received // (1024 * 1024)
                            if megabytes != reported:
                                print(f"  Received {megabytes} MB", flush=True)
                                reported = megabytes
                            if time.monotonic() - start > timeout:
                                raise TimeoutError(
                                    "Lesson download exceeded its time limit."
                                )
                if digest.hexdigest() != sha256:
                    raise ValueError(
                        "Lesson download failed its SHA-256 integrity check."
                    )
                break
            except (OSError, urllib.error.URLError, TimeoutError, ValueError) as error:
                print(f"Download did not complete: {error}", flush=True)
                if attempt == attempts:
                    raise RuntimeError(
                        "Could not download verified lesson files. Check internet access, "
                        "then rerun this cell. No manual ZIP upload is required."
                    ) from error
        stage = Path(tmp) / "verified"
        stage.mkdir()
        print("Checking and unpacking the lesson archive…", flush=True)
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)) or any(
                not (stage / name).resolve().is_relative_to(stage.resolve())
                for name in names
            ):
                raise RuntimeError(
                    "The lesson archive contains invalid or duplicate paths."
                )
            if bundle.testzip() is not None:
                raise RuntimeError("The lesson archive is damaged.")
            bundle.extractall(stage)
        if not (stage / "scripts/setup/setup.py").is_file():
            raise RuntimeError("The release is missing its setup program.")
        work.mkdir(parents=True, exist_ok=True)
        files = [name for name in names if (stage / name).is_file()]
        for name in files:
            target = work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stage / name), target)
        pending = work / ".lesson_release.pending"
        pending.write_text(json.dumps({"sha256": sha256, "url": url, "files": files}))
        pending.replace(marker)
    print(f"Lesson files ready: {work}", flush=True)
