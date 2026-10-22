"""Bounded subprocess execution with complete logs and periodic progress."""

import html
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path


def notebook_progress():
    """Use one live status area in a notebook; keep terminal output unchanged."""
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except ImportError:
        return None
    shell = get_ipython()
    if shell is None or not hasattr(shell, "kernel"):
        return None
    handle = display(HTML("<pre>Starting training…</pre>"), display_id=True)
    latest_step = ""

    def update(message):
        nonlocal latest_step
        if message.startswith("step="):
            latest_step = message
            text = message
        else:
            text = f"{latest_step}\n{message}".strip()
        handle.update(
            HTML(
                '<pre style="white-space:pre-wrap;overflow-wrap:anywhere">'
                + html.escape(text)
                + "</pre>"
            )
        )

    return update


def run_logged(
    command,
    *,
    cwd,
    log_path,
    env=None,
    timeout=1800,
    heartbeat=15,
    show=None,
    label="Working",
    progress=None,
):
    """Stream selected output without mistaking a quiet process for completion."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"{label}. Log: {log_path}", flush=True)
    process = subprocess.Popen(
        list(map(str, command)),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=os.name == "posix",
    )
    messages = queue.Queue()

    def read_output():
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    next_report = started + heartbeat
    try:
        with log_path.open("w", buffering=1) as log:
            ended = False
            while not ended:
                now = time.monotonic()
                if now - started > timeout:
                    raise TimeoutError(
                        f"{label} exceeded {timeout:g}s. Full log: {log_path}"
                    )
                if now >= next_report:
                    message = f"{label}: still running ({now - started:.0f}s)."
                    if progress:
                        progress(message)
                    else:
                        print(message, flush=True)
                    next_report = now + heartbeat
                try:
                    line = messages.get(timeout=min(0.2, heartbeat))
                except queue.Empty:
                    continue
                if line is None:
                    ended = True
                else:
                    log.write(line)
                    if show is None or show(line):
                        if progress and line.startswith("step="):
                            progress(line.rstrip())
                        else:
                            print(line.rstrip(), flush=True)
            returncode = process.wait(
                timeout=max(1, timeout - (time.monotonic() - started))
            )
        if returncode:
            raise RuntimeError(
                f"{label} failed (exit {returncode}). Full log: {log_path}\n"
                + log_path.read_text()[-6000:]
            )
    except Exception:
        if progress:
            progress(f"{label}: stopped; see the error below and the full log.")
        raise
    finally:
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=10)
        reader.join(timeout=2)
        process.stdout.close()
    message = f"{label}: completed in {time.monotonic() - started:.0f}s."
    if progress:
        progress(message)
    else:
        print(message, flush=True)
    return log_path
