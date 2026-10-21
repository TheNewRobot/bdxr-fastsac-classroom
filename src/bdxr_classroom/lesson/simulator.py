"""A small cell magic for executing visible Python in the pinned simulator."""

import atexit
import json
import queue
import subprocess
import threading
import time

from .worker import RESPONSE_PREFIX


class SimulatorCells:
    """Own the simulator worker; stop it explicitly when the exploration ends."""

    def __init__(self, python, work, environment):
        self.process = subprocess.Popen(
            [str(python), "-u", "-m", "bdxr_classroom.lesson.worker"],
            cwd=work,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.messages = queue.Queue()

        def read_output():
            try:
                for line in self.process.stdout:
                    self.messages.put(line)
            finally:
                self.messages.put(None)

        self.reader = threading.Thread(target=read_output, daemon=True)
        self.reader.start()
        atexit.register(self.close)

    def execute(self, source, timeout=600, heartbeat=15):
        if self.process.poll() is not None:
            raise RuntimeError(
                "Simulator worker stopped. Rerun the initialization cell."
            )
        self.process.stdin.write(json.dumps({"source": source}) + "\n")
        self.process.stdin.flush()
        started = time.monotonic()
        next_report = started + heartbeat
        while True:
            now = time.monotonic()
            if now - started > timeout:
                self.close()
                raise TimeoutError(
                    "Simulator cell exceeded its time limit. "
                    "Rerun simulator initialization and environment creation."
                )
            if now >= next_report:
                print(
                    f"Simulator cell still running ({now - started:.0f}s)…", flush=True
                )
                next_report = now + heartbeat
            try:
                line = self.messages.get(timeout=min(0.2, heartbeat))
            except queue.Empty:
                continue
            if line is None:
                break
            if line.startswith(RESPONSE_PREFIX):
                result = json.loads(line[len(RESPONSE_PREFIX) :])
                print(result["output"], end="", flush=True)
                if result["error"]:
                    raise RuntimeError(result["error"])
                return
            # Native library startup messages are separate from the cell result.
            if "Error" in line or "error" in line:
                print(line, end="")
        raise RuntimeError(f"Simulator worker exited with code {self.process.wait()}.")

    def register(self, shell):
        """Install %%robot without requiring simulator imports in the Colab kernel."""
        shell.register_magic_function(
            lambda line, cell: self.execute(cell), "cell", "robot"
        )

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        self.reader.join(timeout=2)
        self.process.stdout.close()
