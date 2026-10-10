"""A small cell magic for executing visible Python in the pinned simulator."""

import atexit
import json
import subprocess

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
        atexit.register(self.close)

    def execute(self, source):
        if self.process.poll() is not None:
            raise RuntimeError(
                "Simulator worker stopped. Rerun the initialization cell."
            )
        self.process.stdin.write(json.dumps({"source": source}) + "\n")
        self.process.stdin.flush()
        for line in self.process.stdout:
            if line.startswith(RESPONSE_PREFIX):
                result = json.loads(line[len(RESPONSE_PREFIX) :])
                print(result["output"], end="")
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
                self.process.wait(timeout=10)
