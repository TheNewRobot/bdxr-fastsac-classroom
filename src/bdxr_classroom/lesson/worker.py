"""Run notebook simulator cells in the pinned Python environment.

The notebook kernel can have a different Python/CUDA stack. Keeping a single
worker preserves variables between cells without modifying that kernel's imports.
Only locally authored notebook code is sent to this worker.
"""

import contextlib
import io
import json
import sys
import traceback

RESPONSE_PREFIX = "__BDXR_CELL_RESULT__"


def main():
    namespace = {"__name__": "__robot_notebook__"}
    for line in sys.stdin:
        request = json.loads(line)
        output = io.StringIO()
        error = None
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                exec(compile(request["source"], "<robot-cell>", "exec"), namespace)
            except Exception:
                error = traceback.format_exc()
        print(
            RESPONSE_PREFIX + json.dumps({"output": output.getvalue(), "error": error}),
            flush=True,
        )


if __name__ == "__main__":
    main()
