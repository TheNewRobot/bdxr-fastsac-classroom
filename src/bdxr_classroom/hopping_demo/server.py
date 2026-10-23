"""Serve the hopping-only panel through the notebook's local port proxy."""

import argparse
import json
import queue
import signal
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from bdxr_classroom.hopping_demo.session import HoppingSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--run-root", default="experiments")
    args = parser.parse_args()
    commands = queue.Queue(maxsize=32)
    shared = {"status": {"ready": False, "message": "Loading your hopper…"}, "frame": None}
    stopping = threading.Event()

    def worker():
        session = None
        try:
            session = HoppingSession(args.checkpoint, args.run_root)
            while not stopping.is_set():
                started = time.monotonic()
                while not commands.empty():
                    command = commands.get_nowait()
                    if command["action"] == "jump":
                        session.jump()
                    elif command["action"] == "reset":
                        session.reset()
                    elif command["action"] == "record":
                        session.set_recording(command.get("enabled") is True)
                session.step()
                shared["status"] = session.status()
                if session.steps % 2 == 0 or shared["frame"] is None:
                    shared["frame"] = session.jpeg()
                time.sleep(max(0, session.env.step_dt - (time.monotonic() - started)))
        except Exception as error:
            traceback.print_exc()
            shared["status"] = {"ready": False, "error": str(error), "message": str(error)}
        finally:
            if session:
                session.close()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlparse(self.path).path.rstrip("/")
            if path.endswith("/api/state"):
                self.respond(json.dumps(shared["status"]).encode(), "application/json")
            elif path.endswith("/api/frame"):
                self.respond(shared["frame"] or b"", "image/jpeg",
                             200 if shared["frame"] else 204)
            else:
                self.respond(Path(__file__).with_name("panel.html").read_bytes(), "text/html")

        def do_POST(self):
            try:
                if not urlparse(self.path).path.endswith("/api/command"):
                    raise ValueError("Unknown endpoint")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError("Invalid command size")
                command = json.loads(self.rfile.read(length))
                if command.get("action") not in {"jump", "reset", "record"}:
                    raise ValueError("Choose Jump, Reset, or Record")
                commands.put_nowait(command)
                self.respond(b'{"accepted":true}', "application/json")
            except (ValueError, queue.Full):
                self.respond(b'{"error":"Invalid or busy request"}', "application/json", 400)

        def log_message(self, *args):
            pass

    thread = threading.Thread(target=worker, daemon=True)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.timeout = 0.2
    thread.start()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopping.set())
    try:
        while not stopping.is_set():
            server.handle_request()
    finally:
        stopping.set()
        server.server_close()
        thread.join(timeout=30)


if __name__ == "__main__":
    main()
