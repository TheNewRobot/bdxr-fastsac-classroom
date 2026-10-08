"""Local HTTP game panel, also usable through Colab's authenticated port proxy."""

import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import traceback
import signal

from .game import GameSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walking", required=True)
    parser.add_argument("--hopping")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run-root", default="experiments")
    args = parser.parse_args()
    commands = queue.Queue()
    shared = {"status": {"message": "Loading the robot and policies…", "ready": False}, "frame": None}
    stopping = threading.Event()

    def worker():
        game = None
        render_requested = True
        last_frame = None
        try:
            game = GameSession(args.walking, args.hopping, run_root=args.run_root)
            while not stopping.is_set():
                start = time.monotonic()
                while not commands.empty():
                    command = commands.get_nowait()
                    try:
                        action = command.get("action", "drive")
                        if action == "load":
                            walking = Path(command["walking"])
                            hopping = Path(command["hopping"]) if command.get("hopping") else None
                            if not walking.is_file() or (hopping and not hopping.is_file()):
                                raise ValueError("Checkpoint path does not exist")
                            game.close()
                            game = None
                            shared["status"] = {"message": "Loading selected policies…", "ready": False}
                            game = GameSession(walking, hopping, run_root=args.run_root)
                            render_requested = True
                        elif game is None:
                            raise ValueError("Load valid policies to start the simulator")
                        elif action == "new":
                            game.new_game()
                            render_requested = True
                        elif action == "pause":
                            game.pause()
                        elif action == "camera":
                            import mujoco
                            camera = game.env._offline_renderer._cam
                            if command.get("view") == "overview":
                                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                                camera.lookat[:] = [6, 0, 0]
                                camera.distance, camera.azimuth, camera.elevation = 14, 90, -65
                            else:
                                camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                                camera.trackbodyid = game.env.scene["robot"].indexing.root_body_id
                                camera.distance, camera.azimuth, camera.elevation = 2, 140, -18
                            render_requested = True
                        else:
                            game.command(command.get("speed", 0), command.get("yaw", 0),
                                         command.get("mode", "walk"))
                    except Exception as error:
                        shared["status"] = dict(game.status() if game else {}, ready=bool(game), error=str(error))
                        if game:
                            game.message = str(error)
                if game is None:
                    time.sleep(.1)
                    continue
                game.step()
                if render_requested or game.last_frame is None:
                    game.render()
                    render_requested = False
                shared["status"] = dict(game.status(), ready=True)
                if game.last_frame is not None and game.last_frame is not last_frame:
                    shared["frame"] = game.jpeg()
                    last_frame = game.last_frame
                time.sleep(max(0, game.env.step_dt - (time.monotonic() - start)))
        except Exception as error:
            traceback.print_exc()
            shared["status"] = {"ready": False, "error": str(error), "message": "Game stopped; inspect the server log."}
        finally:
            if game is not None:
                game.close()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            route = self.path.split("?")[0].rstrip("/")
            if route.endswith("/api/state"):
                self.reply(json.dumps(shared["status"]).encode(), "application/json")
            elif route.endswith("/api/checkpoints"):
                files = sorted(str(p.resolve()) for p in Path(args.run_root).glob("*/checkpoints/*.pt")
                               if p.name != "resume.pt")
                self.reply(json.dumps(files).encode(), "application/json")
            elif route.endswith("/api/frame"):
                frame = shared["frame"]
                self.reply(frame or b"", "image/jpeg", 200 if frame else 503)
            else:
                self.reply(Path(__file__).with_name("web").joinpath("game.html").read_bytes(), "text/html")

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 16384:
                    raise ValueError("Invalid command size")
                command = json.loads(self.rfile.read(length))
                if not isinstance(command, dict):
                    raise ValueError("Command must be an object")
                commands.put(command)
                self.reply(b'{"queued":true}', "application/json")
            except (ValueError, json.JSONDecodeError) as error:
                self.reply(json.dumps({"error": str(error)}).encode(), "application/json", 400)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    def shutdown(*_):
        stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    print(f"Game panel listening on localhost:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        server.server_close()
        thread.join(timeout=30)


if __name__ == "__main__":
    main()
