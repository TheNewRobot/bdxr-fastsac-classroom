"""One learned hopper: advance one phase cycle, then hold its stance cue."""

import dataclasses
import inspect
import io
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import torch
from PIL import Image

from bdxr_classroom.evaluation.command_evaluate import load_agent, set_command
from bdxr_classroom.evaluation.walking_checks import checkpoint_digest
from bdxr_classroom.experiments import create_experiment, write_manifest
from bdxr_classroom.simulation.environment import make_env
from bdxr_classroom.simulation.head_motion import HeadMotionConfig, make_head_motion
from bdxr_classroom.simulation.training_terrain import sampled_sole_clearance


def clear_hop(result):
    """Retain the earlier 25 mm clearance requirement and reject contact flicker."""
    return (
        not result["fell"]
        and result["peak_clearance_m"] >= 0.025
        and result["longest_flight_s"] >= 0.08
        and result["flight_events"] == 1
    )


@dataclasses.dataclass
class JumpCycle:
    """A request changes the phase cue, never the robot's state or leg actions."""

    phase: float = 0.0
    active: bool = False
    elapsed: float = 0.0

    def request(self):
        if self.active:
            return False
        self.active, self.elapsed, self.phase = True, 0.0, 0.0
        return True

    def advance(self, dt):
        if self.active:
            self.elapsed += dt
            self.phase = min(self.elapsed, 1.0)
            if self.elapsed >= 1.0 - 1e-8:
                self.active, self.phase = False, 0.0


class HoppingSession:
    """Keep physics and the same actor running during stance and landing."""

    def __init__(self, checkpoint, run_root="experiments", render=True, seed=263):
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        self.checkpoint = Path(checkpoint).resolve()
        self.digest = checkpoint_digest(self.checkpoint)
        self.agent, self.config = load_agent(self.checkpoint, 1)
        if self.agent.dimensions[0] != 49 or self.config.get("task_version") != "hop":
            raise ValueError("Load a 49-observation hopping policy from this lesson.")
        if self.agent.dimensions[2] != 10:
            raise ValueError("The hopper must control the ten leg actions.")
        parameters = inspect.signature(make_env).parameters
        settings = {k: v for k, v in self.config.items() if k in parameters}
        settings.update(
            num_envs=1, render=render, seed=seed, speed=0,
            command_mode="fixed", training_terrain_height=0,
            initial_jitter=0, course=None, full_link_collisions=True, hopping_practice=True,
        )
        self.env = make_env(**settings)
        self.env.cfg.episode_length_s = 1e9
        self.seed, self.render_enabled = seed, render
        # Physical head and render agree. This expression is not a second policy.
        self.head = make_head_motion(self.env, HeadMotionConfig(
            enabled=True, physical_expression=True, level_gaze_in_render=False,
            max_velocity=0.4, max_acceleration=1.2, response_frequency=4,
        ))
        self.output = create_experiment("hopping_demo", run_root, category="evaluations")
        manifest = {
            "name": "hopping_demo",
            "checkpoint": str(self.checkpoint), "checkpoint_sha256": self.digest,
            "seed": seed, "hardware": torch.cuda.get_device_name(0),
            "single_policy": True, "full_link_collisions": True,
            "control": "One-second phase cue per click; hold stance phase between clicks.",
        }
        (self.output / "config/demo.json").write_text(json.dumps(manifest, indent=2))
        write_manifest(self.output, manifest, self.env.cfg)
        self.log = (self.output / "metrics/hops.jsonl").open("w", buffering=1)
        self.motion_log = (self.output / "metrics/motion.jsonl").open("w", buffering=1)
        self.writer, self.recording = None, False
        self.record_count, self.steps = 0, 0
        self.frame = None
        self.results = []
        self.reset()

    def reset(self):
        self.env.reset(seed=self.seed)
        if self.head:
            self.head.reset(self.seed)
        self.cycle = JumpCycle()
        self.state = "settling"
        self.message = "Let the robot settle on both feet."
        self.supported_seconds = self.wait_seconds = 0.0
        self.current = None
        self.elapsed = 0.0

    def jump(self):
        if self.state != "ready":
            return False
        self.cycle.request()
        self.state, self.message = "jumping", "One jump requested. Watch the landing."
        self.current = {
            "request": len(self.results) + 1, "peak_clearance_m": 0.0,
            "longest_flight_s": 0.0, "flight_events": 0, "fell": False,
        }
        self.flight_seconds = 0.0
        self.supported_seconds = self.wait_seconds = 0.0
        return True

    def _finish(self, passed, reason):
        if self.current is not None:
            row = {**self.current, "passed": passed, "reason": reason}
            self.results.append(row)
            self.log.write(json.dumps(row) + "\n")
            self.current = None
        self.message = reason

    def step(self):
        if self.state == "failed":
            return self.status()
        dt = self.env.step_dt
        obs = set_command(self.env, 0, 0)["actor"]
        angle = 2 * math.pi * self.cycle.phase
        obs[:, -2:] = torch.tensor(
            [math.sin(angle), math.cos(angle)], device=self.env.device
        )
        if self.head:
            obs = self.head.policy_observations(obs)
        with torch.no_grad():
            action = self.agent.act(obs, deterministic=True)
        if self.head:
            self.head.step(dt, 0, 0, "hop")
        if not torch.isfinite(action).all():
            self.state = "failed"
            self._finish(False, "Invalid policy action. Reset to try again.")
            return self.status()
        _, _, terminated, _, _ = self.env.step(action)
        self.cycle.advance(dt)
        self.elapsed += dt
        self.steps += 1
        robot = self.env.scene["robot"].data
        contact = (self.env.scene["feet_ground_contact"].data.found > 0).flatten()
        clearance = float(sampled_sole_clearance(self.env).min())
        supported = bool(contact.all())
        flying = not bool(contact.any())
        upright = float(-robot.projected_gravity_b[0, 2])
        finite = bool(torch.isfinite(self.env.sim.data.qpos).all())
        if self.state != "ready" or self.steps % 50 == 0:
            self.motion_log.write(json.dumps({
                "time": self.elapsed, "state": self.state, "phase": self.cycle.phase,
                "clearance_m": clearance, "contacts": contact.cpu().tolist(),
                "upright": upright, "terminated": bool(terminated.any()),
                "base_height": float(robot.root_link_pos_w[0, 2]),
            }) + "\n")
        if self.current is not None:
            self.current["peak_clearance_m"] = max(
                self.current["peak_clearance_m"], clearance
            )
            self.flight_seconds = self.flight_seconds + dt if flying else 0.0
            self.current["longest_flight_s"] = max(
                self.current["longest_flight_s"], self.flight_seconds
            )
            # Ignore a one-frame contact flicker; report distinct >=40 ms flights.
            if flying and abs(self.flight_seconds - 0.04) < 1e-6:
                self.current["flight_events"] += 1
        if bool(terminated.any()) or not finite or upright < 0.5:
            if self.current is not None:
                self.current["fell"] = True
            self.state = "failed"
            self._finish(False, "The robot fell. Reset to try again.")
        else:
            calm = float(robot.root_link_ang_vel_b[0].norm()) < 0.6
            self.supported_seconds = (
                self.supported_seconds + dt if supported and calm else 0.0
            )
            if not self.cycle.active and self.state != "ready":
                self.state = "settling"
                self.wait_seconds += dt
                if self.supported_seconds >= 0.6:
                    if self.current is not None:
                        passed = clear_hop(self.current)
                        self._finish(
                            passed,
                            "Landed. Ready for another jump." if passed
                            else "Settled, but this was not one clear hop. Review the result.",
                        )
                    else:
                        self.message = "Ready. Click Jump once."
                    self.state = "ready"
                elif self.wait_seconds > 8:
                    self._finish(False, "Did not settle safely. Reset to try again.")
                    self.state = "failed"
        if self.render_enabled and self.steps % 2 == 0:
            self.render()
        return self.status()

    def set_recording(self, enabled):
        self.recording = bool(enabled)
        if not self.recording and self.writer:
            self.writer.close()
            self.writer = None

    def render(self):
        self.frame = self.head.render() if self.head else self.env.render()
        if self.recording:
            if self.writer is None:
                self.record_count += 1
                self.writer = imageio.get_writer(
                    self.output / f"videos/hopping_{self.record_count:03d}.mp4",
                    fps=25, codec="libx264", quality=7, macro_block_size=16,
                )
            self.writer.append_data(self.frame)
        return self.frame

    def jpeg(self):
        if self.frame is None:
            self.render()
        stream = io.BytesIO()
        Image.fromarray(self.frame).save(stream, format="JPEG", quality=85)
        return stream.getvalue()

    def status(self):
        return {
            "ready": True, "state": self.state, "message": self.message,
            "can_jump": self.state == "ready", "recording": self.recording,
            "time": round(self.elapsed, 2), "results": self.results,
            "experiment": str(self.output), "checkpoint": str(self.checkpoint),
        }

    def close(self):
        self.set_recording(False)
        self.log.close()
        self.motion_log.close()
        self.env.close()
