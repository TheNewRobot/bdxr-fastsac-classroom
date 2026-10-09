"""Playable low-level policy validation; position is used only for scoring."""

import dataclasses
import io
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from .command_evaluate import load_agent, set_command
from .course import CourseConfig
from .course_progress import CourseProgress
from .environment import make_env
from .experiments import create_experiment, write_manifest, resolve_experiment_path


@dataclasses.dataclass
class Score:
    elapsed: float = 0.0
    falls: int = 0
    finished: bool = False
    supported_finish_time: float = 0.0

    def advance(self, dt, fell, at_finish):
        if self.finished:
            return None
        self.elapsed += dt
        if fell:
            self.falls += 1
            self.supported_finish_time = 0
            return "fall"
        self.supported_finish_time = self.supported_finish_time + dt if at_finish else 0
        if self.supported_finish_time >= 0.4:
            self.finished = True
            return "finish"
        return None


@dataclasses.dataclass
class PostureMonitor:
    collapsed_seconds: float = 0.0

    def update(self, dt, upright, base_above_feet):
        collapsed = upright < .65 or (upright < .85 and base_above_feet < .17)
        self.collapsed_seconds = self.collapsed_seconds + dt if collapsed else 0.0
        return self.collapsed_seconds >= .5


class GameSession:
    """One simulation owner; call methods on the thread that creates the session."""

    def __init__(self, walking, hopping=None, course=None, render=True, seed=263,
                 run_root="experiments", record=True, flat=False, operator="human",
                 name="desert_droid_game"):
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        walking = resolve_experiment_path(walking)
        hopping = resolve_experiment_path(hopping) if hopping else None
        self.walking_path = str(walking.resolve())
        self.hopping_path = str(hopping.resolve()) if hopping else None
        self.walk, self.walk_cfg = load_agent(walking, 1)
        self.hop, self.hop_cfg = load_agent(hopping, 1) if hopping else (None, None)
        if self.walk.dimensions[0] != 47 or self.walk.dimensions[2] != 10:
            raise ValueError("Walking policy must have the 47-input proprioceptive actor and 10 leg actions")
        if self.hop and (self.hop.dimensions[0] != 49 or self.hop.dimensions[2] != 10):
            raise ValueError("Hopping policy must have the 49-input proprioception/phase actor and 10 leg actions")
        for cfg in [self.walk_cfg] + ([self.hop_cfg] if self.hop else []):
            if cfg.get("task_version") not in ["v2", "hop"]:
                raise ValueError("Policy uses incompatible legacy action scales")
        self.course = CourseConfig(**(course or {}))
        self.flat = flat
        self.seed, self.run_root = seed, run_root
        self.operator = operator
        self.name = name
        self.env = make_env(1, render=render, task_version="hop",
                            course=None if flat else dataclasses.asdict(self.course), seed=seed)
        self.env.cfg.episode_length_s = 1e9
        self.foot_ids, _ = self.env.scene["robot"].find_sites(["left_foot", "right_foot"])
        self.render_enabled, self.record = render, record
        self.video_part = 0
        self.writer, self.log = None, None
        self.last_frame = None
        self.paused = True
        self.new_game()

    def new_game(self):
        if self.log:
            self.save_summary()
            self.log.close()
        if self.writer:
            self.writer.close()
            self.writer = None
        self.output = create_experiment(self.name, self.run_root)
        config = {"name": self.name, "seed": self.seed,
                  "hardware": torch.cuda.get_device_name(0), "walking": self.walking_path,
                  "hopping": self.hopping_path, "course": dataclasses.asdict(self.course),
                  "flat_transition_test": self.flat,
                  "operator": self.operator,
                  "control": "velocity/mode commands; no navigation or heading correction"}
        (self.output / "config/game.json").write_text(json.dumps(config, indent=2))
        write_manifest(self.output, config, self.env.cfg)
        self.log = (self.output / "metrics/gameplay.jsonl").open("w", buffering=1)
        self.video_part = 0
        self.score = Score()
        self.steps = 0
        self.message = "Ready. Load your policies, then drive to the finish beacon."
        self.reset_robot()
        self.paused = True

    def pause(self):
        self.paused = not self.paused
        if self.paused:
            self.save_summary()

    def reset_robot(self):
        self.env.reset(seed=self.seed + self.score.falls)
        self.posture = PostureMonitor()
        self.progress = CourseProgress()
        self.stage = "walk"
        self.requested_mode = "walk"
        self.requested_speed = self.requested_yaw = 0.0
        self.speed = self.yaw = 0.0
        self.stage_time = self.hop_time = 0.0
        self.last_action = torch.zeros(1, 10, device=self.env.device)

    def command(self, speed=0, yaw=0, mode="walk"):
        if mode not in ["walk", "hop"]:
            raise ValueError("Mode must be walk or hop")
        if mode == "hop" and self.hop is None:
            raise ValueError("Load a hopping checkpoint before requesting hop mode")
        self.requested_speed = float(np.clip(speed, -0.25, 0.5))
        self.requested_yaw = float(np.clip(yaw, -0.6, 0.6))
        self.requested_mode = mode
        self.paused = False

    def _policy_action(self):
        dt = self.env.step_dt
        self.stage_time += dt
        # Only actor-available IMU quantities are used by this mode handover.
        proprio = self.env.observation_manager.compute_group("actor")
        calm = bool(proprio[0, :3].norm() < 1.0 and -proprio[0, 5] > .9)
        if self.stage == "walk" and self.requested_mode == "hop":
            self.stage, self.stage_time = "braking", 0
        if self.stage == "braking" and self.requested_mode == "walk":
            self.stage, self.stage_time = "walk", 0
        if self.stage == "braking" and self.stage_time >= .65 and calm:
            self.stage, self.stage_time, self.hop_time = "blend_to_hop", 0, 0
        if self.stage == "hop" and self.requested_mode == "walk" and self.hop_time % 1.0 < .2 and calm:
            self.stage, self.stage_time = "blend_to_walk", 0
        if self.stage == "blend_to_hop" and self.stage_time >= .3:
            self.stage, self.stage_time = "hop", 0
        if self.stage == "blend_to_walk" and self.stage_time >= .3:
            self.stage, self.stage_time = "walk", 0
        transitioning = self.stage in ["braking", "blend_to_hop", "blend_to_walk"]
        target_speed = 0.0 if transitioning else self.requested_speed
        target_yaw = 0.0 if transitioning else self.requested_yaw
        active_cfg = self.hop_cfg if self.stage == "hop" else self.walk_cfg
        if active_cfg and self.stage == "hop":
            if active_cfg.get("command_mode", "fixed") == "fixed":
                target_speed, target_yaw = active_cfg["speed"], 0
            else:
                target_speed = float(np.clip(target_speed, active_cfg["speed_min"], active_cfg["speed_max"]))
                target_yaw = float(np.clip(target_yaw, -active_cfg["yaw_max"], active_cfg["yaw_max"]))
        self.speed += float(np.clip(target_speed - self.speed, -.6 * dt, .6 * dt))
        self.yaw += float(np.clip(target_yaw - self.yaw, -1.2 * dt, 1.2 * dt))
        obs = set_command(self.env, self.speed, self.yaw)["actor"]
        with torch.no_grad():
            walk_action = self.walk.act(obs[:, :47], deterministic=True)
            if self.stage in ["walk", "braking"]:
                return walk_action
            self.hop_time += dt
            hop_obs = obs.clone()
            angle = self.hop_time * 2 * math.pi
            hop_obs[:, -2:] = torch.tensor([math.sin(angle), math.cos(angle)], device=obs.device)
            hop_action = self.hop.act(hop_obs, deterministic=True)
            if self.stage == "hop":
                return hop_action
            fraction = min(1.0, self.stage_time / .3)
            if self.stage == "blend_to_walk":
                fraction = 1 - fraction
            return (1 - fraction) * walk_action + fraction * hop_action

    def step(self):
        if self.paused or self.score.finished:
            return self.status()
        if self.render_enabled and self.record and self.writer is None:
            self.video_part += 1
            self.writer = imageio.get_writer(
                self.output / f"videos/gameplay_{self.video_part:03d}.mp4",
                fps=25, codec="libx264", quality=7, macro_block_size=16)
        action = self._policy_action()
        _, _, terminated, _, _ = self.env.step(action)
        self.steps += 1
        robot = self.env.scene["robot"].data
        position = robot.root_link_pos_w[0].cpu().tolist()
        feet = robot.site_pos_w[0, self.foot_ids].cpu().tolist()
        self.progress.update(feet)
        upright = float(-robot.projected_gravity_b[0, 2])
        contact = bool((self.env.scene["feet_ground_contact"].data.found > 0).any())
        # Scoring-only measurements: a collapsed robot should reset even if the
        # training termination allows a larger recovery window. A short grace
        # period and relative foot height avoid classifying normal hops as falls.
        base_above_feet = position[2] - float(robot.site_pos_w[0, self.foot_ids, 2].min())
        collapsed = self.posture.update(self.env.step_dt, upright, base_above_feet)
        outside = abs(position[1]) > 3.5 or position[0] < -1.5
        fall_reason = ("training_termination" if bool(terminated[0]) else
                       "course_bounds" if outside else "collapsed_posture" if collapsed else None)
        fell = fall_reason is not None
        at_finish = (not self.flat and self.progress.complete and
                     position[0] >= self.course.length - .35 and abs(position[1]) < .65 and
                     position[2] > self.course.finish_height + .16 and upright > .9 and contact)
        event = self.score.advance(self.env.step_dt, fell, at_finish)
        row = {"time": self.score.elapsed, "position": position, "upright": upright,
               "speed_command": self.speed, "yaw_command": self.yaw, "mode": self.stage,
               "action_change_rms": float((action - self.last_action).square().mean().sqrt()),
               "event": event, "falls": self.score.falls,
               "base_above_feet": base_above_feet, "fall_reason": fall_reason,
               "foot_positions": feet, "course_checkpoints": self.progress.completed,
               "foot_contacts": (self.env.scene["feet_ground_contact"].data.found[0] > 0).flatten().cpu().tolist(),
               "body_angular_velocity": robot.root_link_ang_vel_b[0].cpu().tolist(),
               "hop_phase": self.hop_time % 1.0}
        self.last_action = action.clone()
        self.log.write(json.dumps(row) + "\n")
        if self.render_enabled and self.steps % 2 == 0:
            frame = self.render()
            if self.writer:
                self.writer.append_data(frame)
        if event == "fall":
            self.message = "Reset after a fall or leaving the course. Time and falls are retained."
            self.reset_robot()
        elif event == "finish":
            self.message = "Finish reached! Your droid earned the beacon."
            self.save_summary()
        return self.status()

    def render(self):
        frame = Image.fromarray(self.env.render())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, frame.width, 35), fill="#173f32")
        draw.text((12, 11), f"DESERT DROID RUN  |  {self.score.elapsed:.1f}s  |  falls {self.score.falls}  |  {self.stage}  |  checkpoints {self.progress.completed}/4", fill="white")
        self.last_frame = np.asarray(frame)
        return self.last_frame

    def set_camera(self, view):
        """Change the player's view without changing policy observations."""
        import mujoco
        if view not in ("follow", "top", "overview"):
            raise ValueError("Unknown camera view")
        if self.env._offline_renderer is None:
            self.render()
        camera = self.env._offline_renderer._cam
        if view == "overview":
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [6, 0, 0]
            camera.distance, camera.azimuth, camera.elevation = 14, 90, -65
        else:
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = self.env.scene["robot"].indexing.root_body_id
            camera.distance, camera.azimuth, camera.elevation = (
                (3.2, 90, -70) if view == "top" else (2, 140, -18))

    def jpeg(self):
        if self.last_frame is None:
            self.render()
        output = io.BytesIO()
        Image.fromarray(self.last_frame).save(output, format="JPEG", quality=80)
        return output.getvalue()

    def status(self):
        return {"time": round(self.score.elapsed, 2), "falls": self.score.falls,
                "finished": self.score.finished, "mode": self.stage,
                "course_checkpoints": self.progress.completed,
                "paused": self.paused, "message": self.message,
                "experiment": str(self.output), "walking": self.walking_path,
                "hopping": self.hopping_path, "speed": self.speed, "yaw": self.yaw}

    def save_summary(self):
        if self.writer:
            self.writer.close()
            self.writer = None
        (self.output / "evaluation/score.json").write_text(json.dumps(self.status(), indent=2))

    def close(self):
        self.save_summary()
        if self.writer:
            self.writer.close()
        if self.log:
            self.log.close()
        self.env.close()
