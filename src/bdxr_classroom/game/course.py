"""Original, procedural desert-droid course; all traversable surfaces are physics."""

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CourseConfig:
    length: float = 12.0
    width: float = 1.6
    # Calibrated classroom heights: 17.5 mm pipe, 20 mm step, 18 mm threshold.
    obstacle_scale: float = 0.5
    # Optional height in metres for the middle step only.
    step_height: float | None = None
    roughness: float = 0.015
    finish_height: float = 0.18
    seed: int = 263
    walking_only: bool = False


def build_course(spec, cfg=None):
    """Add collision terrain and original scenery to the existing terrain body."""
    cfg = cfg or CourseConfig()
    terrain = spec.body("terrain")
    if terrain is None:
        raise ValueError(
            "Course requires the terrain body used by foot-contact sensors"
        )
    sand = (0.72, 0.54, 0.32, 1)
    cream = (0.82, 0.71, 0.51, 1)
    metal = (0.25, 0.35, 0.38, 1)
    blue = (0.25, 0.8, 0.95, 1)
    orange = (0.95, 0.48, 0.17, 1)

    def geom(name, kind, pos, size, rgba, collision=True, quat=(1, 0, 0, 0)):
        return terrain.add_geom(
            name=name,
            type=kind,
            pos=pos,
            size=size,
            rgba=rgba,
            quat=quat,
            contype=1 if collision else 0,
            conaffinity=1 if collision else 0,
            group=0 if collision else 2,
            friction=(1.0, 0.005, 0.0001),
        )

    for surface in spec.geoms:
        if surface.type == mujoco.mjtGeom.mjGEOM_PLANE:
            surface.material = ""
            surface.rgba = sand
    box, cylinder = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
    sphere = mujoco.mjtGeom.mjGEOM_SPHERE
    # The route's subtle color strip is visual only and has no raised edges.
    geom(
        "route",
        box,
        (6, 0, 0.0004),
        (6.8, cfg.width / 2, 0.0002),
        (0.63, 0.43, 0.25, 1),
        False,
    )
    scale = cfg.obstacle_scale
    # Three different physical challenges: pipe, low step, and narrow threshold.
    geom(
        "obstacle_1_pipe",
        cylinder,
        (2.0, 0, 0.015 * scale),
        (0.02 * scale, 0.65, 0),
        metal,
        quat=(math.sqrt(0.5), math.sqrt(0.5), 0, 0),
    )
    step_height = 0.04 * scale if cfg.step_height is None else cfg.step_height
    if not math.isfinite(step_height) or step_height <= 0:
        raise ValueError("Step height must be a positive finite number")
    geom(
        "obstacle_2_step",
        box,
        (4.8, 0, step_height / 2),
        (0.22, 0.65, step_height / 2),
        cream,
    )
    geom(
        "obstacle_3_threshold",
        box,
        (7.3, 0, 0.018 * scale),
        (0.07, 0.65, 0.018 * scale),
        orange,
    )
    if cfg.walking_only:
        # Offset passages reward steering. Low floor features require walking,
        # while the original hopping course keeps its previous geometry.
        for index, (x, center) in enumerate(((2.0, 0.32), (4.8, -0.32), (7.3, 0.18))):
            for side in (-1, 1):
                geom(
                    f"walking_gate_{index}_{side}",
                    box,
                    (x, center + side * 0.30, 0.13),
                    (0.28, 0.045, 0.13),
                    metal,
                )
                geom(
                    f"walking_gate_light_{index}_{side}",
                    sphere,
                    (x, center + side * 0.30, 0.28),
                    (0.045, 0, 0),
                    blue,
                    False,
                )
    # Deterministic rough patch, with modest relief for the small droid.
    rng = np.random.default_rng(cfg.seed)
    for ix in range(8):
        for iy in range(5):
            h = float(rng.uniform(0.002, cfg.roughness))
            geom(
                f"rough_{ix}_{iy}",
                box,
                (8.0 + ix * 0.17, (iy - 2) * 0.22, h / 2),
                (0.085, 0.11, h / 2),
                tuple(np.array(cream[:3]) * rng.uniform(0.83, 1)) + (1,),
            )
    # Gentle ascent, followed by a level raised finish platform.
    ramp_start, ramp_end = 9.7, 11.4
    angle = math.atan2(cfg.finish_height, ramp_end - ramp_start)
    geom(
        "finish_ramp",
        box,
        ((ramp_start + ramp_end) / 2, 0, cfg.finish_height / 2 - 0.025),
        ((ramp_end - ramp_start) / (2 * math.cos(angle)), 0.72, 0.025),
        cream,
        quat=(math.cos(angle / 2), 0, -math.sin(angle / 2), 0),
    )
    geom(
        "finish_platform",
        box,
        (cfg.length, 0, cfg.finish_height / 2),
        (0.6, 0.72, cfg.finish_height / 2),
        cream,
    )
    # Route lights and sci-fi gates are original simple geometry.
    for x in [0, 2, 4.8, 7.3, 9.7, 12]:
        for y in [-0.93, 0.93]:
            base = cfg.finish_height if x == 12 else 0
            geom(
                f"post_{x}_{y}",
                cylinder,
                (x, y, base + 0.13),
                (0.045, 0.13, 0),
                metal,
                False,
            )
            geom(
                f"light_{x}_{y}",
                sphere,
                (x, y, base + 0.28),
                (0.055, 0, 0),
                blue,
                False,
            )
    # Keep the decorative start beam behind the spawn so the top camera can
    # see the droid clearly before its first step.
    for x in [-0.5, 12]:
        base = cfg.finish_height if x == 12 else 0
        geom(f"gate_{x}", box, (x, 0, base + 0.85), (0.055, 1.0, 0.05), metal, False)
        geom(
            f"gate_glow_{x}",
            box,
            (x - 0.056, 0, base + 0.85),
            (0.004, 0.75, 0.017),
            blue,
            False,
        )
    # Decorative outpost buildings and rocks stay outside the playable corridor.
    for i, (x, y) in enumerate([(1, 2.0), (5.8, -2.1), (10.5, 2.2)]):
        geom(f"hut_{i}", cylinder, (x, y, 0.35), (0.55, 0.35, 0), cream, False)
        geom(f"dome_{i}", sphere, (x, y, 0.69), (0.55, 0, 0), cream, False)
        geom(f"door_{i}", box, (x - 0.55, y, 0.22), (0.01, 0.16, 0.22), metal, False)
        geom(f"antenna_{i}", cylinder, (x, y, 1.34), (0.018, 0.16, 0), metal, False)
        geom(f"antenna_tip_{i}", sphere, (x, y, 1.51), (0.035, 0, 0), orange, False)
    for i in range(18):
        x, y = (
            float(rng.uniform(-1.0, 13.0)),
            float(rng.choice([-1, 1]) * rng.uniform(1.3, 3.0)),
        )
        geom(
            f"rock_{i}",
            mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            (x, y, 0.12),
            (float(rng.uniform(0.12, 0.35)), 0.17, 0.18),
            (0.52, 0.32, 0.18, 1),
            False,
        )
    geom(
        "finish_prize",
        sphere,
        (cfg.length + 0.3, 0, cfg.finish_height + 0.7),
        (0.1, 0, 0),
        (1, 0.8, 0.2, 1),
        False,
    )
