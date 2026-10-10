"""Blind locomotion practice on small steps; terrain heights never enter the actor."""

import math
import mujoco
import torch
from mjlab.utils.lab_api.math import quat_apply


def practice_steps(height):
    """A shared tile for independent simulation worlds, with a clear spawn area."""
    return [
        (sign * x, width, height * factor)
        for sign in (-1, 1)
        for x, width, factor in [(0.65, .08, 1), (1.25, .22, .8),
                                 (1.95, .07, 1), (2.65, .3, .6),
                                 (3.45, .1, 1)]
    ]


def build_training_terrain(spec, height, style="steps"):
    terrain = spec.body("terrain")
    for index, (x, half_width, top) in enumerate(practice_steps(height)):
        pipe = style == "mixed" and index % 5 in (0, 2)
        radius = top * (2 / 3.5)
        terrain.add_geom(
            name=f"practice_step_{index}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER if pipe else mujoco.mjtGeom.mjGEOM_BOX,
            pos=(x, 0, top - radius if pipe else top / 2),
            size=(radius, 4, 0) if pipe else (half_width, 4, top / 2),
            quat=(math.sqrt(.5), math.sqrt(.5), 0, 0) if pipe else (1, 0, 0, 0),
            rgba=(.3, .4, .4, 1) if pipe else (.65, .48, .29, 1),
            contype=1, conaffinity=1, friction=(1, .005, .0001))


def ground_height(env, positions):
    """Privileged reward measurement for this analytic training tile only."""
    result = torch.zeros_like(positions[..., 2])
    height = getattr(env, "training_terrain_height", 0.0)
    if height:
        for index, (x, half_width, top) in enumerate(practice_steps(height)):
            pipe = getattr(env, "training_terrain_style", "steps") == "mixed" and index % 5 in (0, 2)
            dx = positions[..., 0] - x
            if pipe:
                half_width = top * (2 / 3.5)
                surface = top - half_width + (half_width**2 - dx.square()).clamp_min(0).sqrt()
            else:
                surface = result.new_tensor(top)
            inside = ((positions[..., 0] - x).abs() <= half_width) & (
                positions[..., 1].abs() <= 4)
            result = torch.where(inside, torch.maximum(result, surface), result)
    return result


def sampled_sole_clearance(env):
    """Estimate clearance along the actual foot capsules, for rewards only.

    A reference point behind a pipe can report flat ground while the toe is
    already over it. Sample both outside capsule centerlines on each foot and
    subtract their radius. This is a sampled clearance, not an exact collision
    distance, and it introduces no terrain observation into the actor.
    """
    robot = env.scene["robot"]
    if not hasattr(env, "_sole_clearance_geometry"):
        names = ["left_foot_collision1", "left_foot_collision7",
                 "right_foot_collision1", "right_foot_collision7"]
        ids, found = robot.find_geoms(names)
        if list(found) != names:
            raise ValueError("Unexpected BDX-R foot collision geometry order")
        global_ids = robot.indexing.geom_ids[ids].cpu().numpy()
        sizes = torch.tensor(env.sim.mj_model.geom_size[global_ids],
                             device=env.device, dtype=torch.float32)
        distances = torch.linspace(-1, 1, 21, device=env.device)[None, None, :, None]
        env._sole_clearance_geometry = (ids, sizes, distances * sizes[None, :, None, 1:2])
    ids, sizes, distances = env._sole_clearance_geometry
    positions = robot.data.geom_pos_w[:, ids]
    axes = quat_apply(robot.data.geom_quat_w[:, ids],
                      positions.new_tensor([0., 0., 1.]).expand_as(positions))
    points = positions[:, :, None, :] + axes[:, :, None, :] * distances
    clearance = points[..., 2] - ground_height(env, points) - sizes[None, :, None, 0]
    return clearance.reshape(env.num_envs, 2, 2, 21).amin(dim=(-1, -2))
