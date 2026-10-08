"""Blind locomotion practice on small steps; terrain heights never enter the actor."""

import mujoco
import torch


def practice_steps(height):
    """A shared tile for independent simulation worlds, with a clear spawn area."""
    return [
        (sign * x, width, height * factor)
        for sign in (-1, 1)
        for x, width, factor in [(0.65, .08, 1), (1.25, .22, .8),
                                 (1.95, .07, 1), (2.65, .3, .6),
                                 (3.45, .1, 1)]
    ]


def build_training_terrain(spec, height):
    terrain = spec.body("terrain")
    for index, (x, half_width, top) in enumerate(practice_steps(height)):
        terrain.add_geom(
            name=f"practice_step_{index}", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(x, 0, top / 2), size=(half_width, 4, top / 2),
            rgba=(.65, .48, .29, 1), contype=1, conaffinity=1,
            friction=(1, .005, .0001))


def ground_height(env, positions):
    """Privileged reward measurement for this analytic training tile only."""
    result = torch.zeros_like(positions[..., 2])
    height = getattr(env, "training_terrain_height", 0.0)
    if height:
        for x, half_width, top in practice_steps(height):
            inside = ((positions[..., 0] - x).abs() <= half_width) & (
                positions[..., 1].abs() <= 4)
            result = torch.where(inside, torch.maximum(result, result.new_tensor(top)), result)
    return result
