"""Demo-only coverage of visible robot links against the course."""

from dataclasses import replace
from functools import partial

import mujoco
from mjlab.utils.spec_config import CollisionCfg


def visible_link_spec(original):
    """Reuse the model's mesh shapes, transforms and inertias unchanged.

    MuJoCo mesh contact uses convex hulls. These cover the visible links rather
    than the much smaller training collision primitives. A separate collision
    bit enables environment contact without creating new adjacent-link contacts.
    """
    spec = original()
    for index, geom in enumerate(spec.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            geom.name = f"demo_link_{index}"
    return spec


def full_link_collision_config(robot):
    return replace(
        robot,
        spec_fn=partial(visible_link_spec, robot.spec_fn),
        collisions=(
            *robot.collisions,
            CollisionCfg(
                geom_names_expr=("demo_link_.*",),
                contype=2,
                conaffinity=0,
                condim=3,
                friction=(0.6,),
            ),
        ),
    )
