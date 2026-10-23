"""A flat, collidable practice pad with simple original desert scenery."""

import mujoco


def build_arena(spec):
    terrain = spec.body("terrain")
    if terrain is None:
        raise ValueError("The hopping pad requires the terrain contact body.")
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.material = ""
            geom.rgba = (0.72, 0.54, 0.32, 1)
            # Full-link meshes use bit 2; the floor must accept those contacts.
            geom.conaffinity |= 2
    terrain.add_geom(
        name="hop_pad", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(0, 0, -0.009), size=(0.7, 0.01, 0),
        rgba=(0.84, 0.74, 0.55, 1), contype=1, conaffinity=3,
        friction=(1.0, 0.005, 0.0001),
    )
    for index, (x, y) in enumerate([(-0.8, 0.8), (0.8, 0.8), (-0.8, -0.8), (0.8, -0.8)]):
        terrain.add_geom(
            name=f"pad_marker_{index}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=(x, y, 0.09), size=(0.025, 0.09, 0),
            rgba=(0.22, 0.38, 0.38, 1), contype=1, conaffinity=3,
        )
        terrain.add_geom(
            name=f"pad_light_{index}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=(x, y, 0.19), size=(0.035, 0, 0),
            rgba=(0.25, 0.85, 0.9, 1), contype=1, conaffinity=3,
        )
