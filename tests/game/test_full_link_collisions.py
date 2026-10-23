"""Exercise contact for every demo mesh, not merely collision-mask settings."""

import unittest

import mujoco
import numpy as np

from bdx_r_mjlab.robots.bdxr.bdxr_constants import get_spec
from bdxr_classroom.game.collisions import visible_link_spec


class FullLinkCollisions(unittest.TestCase):
    def test_all_visible_links_contact_a_probe_without_changing_inertia(self):
        original = get_spec().compile()
        spec = visible_link_spec(get_spec)
        for geom in spec.geoms:
            if geom.name.startswith("demo_link_"):
                geom.contype, geom.conaffinity = 2, 0
        probe_body = spec.worldbody.add_body(name="probe_body", pos=[10, 10, 10])
        probe_body.add_freejoint(name="probe_free")
        probe_body.add_geom(
            name="contact_probe", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[.01, 0, 0], contype=0, conaffinity=2,
        )
        model = spec.compile()
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        np.testing.assert_array_equal(original.body_mass, model.body_mass[:original.nbody])
        np.testing.assert_array_equal(original.body_inertia, model.body_inertia[:original.nbody])
        data = mujoco.MjData(model)
        probe = model.geom("contact_probe").id
        probe_qpos = int(model.joint("probe_free").qposadr[0])
        mujoco.mj_forward(model, data)
        targets = [i for i in range(model.ngeom) if model.geom(i).name.startswith("demo_link_")]
        self.assertGreater(len(targets), 20)
        for target in targets:
            with self.subTest(link=model.body(model.geom_bodyid[target]).name):
                mesh = model.geom_dataid[target]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = model.mesh_vert[start:start + count]
                # Probe the outer surface, not the hollow interior of a cover.
                rotation = data.geom_xmat[target].reshape(3, 3)
                point = vertices[vertices[:, 0].argmax()]
                data.qpos[probe_qpos:probe_qpos + 3] = (
                    data.geom_xpos[target] + rotation @ point
                )
                mujoco.mj_forward(model, data)
                self.assertTrue(any(set(c.geom) == {target, probe} for c in data.contact))
                self.assertTrue(np.isfinite(data.qacc).all())
                self.assertTrue(all(np.isfinite(c.pos).all() for c in data.contact))
