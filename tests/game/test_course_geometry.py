import unittest

import mujoco
import numpy as np

from bdxr_classroom.game.course import CourseConfig, build_course


def model(config):
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="terrain">'
        '<geom type="plane" size="20 20 .1"/>'
        "</body></worldbody></mujoco>"
    )
    build_course(spec, config)
    return spec.compile()


class CourseGeometryTests(unittest.TestCase):
    def test_step_override_changes_only_middle_obstacle(self):
        original = model(CourseConfig())
        taller = model(CourseConfig(step_height=0.035))
        step = original.geom("obstacle_2_step").id
        self.assertAlmostEqual(
            original.geom_pos[step, 2] + original.geom_size[step, 2], 0.020
        )
        self.assertAlmostEqual(
            taller.geom_pos[step, 2] + taller.geom_size[step, 2], 0.035
        )
        self.assertEqual(taller.geom_contype[step], 1)
        other = np.arange(original.ngeom) != step
        np.testing.assert_array_equal(original.geom_pos[other], taller.geom_pos[other])
        np.testing.assert_array_equal(
            original.geom_size[other], taller.geom_size[other]
        )
        np.testing.assert_array_equal(original.geom_quat, taller.geom_quat)

    def test_invalid_step_height_is_rejected(self):
        for value in [0, -0.01, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                model(CourseConfig(step_height=value))
