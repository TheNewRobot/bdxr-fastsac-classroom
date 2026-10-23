"""Physical head expression with command-relative head proprioception.

This optional controller moves the actual articulation, so rendering and mesh
contacts use the same pose. It leaves the learned leg actions and all non-head
observation channels unchanged. Its CPU scratch state is used only for model
kinematics and gravity compensation; it never writes simulator joint positions.
"""

import math

import mujoco
import numpy as np
import torch

from bdxr_classroom.simulation.head_motion import HeadMotion, JOINT_NAMES


class PhysicalHeadMotion(HeadMotion):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.model = env.sim.mj_model
        self.scratch = mujoco.MjData(self.model)
        self.hinges = [self.model.joint("robot/" + name).id for name in JOINT_NAMES]
        self.addresses = self.model.jnt_qposadr[self.hinges]
        self.dofs = self.model.jnt_dofadr[self.hinges]
        self.head_body = self.model.body("robot/Head_Base_Plate").id
        actuators = []
        for hinge in self.hinges:
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == hinge)
            if len(matches) != 1:
                raise ValueError(
                    "Physical expression requires one position servo per head joint."
                )
            actuators.append(matches[0])
        self.gains = self.model.actuator_gainprm[actuators, 0].copy()
        if np.any(self.gains <= 0):
            raise ValueError("Physical head position gains must be positive.")
        # Verified actor layout: gyro3, gravity3, position14, velocity14,
        # leg actions10, commands3; hopping adds two phase channels.
        if env.scene["robot"].data.joint_pos.shape[-1] != 14:
            raise ValueError("Unexpected robot joint layout for head proprioception.")
        ids = torch.as_tensor(self.joint_ids, device=env.device)
        self.position_columns = 6 + ids
        self.velocity_columns = 20 + ids
        self.expression_mode = "walk"

    def reset(self, seed=None):
        super().reset(seed)
        self.position = np.zeros((self.num_envs, 4))
        self.velocity = np.zeros_like(self.position)
        self.last_acceleration = np.zeros_like(self.position)

    def desired(self, speed, yaw, mode):
        expression = (
            self.display_offsets()
            if self.expression_mode == "walk"
            else dict(neck=0.0, yaw=0.0, roll=0.0, nod=0.0)
        )
        states = self.env.sim.data.qpos.cpu().numpy()
        targets = []
        for state in states:
            data = self.scratch
            data.qpos[:] = state
            data.qvel[:] = 0
            pose = np.array(
                [expression["neck"], 0, expression["yaw"], expression["roll"]]
            )
            data.qpos[self.addresses] = pose
            mujoco.mj_kinematics(self.model, data)
            forward = data.xmat[self.head_body].reshape(3, 3)[:, 0]
            axis = data.xaxis[self.hinges[1]]
            constant = axis[2] * float(axis @ forward)
            cosine = forward[2] - constant
            sine = np.cross(axis, forward)[2]
            correction = 0.0
            for _ in range(5):
                value = (
                    cosine * math.cos(correction)
                    + sine * math.sin(correction)
                    + constant
                )
                derivative = (
                    -cosine * math.sin(correction) + sine * math.cos(correction)
                )
                if abs(derivative) < 1e-6:
                    break
                correction = float(
                    np.clip(
                        correction
                        - (value - math.sin(expression["nod"])) / derivative,
                        -1.2,
                        1.2,
                    )
                )
            pose[1] = correction
            pose = np.clip(pose, self.lower, self.upper)
            data.qpos[self.addresses] = pose
            # These stages compute gravity without running CPU collision or
            # constraint solves. Actual dynamics remain in the GPU simulator.
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            mujoco.mj_comVel(self.model, data)
            mujoco.mj_rne(self.model, data, 0, data.qfrc_bias)
            targets.append(pose + data.qfrc_bias[self.dofs] / self.gains)
        return np.asarray(targets)

    def step(self, dt, speed, yaw, mode="walk"):
        self.expression_mode = mode
        # Explicit hopping suppresses gestures but keeps the physical head
        # supported. The base filter still limits servo-reference changes.
        idle_before = self.idle_fraction
        walking = mode == "walk"
        result = super().step(
            dt, speed if walking else 0, yaw if walking else 0, mode="walk"
        )
        if not walking:
            self.idle_fraction = idle_before * math.exp(-dt / 0.35)
        return result

    def policy_observations(self, observations):
        """Express head servo error relative to the previously applied command.

        Call before step(): the observed state was produced by that previous
        target. Leg channels, inertial sensors, and commands are copied intact.
        """
        if observations.shape not in ((self.num_envs, 47), (self.num_envs, 49)):
            raise ValueError(
                "Unexpected actor observation shape for physical head expression."
            )
        result = observations.clone()
        result[:, self.position_columns] -= torch.as_tensor(
            self.position, dtype=result.dtype, device=result.device
        )
        result[:, self.velocity_columns] -= torch.as_tensor(
            self.velocity, dtype=result.dtype, device=result.device
        )
        return result

    def render(self):
        return self.env.render()

    def metadata(self):
        return {
            **super().metadata(),
            "gaze_correction": "physical position servos with model gravity compensation",
            "head_observation_reference": "procedural servo position and velocity commands",
            "render_pose": "unmodified physical articulation",
        }
