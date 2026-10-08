"""Optional sagittal replay augmentation, following Holosoma's FastSAC approach.

BDX-R's left/right leg joint axes have opposite mirrored signs. The CAD model
is approximately symmetric, so this is an experimental training assumption,
not an exact symmetry claim about the physical robot.
"""

import torch


def joint_map(names):
    indices, signs = [], []
    for name in names:
        if name.startswith("Left_"):
            partner, sign = name.replace("Left_", "Right_", 1), -1
        elif name.startswith("Right_"):
            partner, sign = name.replace("Right_", "Left_", 1), -1
        elif name in ("Neck_Pitch", "Head_Pitch"):
            partner, sign = name, 1
        elif name in ("Head_Yaw", "Head_Roll"):
            partner, sign = name, -1
        else:
            raise ValueError(f"No BDX-R reflection rule for {name}")
        indices.append(names.index(partner))
        signs.append(sign)
    return {"indices": indices, "signs": signs}


def build_maps(env):
    robot = env.scene["robot"]
    joints = joint_map(list(robot.joint_names))
    action_term = env.action_manager.get_term("joint_pos")
    action_names = [robot.joint_names[i] for i in action_term._target_ids.tolist()]
    actions = joint_map(action_names)
    rules = {
        "base_lin_vel": ([0, 1, 2], [1, -1, 1]),
        "base_ang_vel": ([0, 1, 2], [-1, 1, -1]),
        "projected_gravity": ([0, 1, 2], [1, -1, 1]),
        "command": ([0, 1, 2], [1, -1, -1]),
        "joint_pos": (joints["indices"], joints["signs"]),
        "joint_vel": (joints["indices"], joints["signs"]),
        "actions": (actions["indices"], actions["signs"]),
        "foot_height": ([1, 0], [1, 1]),
        "foot_air_time": ([1, 0], [1, 1]),
        "foot_contact": ([1, 0], [1, 1]),
        "foot_contact_forces": ([3, 4, 5, 0, 1, 2], [1, -1, 1, 1, -1, 1]),
        "hop_phase": ([0, 1], [1, 1]),
    }
    maps = {"actions": actions}
    for group in ("actor", "critic"):
        indices, signs, offset = [], [], 0
        for name, shape in zip(env.observation_manager.active_terms[group],
                               env.observation_manager.group_obs_term_dim[group]):
            if name not in rules:
                raise ValueError(f"No reflection rule for {group}/{name}")
            permutation, factors = rules[name]
            if tuple(shape) != (len(permutation),):
                raise ValueError(f"Unexpected shape for {group}/{name}: {shape}")
            indices.extend(offset + i for i in permutation)
            signs.extend(factors)
            offset += len(permutation)
        maps[group] = {"indices": indices, "signs": signs}
    return maps


class ReplayMirror:
    def __init__(self, maps, device):
        self.maps = {
            key: (torch.tensor(value["indices"], device=device),
                  torch.tensor(value["signs"], device=device))
            for key, value in maps.items()
        }

    def reflect(self, value, group):
        indices, signs = self.maps[group]
        return value[..., indices] * signs

    def augment(self, batch):
        mirrored = batch.clone()
        for prefix in ((), ("next",)):
            for key, group in (("observations", "actor"), ("critic_observations", "critic")):
                location = prefix + (key,)
                mirrored[location] = self.reflect(batch[location], group)
        mirrored["actions"] = self.reflect(batch["actions"], "actions")
        # Scalar rewards, falls, timeouts and discounts are reflection invariant.
        # Concatenate raw samples before observation normalization.
        return torch.cat((batch, mirrored), dim=0)
