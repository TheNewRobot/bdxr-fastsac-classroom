# Third-party projects

This lesson builds on:

- **BDX-R-MjLab**, https://github.com/BDX-R/BDX-R-MjLab — Apache License 2.0.
  Robot MJCF, meshes, environment configuration, and reward functions.
- **Holosoma**, https://github.com/amazon-far/holosoma — Apache License 2.0.
  FastSAC networks, distributional critic projection, normalization, replay, and
  update equations.
- **mjlab**, https://github.com/mujocolab/mjlab — Apache License 2.0.
  GPU environment framework.
- **MuJoCo** and **MuJoCo Warp**, Google DeepMind — Apache License 2.0.
  Physics simulation.
- **NVIDIA Warp** — Apache License 2.0. GPU computation.

The setup script fetches the first three projects at recorded revisions, keeping
their upstream license files. The upstream BDX-R demonstration is linked and
credited separately; it is not represented as training output from this lesson.

Our environment adapter changes commands, episode resets, randomization, action
selection, and camera settings. Our FastSAC loop adapts logging and checkpointing
to mjlab while importing the credited upstream network and replay source.
