# %% [markdown]
# # Teach a little droid to walk
# **163A / 263A · Robot kinematics and reinforcement learning**
#
# Build a simulation, inspect the robot's joints, train a walking controller,
# then change one reward and see how its gait changes.
#
# **Today's questions:** What does the robot observe? What does its action mean?
# How does a reward influence movement? How do we check whether it learned?
#
# **Route through the lesson:** robot → environment → kinematics → control problem
# → SAC → your walking policy → one controlled experiment → optional hopping → play.
#
# Examples are part of this text: images and short, silent looping clips. Run the exercises
# to create your own results. Timings are from a local L40S; Colab remains untested.

# %% [markdown]
# ## 1 · Meet the droid
#
# The real-world demonstration is Disney's BDX droid. Our classroom uses the
# separately developed **BDX-R open-source model**, simulated with **MuJoCo Warp**
# through mjlab. We train with **FastSAC**; this is not Disney's controller.
#

# %% [markdown]
# | Real-world inspiration | Our simulated walker |
# |:---:|:---:|
# | ![Disney BDX droid — D23 reference photograph](lesson-asset:reference/disney_bdx.jpg) | ![Recorded instructor walking example](lesson-asset:inline/walking.gif) |
#
# Head gaze is a cosmetic rendering adjustment; leg motion comes from simulation.
#
# Photo: [D23 / Disney](https://d23.com/bdx-droids-disneys-perfect-marriage-of-creativity-and-technological-achievement/), not relicensed. Simulation: the open-source BDX-R model with our recorded walking policy.

# %% [markdown]
# **Before running anything:** which joints do you expect to be most important
# for moving a foot forward? Which might help keep the body upright?
#
# Credits: [BDX-R model](https://github.com/BDX-R/BDX-R-MjLab),
# [FastSAC project](https://younggyo.me/fastsac-humanoid/),
# [Holosoma](https://github.com/amazon-far/holosoma), and
# [mjlab](https://github.com/mujocolab/mjlab). Pinned revisions and license notices
# are included with the lesson files.
