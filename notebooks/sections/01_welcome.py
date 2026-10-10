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
# Read the explanations and embedded instructor examples first. Run activity cells
# to produce your own results below them; your results never replace the references.
# If a frontend asks, trust this notebook to enable saved video players. Images
# remain attached to the text even if outputs are cleared.
#
# Recorded examples below are instructor results. Your own videos are generated
# from the checkpoints you train later. Actual Google Colab execution is still
# untested; reported training times come from a local L40S GPU.

# %% [hidden]
# @title Unpack the lesson files
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from IPython.display import display, HTML, Image, Video

WORK = (
    Path("/content/bdxr_classroom")
    if Path("/content").exists()
    else Path.cwd() / "bdxr_classroom_lab"
)
WORK.mkdir(parents=True, exist_ok=True)
SOURCE_BUNDLE = "__SOURCE_BUNDLE__"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_BUNDLE))) as archive:
    archive.extractall(WORK)
sys.path.insert(0, str(WORK / "src"))
sys.path.insert(0, str(WORK))
from bdxr_classroom.lesson.presentation import (
    media_pair,
    table,
    read_json,
    recorded_details,
)

print("Lesson files:", WORK)

# %% [markdown]
# ## 1 · Meet the droid
#
# The real-world demonstration is Disney's BDX droid. Our classroom uses the
# separately developed **BDX-R open-source model**, simulated with **MuJoCo Warp**
# through mjlab. We train with **FastSAC**; this is not Disney's controller.
#
# **Instructor example — already displayed:** press Play to watch; no kernel is needed.
# The reference photograph is embedded with credit to D23 / Disney. It is
# reference imagery, not our hardware or training result; the source is linked below.

# %% [hidden] [instructor]
# @title Real-world inspiration beside our simulated walking policy
from bdxr_classroom.lesson.presentation import reference_comparison

reference_comparison(
    WORK / "assets/lesson/command_walking/rollout.mp4",
    reference_image="https://d23.com/app/uploads/2025/04/ADK_00797.jpg",
    reference_page="https://d23.com/bdx-droids-disneys-perfect-marriage-of-creativity-and-technological-achievement/",
    reference_photo=WORK / "assets/lesson/reference/disney_bdx.jpg",
)

recorded_details(
    WORK / "assets/lesson/command_walking",
    "Locked instructor reference. The curve covers its unchanged recovery stage "
    "only; the walking experiment below starts from scratch.",
)

# %% [markdown]
# **Before running anything:** which joints do you expect to be most important
# for moving a foot forward? Which might help keep the body upright?
#
# Credits: [BDX-R model](https://github.com/BDX-R/BDX-R-MjLab),
# [FastSAC project](https://younggyo.me/fastsac-humanoid/),
# [Holosoma](https://github.com/amazon-far/holosoma), and
# [mjlab](https://github.com/mujocolab/mjlab). Pinned revisions and license notices
# are included with the lesson files.
