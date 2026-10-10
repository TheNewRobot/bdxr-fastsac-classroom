# %% [markdown]
# ## 2 · Create the environment
#
# Select **Runtime → Change runtime type → GPU**. Installation may take several
# minutes; it is separate from training time.
#
# An environment combines the robot, ground, actuators, observations, reward,
# and rules for ending an episode. We first inspect **one** robot. Training will
# use many independent copies of this same task.

# %% [hidden]
# @title Install pinned dependencies
setup_start = time.monotonic()
setup_log = WORK / "setup.log"
with setup_log.open("w") as log:
    result = subprocess.run(
        [sys.executable, "scripts/setup/setup.py"],
        cwd=WORK,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
if result.returncode:
    raise RuntimeError(setup_log.read_text()[-6000:])
PYTHON = str(WORK / ".venv/bin/python")
runtime_env = dict(
    os.environ,
    PYTHONPATH=str(WORK / "src"),
    MUJOCO_GL="egl",
    BDXR_RUNTIME="colab" if Path("/content").exists() else "local_notebook",
)
gpu = json.loads(
    subprocess.check_output(
        [
            PYTHON,
            "-c",
            "import torch,json; print(json.dumps(dict(name=torch.cuda.get_device_name(0),gib=torch.cuda.get_device_properties(0).total_memory/2**30)))",
        ],
        cwd=WORK,
        env=runtime_env,
        text=True,
    )
)
print("GPU:", gpu)
print("Setup time:", round(time.monotonic() - setup_start), "seconds")

# %% [hidden]
# @title Initialize the notebook's simulator connection
from bdxr_classroom.lesson.runtime import Lesson
from bdxr_classroom.lesson.simulator import SimulatorCells

if "simulator" in globals():
    simulator.close()
simulator = SimulatorCells(PYTHON, WORK, runtime_env)
simulator.register(get_ipython())
lab = Lesson(WORK, PYTHON, runtime_env)
(WORK / "figures").mkdir(exist_ok=True)

# %% [markdown]
# ### Define the walking task
#
# These are the validated baseline settings. Positive forward speed asks for
# forward motion; negative speed asks for backward motion. Positive yaw rate
# requests a left turn. Zero yaw requests no rotation, without a compass or
# an external heading-correction controller.
#
# `balanced` samples several command categories during training. The last two
# settings reward small **sustained** turning error and sideways drift.

# %%
ENV_CONFIG = {
    "task_version": "v2",  # Ten leg-joint targets; the complete robot remains physical.
    "command_mode": "balanced",  # Practice forward/backward movement, stopping, and turns.
    "speed_min": -0.25,  # m/s, backward command limit.
    "speed_max": 0.50,  # m/s, forward command limit.
    "yaw_max": 0.60,  # rad/s, magnitude of the turning command.
    "yaw_tracking": "imu",  # Estimate turning rate from gyro and gravity direction.
    "yaw_std": 0.20,  # rad/s, width of the turning reward.
    "yaw_bias_weight": 6.0,  # Weight for reducing sustained turning error.
    "lateral_bias_weight": 2.0,  # Weight for reducing sustained sideways drift.
}
(WORK / "environment_settings.json").write_text(json.dumps(ENV_CONFIG, indent=2))

# %% [markdown]
# Cells marked `%%robot` run visible Python in the pinned simulator environment.
# Variables persist between these cells. Ordinary cells display figures and
# organize experiments in the notebook.

# %%
%%robot
import json
from pathlib import Path
from bdxr_classroom.simulation.environment import make_env

settings = json.loads(Path("environment_settings.json").read_text())
env = make_env(num_envs=1, seed=7, render=True, **settings)
observations, _ = env.reset(seed=7)
print("Physics timestep:", env.cfg.sim.mujoco.timestep, "seconds")
print("Policy/control timestep:", env.step_dt, "seconds")
print("Episode time limit:", env.cfg.episode_length_s, "seconds")

# %%
%%robot
from bdxr_classroom.lesson.kinematics import save_pose

save_pose(env, "figures/robot.png")
print("Robot joints:", env.scene["robot"].joint_names)

# %%
display(Image(filename=str(WORK / "figures/robot.png"), width=720))

# %% [markdown]
# The simulator advances physics every **5 ms**; the policy chooses new targets
# every **20 ms**. Training episodes last up to **10 s** and also end on the
# configured fall conditions. A reset starts another attempt.
#
# **Check your understanding:** why can physics need more updates than the policy?
