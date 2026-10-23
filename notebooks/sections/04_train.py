# %% [markdown]
# ## 4 · Configure SAC and train
#
# **SAC** learns from experience saved in a replay buffer. The actor proposes
# actions; two critics estimate future return. Replay samples update the critics
# and actor. An entropy term encourages
# exploration rather than choosing the same action every time.
#
# $$J(\pi)=\mathbb{E}\left[\sum_t\gamma^t
# \left(r_t+\alpha\,\mathcal H(\pi(\cdot\mid o_t))\right)\right]$$
#
# Read it as **future rewards + encouragement to explore**. $\gamma$ discounts
# distant rewards; $\alpha$ controls the entropy contribution. FastSAC makes this
# off-policy learning loop efficient on the GPU. This lesson uses SAC, not PPO.

# %% [markdown]
# | Learning from replay |
# |:---:|
# | ![SAC: collect experience, sample replay, update actor and critics](lesson-asset:reference/sac_loop.png) |

# %% [markdown]
# ### Separate learning settings from environment settings
#
# During training, the actor samples exploratory actions and stores transitions
# $(o_t,a_t,r_t,o_{t+1})$ in replay. Learning updates reuse these experiences;
# they do not require a new simulated step for every gradient update. During
# evaluation, the actor uses its deterministic action, without exploration noise.
#
# The actor learns which actions to choose. The critics learn to predict return,
# and their predictions supply a learning signal for the actor. A reward value
# describes the current transition; a critic estimate includes expected future
# rewards. Neither the reward calculation nor the critic is a deployed planner.
#
# `num_envs` counts independent robots collecting experience. `batch_size` counts
# sampled replay transitions; mirroring adds their reflected partners. These
# are different quantities.
# `steps` counts parallel simulation steps: with 512 robots, one step collects
# 512 transitions. The training curve plots collected transitions.

# %% [markdown]
# Keep the first-run settings below. We inspect one interaction, then train
# a fresh policy. The seed identifies an experiment; judge the result using its
# video and measurements rather than expecting identical GPU training.

# %%
LEARNING_CONFIG = {
    "seed": 7,  # Seed for your baseline and matched reward experiment.
    "num_envs": 512,  # Independent simulations collecting experience.
    "batch_size": 2048,  # Replay samples before left/right augmentation.
    "buffer_size": 512,  # Replay slots per parallel robot.
    "steps": 12000,  # Parallel simulation steps.
    "minutes": 20,  # Time cap for 12,000 steps; setup/display time is separate.
    "gamma": 0.97,  # Discount applied to future reward.
    "updates": 4,  # Learning updates per simulation step.
    "actor_hidden": 256,  # Actor hidden-layer width.
    "critic_hidden": 384,  # Critic hidden-layer width.
    "mirror_replay": True,  # Reuse left/right mirrored experience.
    "compile": True,  # Compile the repeated GPU learning operations.
    "save_every": 3000,  # Keep periodic checkpoints; assess the final weights.
}
USE_WANDB = False  # @param {type:"boolean"}
LEARNING_CONFIG["wandb_mode"] = "online" if USE_WANDB else "disabled"

# %% [hidden]
# @title Optional W&B connection
if USE_WANDB and not (
    os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()
):
    import getpass

    lab.environment["WANDB_API_KEY"] = getpass.getpass(
        "W&B API key (not stored in the lesson): "
    )
print("W&B:", LEARNING_CONFIG["wandb_mode"])
print("Local metrics and checkpoints are saved even when W&B is disabled.")

# %% [markdown]
# Use a **GPU runtime** and allow **up to 20 minutes per training run**,
# excluding setup and display. The target remains **12,000 steps**. The actual step count is
# printed and saved; unfinished budgets must not be treated as matching experiments.

# %% [markdown]
# ### Inspect the network and one interaction
#
# Before the long training call, create a small inspection instance with the same
# network widths. It uses one robot, eight replay slots, and no compilation or
# mirroring solely to make the data flow easy to inspect. The full training
# configuration above stays unchanged.

# %%
(WORK / "learning_settings.json").write_text(json.dumps(LEARNING_CONFIG, indent=2))

# %%
%%robot
from bdxr_classroom.learning.fastsac import FastSAC

inspection = json.loads(Path("learning_settings.json").read_text())
inspection.update(num_envs=1, buffer_size=8, compile=False, mirror_replay=False)
agent = FastSAC(
    observations["actor"].shape[-1],
    observations["critic"].shape[-1],
    env.action_manager.total_action_dim,
    inspection,
)
print("Actor inputs → hidden width → actions:",
      agent.dimensions[0], "→", inspection["actor_hidden"], "→", agent.dimensions[2])

# %%
%%robot
# An untrained actor chooses an action; physics determines the result.
actions = agent.act(observations["actor"])
next_observations, reward, terminated, truncated, info = env.step(actions)
agent.add(observations, actions, next_observations, reward, terminated, truncated)
print("Action shape:", actions.shape)
print("Reward:", reward)
print("Replay write index after storing this transition:", agent.buffer.ptr)

# %% [markdown]
# This is one **collection step**, not a trained policy. The full training program
# repeats collection, samples replay batches, and updates the critics and actor.
# `terminated` marks a task termination; `truncated` marks a time limit. Keeping
# the final observation lets the learner handle those endings correctly.

# %%
%%robot
# Release the inspection environment before the full training jobs.
env.close()

# %%
simulator.close()

# %% [markdown]
# ### Train, then watch
#
# Training starts from random networks. The helper saves the exact configuration,
# final weights, resumable state, and real learning metrics in a timestamped folder.

# %%
BASELINE = lab.train("command_walking_baseline", ENV_CONFIG, LEARNING_CONFIG)
print("Saved experiment:", BASELINE)

# %%
from IPython.display import display, Video

baseline_video = lab.record(BASELINE, speed=0.30)
display(Video(str(baseline_video), embed=True, width=640))

# %%
from IPython.display import display, Image

display(Image(filename=str(lab.learning_curve(BASELINE)), width=640))

# %% [markdown]
# Rising return suggests learning, but it does not establish good walking.
# **Observe:** does the droid stay upright, move forward, and drift sideways?
