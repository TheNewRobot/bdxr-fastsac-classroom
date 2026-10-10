# %% [markdown]
# ## 5 · Configure Soft Actor-Critic
#
# **SAC** learns from experience saved in a replay buffer. The actor proposes
# actions; two critics estimate future reward. Training uses the smaller critic
# estimate to reduce overly optimistic targets. An entropy term encourages
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
# `num_envs` counts independent robots collecting experience. `batch_size` counts
# replay samples in a neural-network update. These are different quantities.
# `steps` counts parallel simulation steps: with 512 robots, one step collects
# 512 transitions. The training curve plots collected transitions.

# %% [markdown]
# | Learning setting | Recorded baseline | What it changes |
# |---|---|---|
# | Parallel robots | 512 | Experience collected each simulation step |
# | Batch size | 2,048 | Replay samples used per update |
# | Simulation steps | 10,000 | Up to 5.12 million collected transitions |
# | Discount | 0.97 | Emphasis on future reward |
# | Updates per step | 4 | Learning work per collection step |
#
# ### Your activity · configure and inspect SAC
#
# Keep these settings for the first run. Inspect the network and store one
# transition before launching training. A single untrained action is not walking.

# %%
LEARNING_CONFIG = {
    "seed": 7,  # Reproduce the recorded baseline's training seed.
    "num_envs": 512,  # Independent simulations collecting experience.
    "batch_size": 2048,  # Replay samples per learning update.
    "buffer_size": 512,  # Replay slots per parallel robot.
    "steps": 10000,  # Parallel simulation steps.
    "minutes": 15,  # Stop earlier if the training-time limit is reached.
    "gamma": 0.97,  # Discount applied to future reward.
    "updates": 4,  # Learning updates per simulation step.
    "actor_hidden": 256,  # Actor hidden-layer width.
    "critic_hidden": 384,  # Critic hidden-layer width.
    "mirror_replay": True,  # Reuse left/right mirrored experience.
    "compile": True,  # Compile the repeated GPU learning operations.
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
# The recorded baseline took about **9 minutes** of training on an L40S GPU.
# Setup, compilation, evaluation, and video rendering take additional time.
# A Colab GPU or a different seed may take longer or fail the quality checks.
#
# **Predict:** what changes if we double the number of parallel environments but
# keep the number of simulation steps unchanged?

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
print(agent.actor)

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
