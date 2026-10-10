# %% [markdown]
# ## 4 · What does the policy observe and control?
#
# The policy receives numbers describing the robot and the requested movement.
# It outputs **ten leg-joint position targets**. Motor position controllers
# generate forces to approach those targets; the policy does not directly place
# the feet or teleport the robot.
#
# $$q_i^{\mathrm{target}} = q_i^{\mathrm{default}} + s_i a_i$$
#
# Here $a_i$ is a policy action and $s_i$ is its configured joint scale. This
# connects the action vector to the joint-angle picture we just explored.

# %%
%%robot
print("Actor observation shape:", observations["actor"].shape)
print("Critic observation shape:", observations["critic"].shape)
print("Action dimension:", env.action_manager.total_action_dim)
print("Actor terms:", list(env.cfg.observations["actor"].terms))
print("Critic terms:", list(env.cfg.observations["critic"].terms))
print("Joint target scales:", env.cfg.actions["joint_pos"].scale)

# %% [markdown]
# | Quantity | Used by deployed actor? | Purpose |
# |---|---|---|
# | Angular velocity and gravity direction | Yes | Sense rotation and tilt |
# | Joint positions/velocities and previous actions | Yes | Describe posture and recent motion |
# | Forward-speed and yaw-rate commands | Yes | Specify the requested movement |
# | True base linear velocity and contact information | No; critic/rewards may use them | Help training and evaluation |
# | World position, absolute heading, terrain map | No | No hidden navigation controller |
#
# **Actor:** chooses the next action. **Critic:** estimates how useful that action
# may be during learning. Only the actor drives the deployed robot.

# %%
media_pair(
    [
        (
            "The control loop",
            WORK / "assets/lesson/figures/control_loop.svg",
            "The simulator supplies rewards during training. Privileged measurements do not become actor inputs.",
        )
    ]
)

# %% [markdown]
# ### Define what good walking means
#
# A reward combines several preferences:
#
# $$r_t = w_v r_{\mathrm{speed}} + w_\omega r_{\mathrm{turn}}
#       + r_{\mathrm{posture}} + r_{\mathrm{feet}} - c_{\mathrm{motion}}.$$
#
# This is a grouped explanation; the next cell prints the actual terms and
# weights. For example, an exponential tracking reward has the form
# $r_{\mathrm{track}}=\exp(-e^2/\sigma^2)$: small tracking error earns more reward.
# Simulator measurements may score the task without being given to the actor.

# %%
%%robot
from prettytable import PrettyTable

rewards = PrettyTable(["Reward term", "Weight", "Parameters"])
for name, term in env.cfg.rewards.items():
    parameters = {
        key: value
        for key, value in term.params.items()
        if isinstance(value, (int, float, str, bool, tuple))
    }
    rewards.add_row([name, term.weight, str(parameters)])
print(rewards)

# %% [markdown]
# Speed/yaw rewards ask for command tracking. Posture and foot terms encourage
# usable movement. Motion penalties discourage excessive effort or abrupt changes.
# Later we will change one interpretable reward setting and inspect the result.
# A higher return after changing weights does **not** automatically mean a better gait.
#
# **Discuss:** why could rewarding only forward speed produce a poor walker?
