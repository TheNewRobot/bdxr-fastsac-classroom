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

# %% [markdown]
# ### Instructor example · the policy interface
#
# For one walking robot, the **actor observation vector has length 47** and the
# **critic observation vector has length 62**. These count scalar components,
# not networks. The actor produces an **action vector of length 10**, converted
# into leg-joint targets using the action scales.
#
# The actor observes proprioception and commands. The critic's extra components
# describe simulated motion and foot contact. Hopping adds two phase components,
# so its observation lengths are 49 and 64.
#
# ### Your activity · verify the interface
#
# Print the actual observation terms and action scales below. Use the table to
# explain why true base linear velocity is not available to the deployed actor.

# %%
%%robot
print("Actor observation length per robot:", observations["actor"].shape[-1])
print("Critic observation length per robot:", observations["critic"].shape[-1])
print("Action vector length per robot:", env.action_manager.total_action_dim)
print("Parallel robots in this batch:", observations["actor"].shape[0])
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

# %% [markdown]
# | Observations, actions and physics |
# |:---:|
# | ![Control loop: actor inputs, joint targets, physics, and rewards](lesson-asset:reference/control_loop.png) |

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

# %% [markdown]
# ### Instructor example · selected baseline reward terms
#
# | Term | Weight | Interpretation |
# |---|---:|---|
# | Forward-speed tracking | 8.0 | Follow the requested speed |
# | Turn-rate tracking | 4.0 | Follow the requested yaw rate |
# | Upright posture | 2.0 | Keep the body upright |
# | Foot airtime | 1.5 | Encourage a useful swing interval (0.10–0.60 s) |
# | Foot clearance cost | −4.0 | Penalize deviation from the 0.06 m target |
# | Action-change cost | −0.1 | Discourage abrupt target changes |
#
# These are selected terms from the actual baseline, not the complete reward.
# Weights multiply different quantities with different scales; a larger weight
# alone does not tell us which contribution dominates.
#
# **Your activity:** print the full reward configuration below and identify which
# term you would change to encourage quicker stepping.

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
