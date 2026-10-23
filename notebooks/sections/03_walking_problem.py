# %% [markdown]
# ## 3 · Define the walking problem
#
# **Objective:** stay upright and follow the requested forward speed and turn rate.
# Zero turn rate asks for approximately straight walking; commands can also request
# stopping or reversing. The policy receives the robot state and these commands.
# With no movement requested, aim for quiet standing with small balance corrections.
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
# describe simulated motion and foot contact. These extra inputs are training-only.
#
# ### Your activity · verify the interface
#
# Print the actual observation terms and action scales below. Use the table to
# explain why true base linear velocity is not available to the deployed actor.

# | Actor term | Scalars | Where it comes from |
# |---|---:|---|
# | Angular velocity | 3 | Gyroscope-like body rotation measurement |
# | Gravity direction | 3 | Gravity projected into the body frame; an orientation estimate |
# | Joint positions | 14 | Encoder-like angles relative to their joint references |
# | Joint velocities | 14 | Joint velocities relative to their references |
# | Previous actions | 10 | The controller's own previous leg commands |
# | Movement command | 3 | Requested forward, sideways, and turning velocities |
#
# **47 = 3 + 3 + 14 + 14 + 10 + 3.** The model measures 14 joints, but this
# locomotion policy commands the ten leg joints. The sideways command is held
# at zero in this lesson. Commands are instructions, not sensor measurements.
#
# The critic adds 15 scalars: true linear velocity (3), two foot heights (2),
# airtimes (2), contacts (2), and contact-force components (6). Thus its input
# length is **62**. These simulator-only measurements help learning, but the
# deployed actor cannot read them. In particular, it does not know world position
# or heading and cannot secretly correct its path with a navigation controller.

# %%
%%robot
print("Actor observation length per robot:", observations["actor"].shape[-1])
print("Critic observation length per robot:", observations["critic"].shape[-1])
print("Action vector length per robot:", env.action_manager.total_action_dim)
print("Parallel robots in this batch:", observations["actor"].shape[0])
for group in ("actor", "critic"):
    terms = env.observation_manager.active_terms[group]
    shapes = env.observation_manager.group_obs_term_dim[group]
    print(group, dict(zip(terms, shapes)))
print("Joint target scales:", env.cfg.actions["joint_pos"].scale)

# %% [markdown]
# | Quantity | Used by deployed actor? | Purpose |
# |---|---|---|
# | Gyroscope angular velocity; estimated gravity direction | Yes | Sense rotation and tilt |
# | Joint encoders/velocity estimates; previous actions | Yes | Describe posture and recent motion |
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
# | Reward preference | What it encourages |
# |---|---|
# | Speed and turn tracking | Move according to the command |
# | Upright posture | Keep the torso balanced |
# | Foot airtime and clearance | Lift and place the feet during stepping |
# | Quiet standing | Keep both feet supported when no movement is requested |
# | Low planted-foot rotation | Turn by stepping instead of twisting planted feet |
# | Small action changes | Avoid abrupt changes in motor targets |
# | Low sustained yaw/sideways bias | Reduce persistent turning and sideways motion |
#
# These preferences can compete: excessive smoothness can discourage useful
# stepping, while chasing speed alone can produce unstable movement. Training
# finds a compromise rather than executing a hand-written sequence of leg poses.
#
# The next cell prints actual weights and parameters. These terms have different
# units and scales; the largest weight need not make the largest contribution.

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

# %% [markdown]
# ### When does an episode end?
#
# A task termination ends an unsuccessful attempt; a time limit ends a full
# practice episode. The simulator then resets for another attempt. These endings
# must be distinguished when learning how much future reward remains.
# Here, a fall means torso tilt beyond 70° or base height below 0.12 m.
# Inspect the actual rule parameters below; angles are stored in radians.

# %%
%%robot
print("Episode time limit:", env.cfg.episode_length_s, "seconds")
for name, rule in env.cfg.terminations.items():
    print(name, "timeout:", rule.time_out, "parameters:", rule.params)
