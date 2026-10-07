# Short demo class

## Learning objectives

Students should be able to:

1. Identify the observations, actions, and reward in a locomotion task.
2. Explain what the actor, critic, and replay buffer do in SAC.
3. Connect joint-angle trajectories to visible robot motion.
4. Evaluate a reward change using both video and measurements.

## Suggested 12-minute presentation

| Minutes | Activity | Prompt |
| --- | --- | --- |
| 0–2 | Show the complete BDX-R and a walking result | “What has to happen for the robot to avoid falling?” |
| 2–4 | Identify observations, actions, and rewards | “We request joint positions. What produces the forces?” |
| 4–6 | Show the initial and trained policy | “What improved? What would a reward curve miss?” |
| 6–8 | Show speed and left/right knee angles | “How does changing configuration produce motion?” |
| 8–10 | Compare a walking variation | “What changed, and what stayed fixed?” |
| 10–12 | Explain SAC and discuss an experiment | “How could a robot maximize reward in an unwanted way?” |

Prepare training before class if only 12 minutes are available. Installation,
GPU compilation, training, and video generation are separate from presentation
time. A student lab can run training live, subject to the assigned GPU.

The notebook's walking preset uses 512 environments, a 2,048-sample update
batch, seed 7, and at most 6,000 training steps. The recorded gallery uses
the separately documented 1,024-environment instructor runs.

## Concise explanations

**Observation:** information supplied to the policy: body angular velocity,
gravity direction, joint positions and velocities, previous actions, and command.
The walking actor receives 47 numbers; the critic receives 62, including extra
simulation measurements. The hopping extension adds two phase signals.

**Action:** ten normalized leg commands. The environment scales these to
joint-position offsets, and position controllers produce bounded actuator forces.
There are five controlled joints per leg. The complete head remains in the
physical model and its actuators target a neutral position.

**Reward:** a weighted combination of speed tracking, upright posture, posture
regularization, and foot-motion terms, with penalties for abrupt or unwanted
motion. The implemented terms are in `environment.py` and the pinned BDX-R task.
Reward weights are experiment settings, not universal definitions of good walking.

**SAC:** the actor chooses actions; critics learn future values from replayed
experience; an entropy term encourages exploration. Our adapter uses Holosoma's
FastSAC networks, distributional critics, normalization, and replay buffer.
The optional distributional-critic details are beyond this 101 lesson.

**Kinematics versus dynamics:** joint angles describe configuration; trajectories
describe motion. Gravity, actuator forces, and contact dynamics determine whether
that motion is physically achievable.

**Fine-tuning versus inference:** changing a command for an existing policy is
not retraining. The notebook's comparison explicitly fine-tunes on the changed
task and starts a fresh replay buffer.

## What the measurements mean

- Training episode metrics include exploration and resets.
- Evaluation turns exploration off and reports every evaluated episode.
- A speed target is compared with measured body-frame forward speed.
- Foot-contact changes help distinguish stepping from sliding.
- Hopping requires both-foot clearance and an upright landing; the initial
  drop onto the floor is excluded.
- More reward does not necessarily mean more appealing or more robust motion.

## Scope

This is a simulation teaching exercise. It does not establish hardware deployment
readiness. Flat-ground evaluation and modest initial-orientation variation are
not comprehensive robustness tests.

The original reference notebook used PPO and BipedalWalker. This notebook teaches
SAC with BDX-R; do not reuse the old PPO-specific actor/critic explanation unchanged.

## Optional precision experiment

The command-line trainer supports `--amp fp16` and `--amp bf16`; the notebook
keeps full precision as its default. FP16 was tested locally on the L40S with
512 environments and seed 7. Its 4,000-step checkpoint completed five
perturbed-start 15-second evaluations without falling, averaging 0.282 m/s.
Gradient-scaler state is saved and restored in continuation checkpoints.

This is an option to benchmark on the actual classroom GPU, not a measured
Colab speedup. Neither T4 execution nor actual Google Colab runtime execution
has been validated here.
