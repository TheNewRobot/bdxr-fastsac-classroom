# Short demo class

| Minutes | Activity |
| --- | --- |
| 0–2 | Show the real droid reference beside simulated walking |
| 2–5 | Follow one leg through its joints and move the knee slider |
| 5–7 | Inspect the environment, observations, actions, and reward |
| 7–9 | Explain actor, privileged critic, replay, and one collection step |
| 9–12 | Compare a scratch-trained reward variation and its control |
| 12–15 | Show hopping and introduce the interactive course |

Prepare training beforehand for a short presentation. Installation, training,
video generation and gameplay are separate from presentation time.

The actor receives proprioception and commands; the critic may use additional
simulator information during training. The actor controls ten leg joints.

Speed is not straightness. A robot can move forward in its own body frame
while curving across the floor. Trajectory and yaw-rate plots expose this.
Evaluation position measurements do not become policy inputs.
Sideways motion can also cause drift without much turning. Simulator velocity
can score this during training while staying out of the actor's observations.

Changing the speed command differs from retraining a reward setting.
Train the style and its control from scratch with the same seed and step budget.
The recorded choices are flatter feet, quicker steps, and crouched walking.
Recheck backward motion, stops and turns after a style change.
An unchanged continuation can also get worse: more updates do not guarantee
better control. Use the acceptance table to decide which policies to keep,
and discuss failed experiments instead of hiding them.

Hopping phase is a command. Foot clearance is not body jump height, and a
successful flat-floor hop does not prove obstacle clearance. Mode transitions
and course traversal require their own tests.

The game is a human-controlled validation task, not a navigation policy.
The 25 mm middle step has a recorded walking-versus-hopping comparison.
Replaying fixed commands can miss the finish; players must steer.
Ask students to explain a failure using the video, commands and measurements.

For a hands-on class, run sections 1–7 and reserve extra time for GPU setup,
training, evaluation, and rendering. Section 8's eight-stage hopping curriculum
is optional; its recorded videos are available without retraining. Section 9
accepts compatible uploaded checkpoints as well as student-trained policies.
Section 10 exports configurations, metrics, videos, and resumable checkpoints.

The notebook's `%%robot` cells share a pinned simulator process. Readable model
and environment code stays in the lesson; package installation and repetitive
subprocess handling are collapsed or delegated to the lesson helpers. If that
process is restarted, rerun environment creation before later simulator cells.
