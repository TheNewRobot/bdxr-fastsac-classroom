# Short demo class

| Minutes | Activity |
| --- | --- |
| 0–2 | Show the droid and a real learned behavior |
| 2–4 | Identify observations, actions and rewards |
| 4–6 | Explain actor, privileged training critic and replay |
| 6–8 | Compare requested speed/yaw with measured motion and drift |
| 8–10 | Predict and inspect a one-parameter variation |
| 10–12 | Load policies and introduce the validation game |

Prepare training beforehand for a short presentation. Installation, training,
video generation and gameplay are separate from presentation time.

The actor receives proprioception and commands; the critic may use additional
simulator information during training. The actor controls ten leg joints.
The full head remains physically present and is held near its neutral pose.

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
