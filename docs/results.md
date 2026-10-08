# Results and scope

Raw measurements, checkpoints, configurations and videos belong in
`experiments/<timestamp>_<name>/`, which is gitignored. Each new training run
records resolved environment settings and a source snapshot. Selected media
used in the lesson are copied to `assets/lesson/`.

The initial fixed-speed examples were useful demonstrations, but their
zero-falls results did not establish directional control. A later 15-second
test measured roughly 0.78 m of average maximum lateral deviation for the
original walking policy. Those recordings are labeled preliminary.

The new command-policy study uses the same proprioceptive actor with a
privileged critic. It measures forward/backward speeds, stopping, both yaw
directions, command transitions, and drift relative to the starting heading.
No position or heading correction is supplied to the controller.

Early balanced-command evaluations show approximately 0.08 m maximum drift
at normal forward speed, 0.14 m at fast speed, and 0.19 m backward, averaged
across five perturbed starts over 15 seconds. The selected checkpoint passed
all ten command cases, including the declared drift thresholds, with no falls.
It was fine-tuned through several experiments; it is not a from-scratch timing
claim. Independent training and style comparisons are still being completed.

The browser game has passed a short local interaction test, including
walking → hopping → walking and pause, with no browser errors or falls.
Repeated flat-ground tests also completed 15 walking-to-hopping and 15
hopping-to-walking transitions across five starts without falls. Two initial
physical course attempts failed at the first pipe. Small-step training is in
progress; flat-ground success does not establish obstacle capability.

All reported timings so far use a local NVIDIA L40S. Some jobs share the GPU.
Actual Google Colab execution and classroom-GPU timing remain untested.
Evaluation starts are not independent training seeds.

The earlier recordings are also available in the
[instructor W&B gallery](https://wandb.ai/afloresa/bdxr-fastsac-classroom/runs/mhyyosrc).
Current experiment directories contain their own W&B run links.
