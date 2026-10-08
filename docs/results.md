# Results and scope

Raw measurements, checkpoints, configurations and videos belong in
`experiments/<timestamp>_<name>/`, which is gitignored. Each new training run
records resolved environment settings and a source snapshot. Selected media
used in the lesson are copied to `assets/lesson/`.

Read each recording's caption and provenance before comparing policies.
Earlier fixed-speed recordings are marked preliminary; attractive motion and
zero falls alone do not establish directional control.

For command walking, inspect speed/yaw tracking, straight-line drift, falls,
and command transitions. The declared acceptance thresholds are in
`configs/evaluation_targets.json`. World positions and headings measure the
result; they never correct the deployed controller.

For style experiments, compare against an unchanged continuation from the same
parent with the same training budget. For hopping, inspect actual flights and
stable landings. Requested height is not achieved jump height, and success on
flat ground is not evidence of obstacle capability.

Detailed reports live beside each checkpoint, including failed trials, source
snapshots, timings, and W&B links. Curated lesson media retain their provenance.
Five perturbed evaluation starts test one trained policy; they are not five
independent training seeds.

Timing reports identify hardware and shared-GPU execution. Local notebook
execution does not establish actual Google Colab performance or compatibility.
