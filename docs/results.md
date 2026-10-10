# Results and scope

Raw measurements, checkpoints, configurations and videos belong in
`experiments/<group>/<timestamp>_<name>/`, which is gitignored. Start with
the local `experiments/INDEX.md`. Each new training run
records resolved environment settings and a source snapshot. Selected media
used in the lesson are copied to `assets/lesson/`.

Read each recording's caption and provenance before comparing policies.
Retired preliminary recordings stay in local archives. Attractive motion and
zero falls alone do not establish directional control.

For command walking, inspect speed/yaw tracking, straight-line drift, falls,
and command transitions. The declared acceptance thresholds are in
`configs/evaluation_targets.json`. World positions and headings measure the
result; they never correct the deployed controller.

For style experiments, train both the variant and an unchanged control from
scratch with the same seed and step budget. A failed control remains a
comparison result, not a validated baseline. For hopping, inspect actual flights and
stable landings. Requested height is not achieved jump height, and success on
flat ground is not evidence of obstacle capability.

Detailed reports live beside each checkpoint, including failed trials, source
snapshots, timings, and W&B links. Curated lesson media retain their provenance.
Unused trials are consolidated into local evidence archives; obsolete payloads
are listed in cleanup manifests.
Five perturbed evaluation starts test one trained policy; they are not five
independent training seeds.

Timing reports identify hardware and shared-GPU execution. Local notebook
execution does not establish actual Google Colab performance or compatibility.
