# Implementation record

## October 7, 2026

- Local GPU: NVIDIA L40S, 46,068 MiB reported by nvidia-smi.
- W&B authentication verified; classroom project: `bdxr-fastsac-classroom`.
- Fetched pinned BDX-R and Holosoma revisions (see `configs/upstream_revisions.json`).
- BDX-R's recorded MuJoCo nightly wheel is no longer downloadable. Replaced it
  with release MuJoCo 3.8.1 and compatible MuJoCo Warp 3.8.0.3 / Warp 1.12.1.
- GPU simulation and headless rendering passed with the complete robot.
- Actor controls ten leg joint position targets; head actuators retain neutral
  targets and the model retains the head mass.
- FastSAC uses upstream Holosoma actor, categorical critics, normalization and
  replay buffer, with a classroom mjlab training-loop adapter.
- Small learning smoke run completed, wrote checkpoints and real W&B metrics.
- First substantial walking experiment launched; quality is not yet established.
- Initial deterministic policy falls after approximately one second in the first
  three evaluation episodes. This is the actual before-training comparison.

No actual Google Colab runtime has been tested yet.

## First measured results

- `walk_02/step_003000.pt`: 151 seconds of local training, five 10-second
  evaluations, 0.27 m/s mean speed, no falls.
- `walk_02/step_004000.pt`: five 15-second evaluations, 0.28 m/s, no falls;
  repeated foot-contact changes confirm stepping.
- Selected comparison parent: `walk_02/step_006000.pt`, 0.29 m/s.
- `faster_01/step_002000.pt`: fine-tuned from that parent toward 0.45 m/s;
  0.445 m/s measured, no falls in five 15-second episodes.
- `smoother_01/step_002000.pt`: fine-tuned from the same parent with triple
  action-change penalty; 0.29 m/s and action-change RMS about 0.094 versus
  about 0.110 for the parent (roughly a 15% reduction).
- `hop_01/step_003000.pt`: tiny learned flights with stable landings.
- `hop_higher_01/step_003000.pt`: fine-tuned from `hop_01/step_004000.pt`
  with a stronger foot and body-height objective; visibly larger small hops.
  Final landing counts distinguish rebounds from stable landings.

These initial evaluations vary yaw. Final selected evaluations additionally
perturb initial joints and roll/pitch by up to 0.015 radians. Those reports
are written to each run's `eval_selected/` directory.

Timings use a warm simulator cache. Some training and evaluation jobs shared
the GPU, so they should not be interpreted as isolated performance benchmarks.

## Validation and packaging

- Fresh Python 3.12 environment installation passed.
- Sparse Holosoma checkout tested: approximately 1 MB, avoiding roughly 509 MB
  of unrelated assets.
- GPU tests passed for full-robot mass/head preservation, ten-action mapping,
  terminal versus reset observations, and replay/optimizer checkpoint continuation.
- Every notebook cell passed a short, real end-to-end execution:
  `artifacts/notebook_smoke_executed.ipynb`.
- Full notebook execution with 512 environments and W&B passed twice:
  the seed-42 diagnostic and the final seed-7 preset.
  Final baseline: 0.288 m/s, zero falls in five evaluations; final smoother
  variation: 0.293 m/s, zero falls in five evaluations.
  This remains local validation, not execution on Google's servers.
  Machine-readable evidence: `artifacts/final_notebook_validation.json`.
- Compact inference-only policies and compressed recorded videos are packaged
  in `examples/`; the notebook embeds these with its source bundle.
- Complete W&B gallery, including all seven selected training checkpoints:
  https://wandb.ai/afloresa/bdxr-fastsac-classroom/runs/mhyyosrc

## Additional behavior iteration

The four initial selected examples and independent 512-environment policy
passed five perturbed-start evaluations each. Full FP16 training and checkpoint
continuation were also tested, including saved gradient-scaler state.

The user's requested extension adds a slower target from the same walking
parent, and tests a higher hop target from the reliable hopping parent.
Each new experiment changes one behavior parameter. Final outcomes are
recorded in `results.md` and `results.json`.

Slower walking, higher-clearance hopping, and an unchanged training control
all passed five 15-second evaluations with zero falls. The smoother variant
reduces command-change RMS by about 16% relative to the control. Higher hop
targets increase measured foot clearance from about 6.3 to 9.2 cm while
retaining 14 settled hops per evaluation.

The updated notebook also passed a final short end-to-end execution. Videos,
learning curves, compact policies, and exact configurations are versioned;
full replay checkpoints and detailed execution artifacts stay gitignored.

## Remaining external validation

Actual Google Colab execution and timing on the GPU assigned to students
remain untested. Dependency installation, the complete notebook flow, videos,
W&B, and student ZIP export have been tested locally in a fresh environment.
The 15-minute training limit is a budget, not a guarantee of successful
learning on every Colab GPU or random seed.
