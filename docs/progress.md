# Validation · October 23, 2026

## Standalone hopping lesson

The independent lesson now trains a single stationary hopper, checks three
button-triggered hops, and opens a flat practice pad with Jump, Reset and optional
recording. It needs no walking policy. The walking notebook is unchanged.

A fresh seed-7 run of `configs/stationary_hopping.json` completed **6,000 steps in
592 seconds** on a shared local L40S (608 seconds including process startup).
Its three full-collision trials cleared **5.63–5.74 cm**, each with **0.20 seconds**
of flight, no falls, and no foot lifts during two seconds of stance after landing.
The recorded check took **25.9 seconds**, excluding Python process import time.
Physics and the same actor run continuously between Jump requests.

The earlier eight-stage curriculum completed 24,000 steps in 34.1 training
minutes but produced no flight in the final three-click check. That failure is
retained locally; the new flat-ground recipe explicitly rewards bilateral flight.
These are single-seed results, not evidence of reliable Colab reproduction.
All ten final notebook code cells executed locally with a short training budget.
A second execution reused the exact separately trained 6,000-step weights and
passed the recorded check in **35.7 seconds including process startup**, followed
by two live browser jumps, Reset, recording on/off, and desktop/mobile checks.
The same actor kept running while idle. An initial browser test raced the Reset
response; waiting for the backend to finish settling fixed the test.
All 144 regression tests passed before the recipe update; all 14 affected
notebook/hopping tests passed again afterward. Rendered layouts were inspected.
Actual Colab execution and a fresh dependency installation remain unverified.

## Reported Colab run

The user's T4 run completed both 12,000-step trainings in about 982/980 seconds
with the supplied settings. The baseline failed its short forward check:
mean speed was about **0.018 m/s** for a 0.30 m/s command. The variation passed
at about **0.270 m/s**. Replaying the uploaded baseline locally reproduced a fall
1.62 seconds after Forward. The uploaded variation completed the local 44-second
forward/reverse/turn/stop/restart sequence without falls or settled idle stepping.
This is not a full-course guarantee or proof of reliable baseline training.

The lesson now summarizes both existing checks and defaults to a passing policy,
preferring baseline only when it passes. Failed, missing or stale checks cannot
qualify a selection. No extra training, checkpoint search or assessment is added.
The exact cause of the baseline's poor learning remains unresolved.

## Earlier local campaign

These are **local L40S measurements**, with other desktop training sharing the
GPU. Actual Google Colab execution remains unverified. The core lesson trains
from scratch; it does not substitute pretrained weights or search checkpoints.

## Current classroom pair

The recipe targets **12,000 steps**, with a **22-minute allowance per run**.
Commands change every two seconds. Quiet-standing, idle-motion and planted-foot
rotation rewards encourage stopping without repetitive stepping. The seed is 7.

| Final policy | Steps | Training | Short walking check | Swing-foot tilt | Foot landings/s |
|---|---:|---:|---:|---:|---:|
| Baseline | 12,000 | 1,182 s | Pass · 23.7 s | 9.7° | 4.11 |
| Sole-level reward = 0.75 | 12,000 | 923 s | Pass · 30.4 s | 5.2° | 4.52 |

Only the sole-level reward changes among the environment and learning settings.
The selected runs had 20/22-minute limits, but neither reached its limit. Both
completed the same step count. This pair shows flatter feet and slightly quicker
steps; it does not establish that every seed will produce the same change.

Both policies also passed three-start forward, reverse, left/right turn,
stop and restart checks: **no falls, stalled starts, or settled idle foot lifts**.
These deeper development checks are separate from the brief classroom assessment.
An earlier fresh run of the same baseline recipe also passed in 1,035 seconds.
The actor still receives proprioception and commands, without true linear
velocity, world position, heading, or terrain information.

## Full-collision game

| Policy/course | Simulated completion time | Falls |
|---|---:|---:|
| Baseline · walking course | 40.60 s | 0 |
| Reward variation · walking course | 40.98 s | 0 |
| Baseline + existing hopper · optional course | 85.68 s | 0 |

An operator used the button command palette, pausing simulation between holds.
These are traversal demonstrations, not real-time human driving benchmarks or
robustness guarantees. The baseline's first walking-course attempt clipped gate
three and fell; correcting the approach produced the recorded completion without
changing the policy or terrain. The optional course's steeper ramp required slow
progress and heading corrections.

The demo uses all 81 visible-link mesh geometries with convex contacts, including
the physical head. A captured nonfinite multi-point contact failure was reproduced
and fixed with single-point convex contacts. Training keeps its simpler geometry.
Both walkers passed the recorded game transition sequence with quiet stops.
Direct Hop → Stop → restart also passed with the baseline and existing hopper.

## Execution and preservation

All 32 core code cells executed locally using the pinned environment: the
baseline completed 12,000 steps; the first variation hit its 20-minute cap at
11,400. The notebook retained the unequal-budget warning. A fresh variation then
completed 12,000 steps and used the same recording, assessment and gait helpers.
The capped run also passed its behavioral checks, but is not the published
matched comparison.

Both final notebook workflows passed short integration runs: 32 walking code
cells and seven hopping code cells, including all eight hopping stages.
These test execution paths, not learned hopping quality. **All 129 regression
tests passed**, including browser controls, CUDA, collision coverage and opt-in
replay. All 11 instructor images/animations loaded in local rendered reviews.
The release bundle's code is identical to the integrated bundle; two media
metadata files changed when the comparison GIF was resized. The final notebook's
unpack cell extracted the release bundle and every file matched its archive.

All 40 protected checkpoints retain their original hashes. New selected policies
are under `experiments/lesson/`; tuning failures, capped runs, configurations,
source snapshots, resumable checkpoints and W&B records are retained locally.
See the [media provenance](../assets/lesson/classroom_walking/provenance.json),
[reward comparison](../assets/lesson/classroom_walking/reward_comparison.json),
and local `experiments/INDEX.md`.

Earlier tuning produced idle stepping, stalled commands and falls. Those results
remain in the experiment records. A user-reported earlier Colab T4 recipe trained
12,000 steps in 1,024 seconds and passed its 33-second check; that does not validate
this revised recipe on Colab. Setup, download, rendering and interactive-proxy
behavior still need an actual Colab session before advertising classroom readiness.
