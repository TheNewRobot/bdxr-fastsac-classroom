# Classroom validation · October 22, 2026

The core now has seven walking activities, with 31% less explanatory text and
38% fewer displayed code lines than the previous version. Students train a baseline and one
reward variation, briefly assess each, then drive their own walking policy.
Hopping is separate. No pretrained substitution, recovery chain, or checkpoint
search runs in the required lesson.

## Fresh training evidence

The classroom recipe keeps 12,000 requested steps, a 17-minute training cap,
a Colab GPU runtime, 512 environments, seed 7, and the visible notebook settings.
The earlier local trials below used a 15-minute cap. We inspect ordinary
**final** checkpoints. The actor has 47 proprioceptive/command inputs and ten
outputs; the critic has 62 inputs. No world-position or heading feedback drives
the actor.

| Run | Completed steps | Training time | Three 10-second forward trials |
|---|---:|---:|---|
| First seed 7 | 12,000 | 718 s | No falls; speed error 0.031 m/s, mean max drift 0.032 m; passes |
| Seed 8 sensitivity check | 12,000 | 854 s | No falls; speed error 0.084 m/s, mean max drift 0.093 m; **speed check fails** |
| Default seed 7 repeat | 11,074 (time cap) | 901 s | No falls; speed error 0.027 m/s, mean max drift 0.070 m; passes |
| Foot-level pilot | 11,525 (time cap) | 901 s | No falls; speed error 0.013 m/s, mean max drift 0.113 m; passes; unequal budget |
| Full-budget seed 7 baseline (17-minute allowance) | 12,000 | 947 s | No falls; speed error 0.029 m/s, mean max drift 0.017 m; passes |
| Full-budget foot-level variation (17-minute allowance) | 12,000 | 899 s | No falls; speed error 0.052 m/s, mean max drift 0.092 m; **speed check fails** |

The assessments took 26–34 seconds each on the local L40S. Existing speed,
yaw, and drift tolerances were retained; only the classroom scope was shortened.
Different seeds are not guaranteed to learn equally well. This GPU is shared
with other desktop work, so these are measured wall times, not dedicated-GPU or
Colab performance claims. The default-seed repeat remained useful despite completing fewer steps; it does not establish identical outcomes.

The full-budget variation walks at about 0.248 m/s for a 0.30 m/s command.
Its feet are flatter during swing (5.6° versus 8.7° for the matched baseline),
but it misses the unchanged speed tolerance. This is a measured reward tradeoff,
not an unqualified improvement. The fresh baseline with the same 17-minute
allowance passed. Their saved configurations differ only in the sole-level reward
weight, experiment name, and folder.
Foot landings increased from 3.70 to 4.48 per second. This is a modest gait
change from one matched pair, not evidence that every seed improves.
W&B confirms both full-budget comparison runs finished at step 12,000; their
learning metrics also remain in each local experiment record.

The first policy also passed seven development-only command checks, including
reverse, stop, and left/right turns (21 trials; 109 s). It completed all four
walking-course checkpoints and the raised finish with no falls in **62.06
simulated seconds**. The time-capped default-seed repeat also completed the
same course with no falls, in **42.70 simulated seconds**. An operator steered
from rendered images with pauses;
this is traversal evidence, not a real-time driving benchmark or autonomous
navigation. See the [course video](../assets/lesson/classroom_walking/course.mp4)
and [media provenance](../assets/lesson/classroom_walking/provenance.json).

## Execution and presentation

All 32 core code cells executed from the public release in a fresh local lesson
folder and dependency environment, with short integration budgets and no errors.
Setup took 11 seconds with package caches available; this is not a cold-download
or Colab setup estimate. The separate hopping notebook also executed all eight
stages at short budgets using the pinned local environment. These runs check code
paths, not learned-policy quality. Exported baseline/variation and eight hopping
checkpoints, including resumable states, passed ZIP integrity checks.
Training progress remains visible as separate lines, with complete logs and errors
saved. The earlier single updating display was removed after classroom feedback.

The local browser passed movement-key release, turning, reverse limits, stop,
pause, camera, reset, and checkpoint-load checks. The actual knee slider changed
its rendered image in JupyterLab. Instructor figures remain in Markdown; student
results remain executable. The regression suite ran 106 tests, with its
optional simulator contract also passing separately against the new policy.
Anonymous downloads of the release ZIP, bootstrap, checksums, and all eight
Markdown images/animations matched their expected hashes. The main notebook has
no saved student outputs. Local rendering confirmed the example layout.

## What the reported Colab failure establishes

The latest supplied evaluation has 50 trials with no falls, but substantial drift
and command-tracking misses. It took 255 seconds. Its checkpoint prefix is
`2bf8502c499d`; the corresponding raw checkpoint, GPU, and actual training-step
count are absent from the supplied records. We cannot determine its exact cause.
The older saved L4 notebook is a different run and must not be used to fill those
gaps. Local runs confirm that poor command tracking is possible even without
falls. The short assessment now prints saved training steps, duration, and GPU
to make subsequent reports easier to diagnose.

The published notebook opened in Colab's anonymous frontend, where all eight
embedded images and animations loaded. No authenticated GPU runtime was
available to the developer. The user-reported GPU run below adds evidence, but does
not yet justify calling the lesson classroom-ready on Colab GPUs, including T4.

## T4 setup correction

A reported T4 session was stopped by our L4-only preflight, before installation.
That artificial model-name restriction has been removed. Setup requires a GPU
and reports its name; dependency/runtime checks still apply. The 12,000-step
target and 17-minute cap are unchanged. A subsequent user-reported T4 run
completed 12,000 steps in 1,024 seconds; local L40S timings are not T4 promises.

## Reported T4 recording slowdown and graphics repair

The supplied checkpoint `8f518d7436e8` completed 12,000 training steps in
1,024 seconds on Tesla T4. Its short forward check passed in 33 seconds:
zero falls in three starts, speed error 0.032 m/s, yaw error 0.027 rad/s,
and mean maximum drift 0.143 m. This establishes short forward walking,
not turning or course completion.

Recording its ten-second video took 391 seconds. The supplied recording log
has no exception; all 31 listed Warp kernel loads were cached (0.137 seconds
combined). The user's graphics probe reported Mesa / llvmpipe: CPU software
rendering despite CUDA training. The old log does not separate simulation,
rendering, and encoding time, so the exact contribution is unknown.

Setup now probes a rendered frame in a fresh process. When NVIDIA is not
selected, it tries a workspace-local EGL vendor manifest and adopts that
setting only after verifying an NVIDIA renderer. It does not replace drivers
or change training. Diagnostics stay in `.runtime/rendering.json`; an
unsuccessful repair reports the problem and preserves the previous selection.
Five mocked selection/failure tests pass. Local L40S testing verifies actual
NVIDIA frame rendering and recovery from an invalid vendor selection.
This machine has no Mesa EGL vendor installed, so the exact llvmpipe-to-NVIDIA
transition and the resulting T4 recording duration still require Colab retesting.

## Joint explorer correction

Cell 10 now uses browser-side pose switching in Colab instead of Python widget
callbacks. The rendered poses, foot coordinates, and path marker update without
waiting for the kernel to service an interaction during training. Local Jupyter
keeps its standard widget controls. Browser interaction and generated poses are
checked locally; authenticated Colab interaction remains unverified.
