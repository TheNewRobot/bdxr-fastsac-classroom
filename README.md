# Teach a little droid to walk

A 163A/263A robotics lesson using **BDX-R + FastSAC + MuJoCo Warp**.
Train, measure, change one parameter, and play a small desert-droid course.

![Desert Droid Run interface](assets/lesson/course_preview.png)

**Status:** the full lesson has completed local execution from fresh training,
including command walking, hopping, policy handovers and exports. Browser
controls and exported checkpoints have been checked. The fresh policies
completed the course without falling. Curated style examples passed evaluation;
fresh style attempts can fail and are labeled accordingly.
Actual Google Colab execution has not been performed.

- [Student notebook](notebooks/BDXR_Walking_101.ipynb)
- [Short instructor notes](docs/instructor_notes.md)
- [Measured results and their limits](docs/results.md)
- [Implementation status](docs/progress.md)
- [Original class notebook](ref/25F_163A_263A_BipedalWalker_SeminarRL101.ipynb)

Upload the notebook to Colab and select a GPU runtime. It includes lesson source
and curated recordings, then fetches pinned upstream robot assets. Publishing
this repository is not required.

## What the policy sees

The actor receives joint positions/velocities, body angular velocity, gravity
direction, previous actions, and requested speed/yaw rate. It does not receive
true base linear velocity, world position, absolute heading, or a terrain map.
The training critic may use privileged simulator observations.

The deployed controller follows user commands. There is no planner or external
heading correction. World measurements are used for evaluation and game scoring.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/bdxr_classroom/` | Simulator, FastSAC, evaluation, and game |
| `notebooks/` | Portable student lesson |
| `configs/` | Pinned upstream revisions |
| `assets/lesson/` | Only selected media used by the lesson |
| `scripts/` | Setup, comparisons, and notebook packaging |
| `tests/` | Observation, learning, and game contracts |
| `experiments/<timestamp>_<name>/` | **Gitignored** configs, checkpoints, metrics, evaluations, and videos |

Vendor checkouts, caches, raw results, checkpoints, and private working notes
stay out of new commits. Curated media captions distinguish preliminary results
from selected command policies.

## Local use

```bash
python scripts/setup.py
export PYTHONPATH="$PWD/src"
export MUJOCO_GL=egl
.venv/bin/python -m bdxr_classroom.train \
  --name command_walking --num-envs 512 --batch-size 2048 --seed 7 \
  --steps 8000 --minutes 15 --command-mode balanced \
  --yaw-tracking imu --yaw-std .2 --yaw-bias-weight 2 \
  --lateral-bias-weight 2 --mirror-replay --compile
```

These settings match the notebook's walking recipe. Runtime depends on the GPU;
the time limit can stop training before 8,000 steps. The trainer prints the
actual timestamped experiment path. To evaluate it:

```bash
.venv/bin/python -m bdxr_classroom.command_evaluate \
  experiments/<timestamp>_command_walking/checkpoints/final.pt
python scripts/check_command_results.py \
  experiments/<timestamp>_command_walking/evaluation/final_commands/summary.json
```

Resume with `--resume .../checkpoints/resume.pt`; policy, critics, optimizers,
normalization and replay are restored, while simulation episodes restart.
Use `--finetune` to start a new experiment from a compatible training checkpoint.
Fine-tuning uses the requested seed for a new random stream; resuming restores
the checkpoint's random stream.
W&B is on by default; students can use `--wandb-mode disabled`.

Launch the game with walking and optional hopping checkpoints:

```bash
.venv/bin/python -m bdxr_classroom.game_server \
  --walking experiments/<walking-run>/checkpoints/final.pt \
  --hopping experiments/<hopping-run>/checkpoints/final.pt
```

The local panel runs on port 8765. The notebook includes the Colab port-proxy
integration. Falls reset the robot; simulation time and fall count accumulate.
The calibrated course uses three 17.5–20 mm obstacles, rough ground and an
18 cm ascent. Four route checkpoints must be crossed before the finish counts.
Use `--course-config path/to/settings.json` to explore a harder course;
`{"obstacle_scale": 1.0}` doubles the classroom obstacle heights.

## Credits

[BDX-R](https://github.com/BDX-R/BDX-R-MjLab) provides the robot and starting task.
[Holosoma / FastSAC](https://younggyo.me/fastsac-humanoid/) provides the learning
components. The desert outpost is original procedural scenery.
See [third-party notices](THIRD_PARTY_NOTICES.md) and [citation metadata](CITATION.cff).
