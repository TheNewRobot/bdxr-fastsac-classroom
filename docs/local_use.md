# Local setup and development

[Back to the lesson](../README.md)

Run these commands from the repository root.

```bash
python scripts/setup/setup.py
export PYTHONPATH="$PWD/src"
export MUJOCO_GL=egl
.venv/bin/python -m bdxr_classroom.learning.train \
  --name command_walking --num-envs 512 --batch-size 2048 --seed 7 \
  --steps 10000 --minutes 15 --command-mode balanced \
  --yaw-tracking imu --yaw-std .2 --yaw-bias-weight 6 \
  --lateral-bias-weight 2 --mirror-replay --compile
```

These settings match the notebook's walking recipe. Runtime depends on the GPU;
the time limit can stop training before 10,000 steps. The trainer prints the
actual timestamped experiment path. To evaluate it:

```bash
.venv/bin/python -m bdxr_classroom.evaluation.command_evaluate \
  experiments/candidates/<timestamp>_command_walking/checkpoints/final.pt
python scripts/evaluation/check_command_results.py \
  experiments/candidates/<timestamp>_command_walking/evaluation/final_commands/summary.json
```

Resume with `--resume .../checkpoints/resume.pt`; policy, critics, optimizers,
normalization and replay are restored, while simulation episodes restart.
Use `--finetune` to start a new experiment from a compatible training checkpoint.
Fine-tuning uses the requested seed for a new random stream; resuming restores
the checkpoint's random stream.
W&B is on by default; students can use `--wandb-mode disabled`.

Launch the game with walking and optional hopping checkpoints:

```bash
.venv/bin/python -m bdxr_classroom.game.server \
  --walking experiments/lesson/<walking-run>/checkpoints/final.pt \
  --hopping experiments/lesson/<hopping-run>/checkpoints/<selected-checkpoint>.pt \
  --course-config configs/classroom_course.json \
  --head-motion-config configs/head_motion.json
```

The local panel runs on port 8765. The notebook includes the Colab port-proxy
integration. Falls reset the robot; simulation time and fall count accumulate.
The calibrated course uses three 17.5–25 mm obstacles, rough ground and an
18 cm ascent. Four route checkpoints must be crossed before the finish counts.
Use `--course-config path/to/settings.json` to explore a harder course;
`{"obstacle_scale": 1.0}` doubles the classroom obstacle heights.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/bdxr_classroom/simulation/` | Robot environment, commands, terrain, and rewards |
| `src/bdxr_classroom/learning/` | FastSAC, replay symmetry, training, and policy export |
| `src/bdxr_classroom/evaluation/` | Rollouts, command checks, measurements, and plots |
| `src/bdxr_classroom/game/` | Course, controller handovers, scoring, and web interface |
| `src/bdxr_classroom/lesson/` | Notebook execution, visualizations, assessment, and exports |
| `notebooks/` | Portable notebook and readable chapter sources in `sections/` |
| `configs/` | Pinned revisions, evaluation targets, course, and hopping curriculum |
| `assets/lesson/` | Only selected media used by the lesson |
| `scripts/` | Entry points grouped into setup, lesson, training, evaluation, and media |
| `tests/` | Tests grouped by simulation, learning, evaluation, game, lesson, and storage |
| `experiments/lesson/` | **Gitignored** selected policy runs |
| `experiments/candidates/` | New or unfinished training runs |
| `experiments/evaluations/` | Course, obstacle, and controller checks |
| `experiments/support/` | Parent policies and historical media sources |
| `experiments/archive/` | Retired trial evidence and cleanup manifests |

Vendor checkouts, caches, raw results, checkpoints, and private working notes
stay out of new commits. Curated media captions distinguish preliminary results
from selected command policies.
Local results are explained in `experiments/INDEX.md`; its linked catalog records
individual trials. Each run keeps its `<timestamp>_<name>` directory and its
configuration, checkpoints, metrics, and videos together. New training runs go
to `candidates/`; selection requires review of their results.

Edit the teaching cells in `notebooks/sections/`, then rebuild the portable file
with `.venv/bin/python scripts/lesson/build_notebook.py`. It bundles the classroom
code and selected recordings; students do not need to clone the repository.
Instructor figures and short animated examples use public image links directly
in Markdown, visible without executing cells. Internet access is required for
these previews; Colab did not display the former Jupyter attachment links.
Student activities generate separate results.
The notebook downloads its versioned archive from a public GitHub release; it does
not read notebook metadata through the Colab frontend. Students need no manual ZIP
upload. Rebuilding writes release files under ignored `artifacts/lesson_releases/`;
publish the matching ZIP and checksum before sharing the rebuilt notebook.
