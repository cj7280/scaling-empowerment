# empowerment-warp

Code for *Scaling Empowered Agents*. SEE maximizes empowerment by pairing two
policies:

- a **potential policy** that moves the agent to states that are good starting
  points for many skills;
- a **skill policy** whose skills a learned InfoNCE classifier can tell apart.

Ant, AntSoccer and BuilderBench train with SAC on MJX-Warp physics. Craftax-Classic
trains with recurrent PQN. The VIM and LatentGCE baselines are included.

## Quick start

Use Linux with an NVIDIA GPU and a driver compatible with CUDA 12.
The project uses Python 3.11–3.13 and a locked dependency set managed by
[uv](https://docs.astral.sh/uv/getting-started/installation/). Use an up-to-date uv
release and the latest security patch of your chosen Python version.

```bash
uv sync --locked --extra test
cp .env.example .env
```

Set `PYTHON` in `.env` to the absolute path of `.venv/bin/python`. W&B logging
defaults to offline mode. Configure `RUN_ROOT` and the `SLURM_*` settings if you
use Slurm.

Inspect the resolved configuration, then train one seed:

```bash
bash scripts/run.sh resolve --config configs/ant/see.yaml
bash scripts/run.sh train --config configs/ant/see.yaml --seed 0 --run-dir artifacts/ant-see-0
```

- `--set key=value` overrides a setting.
- `--resume` continues a run in the same directory.
- `--max-chunks N` stops early without changing the schedule.

Each run directory holds the resolved config, metrics, checkpoints and
evaluations.

To run the tests on a GPU:

```bash
uv run --locked --extra test python -m pytest -q tests
```

## Recipes

[`configs/<environment>/<name>.yaml`](configs) holds every scientific setting.
`see.yaml` defines SEE for each main environment. SEE ablations extend that
recipe. VIM and LatentGCE have separate configurations.

| Recipe | Change |
|---|---|
| `see.yaml` | SEE. Continuous environments use joint MI; Craftax uses conditional MI on branched rollouts. |
| `cmi.yaml` | Conditional MI (continuous environments). |
| `joint_mi*.yaml` | Joint MI (Craftax). |
| `lambda0.yaml`, `lambda1.yaml` | Fixed λ = 0 or λ = 1 instead of the anneal. |
| `no_switch.yaml` | No switch: episodes run either the potential policy or a single skill. |
| `pair_decoder.yaml` | Pair-MLP classifier instead of the difference form. |
| `see_reset_memory.yaml` | Craftax: reset the recurrent memory at the switch. |
| `vim.yaml`, `latentgce.yaml` | Baselines, trained with PPO. |

The `ant` environment includes the AntSoccer ball and arena.

## Evaluation

```bash
bash scripts/run.sh evaluate artifacts/ant-see-0 --output artifacts/ant-see-0-eval
bash scripts/run.sh branch-media artifacts/ant-see-0 --output artifacts/ant-see-0-media
```

Use an output directory outside the source run. `branch-media` renders skill
branches for continuous SEE checkpoints.

Training also evaluates on a fixed schedule. SEE evaluation starts from 128 fresh
worlds, takes 100 potential-policy actions, then switches to a freshly sampled
skill:

| | Skill actions | Reported |
|---|---|---|
| Continuous environments | 300 | coverage and distance |
| Craftax | up to 900 | achievements |

Continuous VIM and LatentGCE evaluation runs their single controller for 100
actions. Reset and action keys are matched across methods for the same training
seed and evaluation target.

## Slurm

Set `SLURM_MEM` and `SLURM_TIME` in `.env` for the selected workload, or pass
`--mem` and `--time` explicitly. Preview a bounded submission before launching:

```bash
bash scripts/submit.sh --config configs/ant/see.yaml --seed 0 --max-chunks 1 --dry-run
```

Remove `--dry-run` to submit. For a full run, supply evaluation evidence from the
same recipe and code. Replace the example path with your evidence file:

```bash
bash scripts/submit.sh --config configs/ant/see.yaml --evaluation-evidence /path/to/evidence.json --dry-run
```

`submit.sh` freezes a copy of the source and submits the recipe's seeds as one
array; `--seed N` submits a single seed and `--dry-run` checks without submitting.
A full-length run requires `--evaluation-evidence`, a file listing matching
evaluation outputs. A one-chunk run alone need not produce all required evidence.
The format and checks are in
[`evaluation_checks.py`](src/empowerment_warp/evaluation_checks.py).

## Code

| Path | Contents |
|---|---|
| [`agents/`](src/empowerment_warp/agents) | `see_sac.py` and `see_pqn.py` (SEE agents), `skill_classifier.py` (shared classifier), `vim.py`, `latent_gce.py`, `ppo.py` |
| [`utils/`](src/empowerment_warp/utils) | `networks.py` (all modules), `objectives.py` (classifier losses), `sampling.py` (replay and hindsight batches) |
| [`core/`](src/empowerment_warp/core) | replay buffer; rollout collectors, including the switch and branch rollouts |
| [`training/`](src/empowerment_warp/training) | runners, configs, logging and checkpoints for SAC, PQN and the baselines |
| [`envs/`](src/empowerment_warp/envs) | Ant/AntSoccer, BuilderBench, Craftax-Classic |

A good reading order:

1. [`recipes.py`](src/empowerment_warp/recipes.py).
2. [`utils/objectives.py`](src/empowerment_warp/utils/objectives.py).
3. [`agents/skill_classifier.py`](src/empowerment_warp/agents/skill_classifier.py):
   - `_loss`, the classifier loss;
   - `_intrinsic_impl`, the skill reward with its λ-weighted log-partition;
   - `relabel_achieved_skills`, hindsight relabeling.
4. [`core/rollout.py`](src/empowerment_warp/core/rollout.py).
5. [`agents/see_sac.py`](src/empowerment_warp/agents/see_sac.py):
   - `ContinuousSAC.potential_reward`, the potential policy's reward;
   - `SEEAgent.update`.
6. [`agents/see_pqn.py`](src/empowerment_warp/agents/see_pqn.py), the same
   pieces for Craftax.

## Third-party code and license

- **BuilderBench assets and tasks**: from
  [BuilderBench](https://github.com/RajGhugare19/builderbench). MIT,
  [`envs/builderbench/LICENSE`](src/empowerment_warp/envs/builderbench/LICENSE).
- **Ant and AntSoccer models and tasks**: from
  [OGBench](https://github.com/seohongpark/ogbench). MIT,
  [`envs/locomotion/LICENSE`](src/empowerment_warp/envs/locomotion/LICENSE).
- **Craftax-Classic**: from the `craftax` package (Matthews et al.), pinned to
  1.5.0 in the project dependencies.
- **VIM and LatentGCE**: the original source files are listed in each
  `reference_manifest.json`, with their licences beside them.
- **`_vendor/rendering`**: MIT. The bundled fonts keep their own licence files.

