#!/usr/bin/env bash
set -euo pipefail
# Slurm relocates this file. All inputs are absolute arguments from submit.sh.
source_root="$1"
config="$2"
run_root="$3"
shift 3
: "${PYTHON:?Submission must export PYTHON}"
: "${SLURM_ARRAY_TASK_ID:?Submit as a seed array}"
[[ "${source_root}" = /* && "${config}" = /* && "${run_root}" = /* ]] || exit 2
[[ -f "${source_root}/src/empowerment_warp/cli.py" ]] || exit 2
[[ -f "${config}" ]] || exit 2
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
export PYTHONPATH="${source_root}/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export MUJOCO_GL="${MUJOCO_GL:-egl}"
"${PYTHON}" -c 'import jax; assert jax.devices()[0].platform == "gpu", "GPU required"'
exec "${PYTHON}" -m empowerment_warp.cli train --config "${config}" \
    --seed "${SLURM_ARRAY_TASK_ID}" --run-dir "${run_root}/seed${SLURM_ARRAY_TASK_ID}" "$@"
