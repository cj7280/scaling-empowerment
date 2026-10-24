#!/usr/bin/env bash
# Source from a checked-out command, never relative to a Slurm spool file.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# A per-command tracking mode overrides the local default.
requested_wandb_mode="${WANDB_MODE-}"
if [[ -f "${project_root}/.env" ]]; then
    set -a
    source "${project_root}/.env"
    set +a
fi
if [[ -n "${requested_wandb_mode}" ]]; then
    export WANDB_MODE="${requested_wandb_mode}"
fi
unset requested_wandb_mode
: "${PYTHON:?Set PYTHON in .env}"
export PYTHONPATH="${project_root}/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
