"""The saved toy runner's local-first W&B failure handling."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import wandb


_WANDB_STATE = {
    'enabled': False,
    'degraded': False,
    'init_error': None,
    'run_id': None,
    'run_name': None,
    'project': None,
}


def record_wandb_failure(action, save_dir=None, exc=None, **payload):
    _WANDB_STATE['degraded'] = True
    record = {'time': datetime.now().isoformat(), 'action': action}
    if exc is not None:
        record.update(error_type=type(exc).__name__, error=str(exc))
    record.update(payload)
    if action == 'init' and exc is not None:
        _WANDB_STATE['init_error'] = record
    if save_dir:
        path = Path(save_dir) / 'wandb_pending_uploads.jsonl'
        with path.open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')
    return record


def write_wandb_status(save_dir):
    if save_dir:
        path = Path(save_dir) / 'wandb_status.json'
        path.write_text(json.dumps(_WANDB_STATE, indent=2, sort_keys=True) + '\n')
        return path
    return None


def safe_wandb_log(metrics, step=None, save_dir=None):
    if getattr(wandb, 'run', None) is None:
        return False
    try:
        wandb.log(metrics, step=step)
        return True
    except Exception as exc:
        record_wandb_failure('log', save_dir=save_dir, exc=exc, step=step,
                             metric_keys=sorted(str(key) for key in metrics))
        write_wandb_status(save_dir)
        return False


def safe_wandb_finish(save_dir=None):
    if getattr(wandb, 'run', None) is None:
        return False
    try:
        wandb.finish()
        return True
    except Exception as exc:
        record_wandb_failure('finish', save_dir=save_dir, exc=exc)
        write_wandb_status(save_dir)
        return False
