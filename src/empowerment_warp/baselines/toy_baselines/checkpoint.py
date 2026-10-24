"""Atomic complete-state checkpoints for Latent-GCE."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flax import serialization


CHECKPOINT_VERSION = 2


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def save_checkpoint(
    output_dir: str | Path,
    name: str,
    payload: dict[str, Any],
    *,
    controller_steps: int,
    probe_steps: int,
) -> dict[str, Path]:
    """Write a resumable PyTree plus a small inspectable metadata sidecar."""

    checkpoint_dir = Path(output_dir) / 'checkpoints'
    checkpoint_path = checkpoint_dir / f'{name}.msgpack'
    metadata_path = checkpoint_dir / f'{name}.json'
    _atomic_write(checkpoint_path, serialization.to_bytes(payload))
    metadata = {
        'checkpoint_version': CHECKPOINT_VERSION,
        'controller_env_steps': int(controller_steps),
        'probe_steps': int(probe_steps),
        'env_steps': int(controller_steps + probe_steps),
        'payload_keys': sorted(payload),
        'checkpoint_path': str(checkpoint_path),
    }
    _atomic_write(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + '\n').encode('utf-8'),
    )
    return {'checkpoint': checkpoint_path, 'metadata': metadata_path}


def restore_checkpoint(path: str | Path, template: dict[str, Any]) -> dict[str, Any]:
    """Restore into an initialized template so optimizer/static structure is exact."""

    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / 'checkpoints' / 'latest.msgpack'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Latent-GCE checkpoint not found: {checkpoint_path}')
    return serialization.from_bytes(template, checkpoint_path.read_bytes())
