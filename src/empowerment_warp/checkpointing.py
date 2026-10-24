"""Orbax checkpoints and explicit pickle-checkpoint extraction."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, NamedTuple

import jax
import numpy as np
import orbax.checkpoint as ocp


class ModelCheckpoint(NamedTuple):
    """Learner state required for evaluation and warm-start resume."""

    sac: Any
    skill: Any
    env_steps: Any


def model_checkpoint(state: Any) -> ModelCheckpoint:
    return ModelCheckpoint(sac=state.sac, skill=state.skill, env_steps=state.env_steps)


def _is_zero_array(value: Any) -> bool:
    return isinstance(value, (jax.Array, np.ndarray)) and value.size == 0


def _pack_zero_arrays(tree: Any) -> Any:
    """Orbax 0.11 rejects empty arrays; a one-cell sentinel preserves structure."""

    return jax.tree_util.tree_map(
        lambda value: jax.numpy.zeros((1,), dtype=value.dtype) if _is_zero_array(value) else value,
        tree,
    )


def _restore_zero_arrays(restored: Any, target: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda value, template: template if _is_zero_array(template) else value,
        restored,
        target,
    )


class CheckpointStore:
    """Versioned Orbax state directories with a small JSON manifest."""

    format_version = 1

    def __init__(self, root: str | Path, *, keep: int = 0):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep = keep

    def _path(self, step: int) -> Path:
        return self.root / f"step_{step:012d}"

    def save(
        self,
        step: int,
        state: Any,
        config: dict[str, Any],
        *,
        payload_kind: str = "full",
    ) -> Path:
        if payload_kind not in {"full", "model"}:
            raise ValueError("payload_kind must be 'full' or 'model'")
        path = self._path(step)
        if self.latest_step() != step:
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(
                path,
                _pack_zero_arrays(state),
                force=path.exists(),
                custom_metadata={
                    "format_version": self.format_version,
                    "payload_kind": payload_kind,
                    "config": config,
                },
            )
            checkpointer.close()
        manifest = {
            "format_version": self.format_version,
            "step": int(step),
            "checkpoint": path.name,
            "payload_kind": payload_kind,
            "config": config,
        }
        (self.root / "latest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if self.keep:
            checkpoints = sorted(self.root.glob("step_[0-9]" + "[0-9]" * 11))
            for stale in checkpoints[:-self.keep]:
                if stale.parent != self.root or not stale.name.startswith("step_"):
                    raise RuntimeError(f"refusing to prune unexpected checkpoint path: {stale}")
                shutil.rmtree(stale)
        return path

    def latest_step(self) -> int | None:
        manifest_path = self.root / "latest.json"
        if not manifest_path.exists():
            return None
        return int(json.loads(manifest_path.read_text())["step"])

    def latest_manifest(self) -> dict[str, Any] | None:
        manifest_path = self.root / "latest.json"
        return json.loads(manifest_path.read_text()) if manifest_path.exists() else None

    def restore(self, target: Any, step: int | None = None) -> Any:
        resolved_step = self.latest_step() if step is None else step
        if resolved_step is None:
            raise FileNotFoundError(f"no checkpoint under {self.root}")
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(
            self._path(resolved_step),
            target=_pack_zero_arrays(target),
        )
        checkpointer.close()
        return _restore_zero_arrays(restored, target)
