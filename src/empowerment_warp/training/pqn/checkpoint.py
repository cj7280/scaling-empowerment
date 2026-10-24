"""Potential-PQN parameter and full-state checkpoints, log trimming, and resume configs."""

from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _save_checkpoint(run_dir: Path, step: int, agent, *, skill_state=None,
                     keep_checkpoints: int = 0) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_steps": int(np.asarray(jax.device_get(agent.env_steps))),
        "n_updates": int(np.asarray(jax.device_get(agent.n_updates))),
        "params": jax.device_get(agent.network.params),
        "batch_stats": jax.device_get(agent.batch_stats),
    }
    if skill_state is not None:
        payload["skill_params"] = jax.device_get(skill_state.params)
        payload["skill_target_params"] = jax.device_get(skill_state.target_params)
        payload["skill_step"] = int(skill_state.step)
    with (ckpt_dir / f"params_{step:012d}.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    if keep_checkpoints > 0:
        for stale in sorted(ckpt_dir.glob("params_*.pkl"))[:-keep_checkpoints]:
            stale.unlink()


FULL_STATE_PREFIX = "full_state_"


def _tree_leaves_numpy(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return [np.asarray(jax.device_get(leaf)) for leaf in leaves], treedef


def _tree_restore(template, leaves):
    """Rebuild `template`'s pytree from saved leaves (the template supplies the structure, so
    unpicklable members such as optax transformations never touch the checkpoint)."""
    template_leaves, treedef = jax.tree_util.tree_flatten(template)
    if len(template_leaves) != len(leaves):
        raise ValueError(f"full-state leaf count {len(leaves)} != live structure {len(template_leaves)}")
    restored = []
    for live, saved in zip(template_leaves, leaves):
        live_arr = np.asarray(jax.device_get(live))
        if live_arr.shape != np.asarray(saved).shape:
            raise ValueError(f"full-state leaf shape {np.asarray(saved).shape} != live {live_arr.shape}")
        restored.append(jnp.asarray(saved, dtype=live_arr.dtype) if hasattr(live, "dtype") else saved)
    return jax.tree_util.tree_unflatten(treedef, restored)


def _save_full_state(run_dir: Path, step: int, *, agent, skill_state, archive_state, carry, rng,
                     global_step: int, segment: int, expl_acc: dict, last_media_step, last_eval_step,
                     keep: bool = False, include_archive: bool = True) -> Path:
    """Everything the training loop needs to continue EXACTLY: agent (params, optimizer, batch
    stats, RNG, step counters), encoder SkillState (params, EMA copy, optimizer, dual, input
    stats), the encoder archive ring (the 'buffer': ~3 GB at 10M rows for Craftax classic),
    the collector carry (env state, recurrent state, lane skills/anchors) and the loop RNG.
    Written atomically. Rotation: intermediate full states are replaced by the newest one; a full
    state written with keep=True (the END of a budget) is renamed full_state_keep_*.pkl and never
    rotated, so a finished run can always be continued EXACTLY from its final step, however many
    continuation attempts come later."""
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 1, "env_steps": int(global_step), "segment": int(segment),
        "agent_leaves": _tree_leaves_numpy(agent)[0],
        "skill_leaves": _tree_leaves_numpy(skill_state)[0],
        # The encoder archive ring is ~3 GB at 10M Craftax rows (95% of the file). Stored by default
        # because an exact resume needs it; include_archive=False gives ~100 MB states whose
        # resume restarts the archive empty (it refills in ~10M env steps).
        **({"archive_leaves": _tree_leaves_numpy(archive_state)[0]} if include_archive else {}),
        "carry_leaves": _tree_leaves_numpy(carry)[0],
        "rng": np.asarray(jax.device_get(rng)),
        "expl_acc": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in expl_acc.items()},
        "last_media_step": last_media_step, "last_eval_step": last_eval_step,
    }
    target = ckpt_dir / (f"{FULL_STATE_PREFIX}keep_{step:012d}.pkl" if keep else f"{FULL_STATE_PREFIX}{step:012d}.pkl")
    tmp = target.with_suffix(".pkl.tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(target)
    for stale in sorted(ckpt_dir.glob(f"{FULL_STATE_PREFIX}*.pkl")):
        if stale != target and "keep_" not in stale.name:
            stale.unlink()
    return target


def _full_state_step(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _latest_full_state(run_dir: Path) -> Path | None:
    """The full state with the LARGEST step (kept budget-end states and rotating ones alike)."""
    found = list((run_dir / "checkpoints").glob(f"{FULL_STATE_PREFIX}*.pkl")) if (run_dir / "checkpoints").exists() else []
    return max(found, key=_full_state_step) if found else None


def _latest_params_checkpoint(run_dir: Path) -> Path | None:
    found = sorted((run_dir / "checkpoints").glob("params_*.pkl")) if (run_dir / "checkpoints").exists() else []
    return found[-1] if found else None


def _trim_logs_past(run_dir: Path, step: int) -> dict:
    """Drop metrics.jsonl / eval_reference.jsonl rows with env_steps > `step`: they come from an
    attempt that was cancelled after this checkpoint was written and would otherwise sit in the
    history as a duplicate band. Originals are kept as *.pre_resume_<step>."""
    trimmed = {}
    for name in ("metrics.jsonl", "eval_reference.jsonl"):
        path = run_dir / name
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        keep = []
        for line in lines:
            if not line.strip():
                continue
            try:
                if float(json.loads(line).get("env_steps", -1)) > step:
                    continue
            except (ValueError, json.JSONDecodeError):
                pass
            keep.append(line)
        if len(keep) != len(lines):
            path.with_name(f"{name}.pre_resume_{step}").write_text("\n".join(lines) + "\n")
            path.write_text("\n".join(keep) + ("\n" if keep else ""))
        trimmed[name] = len(lines) - len(keep)
    return trimmed


def resume_config_from_run(run_dir: Path, config_cls, *, total_env_steps: int | None = None,
                           schedule_env_steps: int | None = None, overrides: dict | None = None):
    """Build the resume config FROM THE FROZEN config.json (the source of truth for what the run is),
    applying only the budget knobs. A larger budget re-pins pqn_schedule_env_steps to the new
    budget unless one is given explicitly, because the linear LR decay holds ~0 past its horizon
    and a continuation would otherwise not train. `empowerment-warp resume-pqn` uses this, so a
    continuation needs neither the recipe script nor its launch environment."""
    frozen = json.loads((run_dir / "config.json").read_text())
    known = set(getattr(config_cls, "__dataclass_fields__", {}))
    fields = (config_cls.from_dict(frozen).to_dict() if hasattr(config_cls, "from_dict")
              else {k: v for k, v in frozen.items() if k in known})
    fields["run_dir"] = str(run_dir)
    if total_env_steps is not None:
        if total_env_steps < int(frozen["total_env_steps"]):
            raise ValueError("resume budget must not be smaller than the frozen budget")
        fields["total_env_steps"] = int(total_env_steps)
        if schedule_env_steps is None and total_env_steps > int(frozen["total_env_steps"]):
            schedule_env_steps = int(total_env_steps)
    if schedule_env_steps is not None:
        fields["pqn_schedule_env_steps"] = int(schedule_env_steps)
    for k, v in (overrides or {}).items():
        if k not in known:
            raise ValueError(f"unknown config field {k!r}")
        fields[k] = v
    return config_cls(**fields)


def _check_resume_config(run_dir: Path, config) -> dict:
    """The frozen config may differ from the requested one only by a LARGER total_env_steps
    (budget extension) -- the same rule as SAC's resume-run."""
    frozen = json.loads((run_dir / "config.json").read_text())
    if hasattr(type(config), "from_dict"):
        frozen = type(config).from_dict(frozen).to_dict()
    requested = config.to_dict()
    # pqn_schedule_env_steps is tolerated too: extending a finished run's budget past its schedule horizon
    # would otherwise continue at LR ~0 (linear decay holds its end value), so a budget extension re-pins
    # the schedule to the new budget (LR resumes at lr * (1 - done/new) and decays to 0 at the new end).
    ignore = {"total_env_steps", "pqn_schedule_env_steps", "run_dir", "resume", "max_chunks", "max_train_seconds",
              "full_state_every_segments", "full_state_include_archive", "checkpoint_every_segments",
              "keep_checkpoints", "log_every_segments"}
    defaults = dataclasses.asdict(type(config)(run_dir=config.run_dir, environment=config.environment)) \
        if hasattr(type(config), "__dataclass_fields__") else {}
    diff = {}
    for k in set(frozen) | set(requested):
        if k in ignore or frozen.get(k) == requested.get(k):
            continue
        # A knob added AFTER the run was frozen is fine as long as the resume asks for its default
        # (the frozen run behaved as the default); anything else is a real config change.
        if k not in frozen and k in defaults and requested.get(k) == defaults[k]:
            continue
        diff[k] = (frozen.get(k), requested.get(k))
    if diff:
        raise ValueError(f"resume config differs from the frozen run config: {diff}")
    if int(requested["total_env_steps"]) < int(frozen["total_env_steps"]):
        raise ValueError("resume total_env_steps must be >= the frozen budget")
    return frozen
