"""SAC run loading, evaluation skill schedules and traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.checkpointing import CheckpointStore, ModelCheckpoint
from empowerment_warp.core.rollout import potential_switch_steps
from empowerment_warp.training.sac.config import ExperimentConfig
from empowerment_warp.training.sac.runner import (
    make_experiment,
    require_runtime,
)


def evaluation_switch_steps(config: ExperimentConfig, count: int, seed_offset: int) -> jax.Array:
    """Fresh episodes use an independent schedule stream, stable across checkpoints."""
    if (config.skill_start_mode == "marginal_branch" and not config.live_potential_only
            and not config.branch_live_two_phase):
        # potential_steps sets the branch horizon, not the live policy's prefix.
        return jnp.zeros(count, dtype=jnp.int32)
    return potential_switch_steps(
        jnp.arange(count, dtype=jnp.int32), config.potential_steps,
        config.random_potential_switch_max_steps, config.seed + seed_offset + 1_000_000,
    )


def potential_schedule_metadata(
    config: ExperimentConfig, boundaries: Any, seed_offset: int, *,
    shared: bool = False, prefix_steps_override: int | None = None,
) -> dict[str, Any]:
    """Record both the recipe and realized timing; explicit scalar overrides are diagnostic."""
    return {
        "schedule": "fixed_override" if prefix_steps_override is not None else (
            "uniform_integer" if config.random_potential_switch_max_steps > 0 else "fixed"
        ),
        "nominal_potential_steps": int(config.potential_steps),
        "random_potential_switch_max_steps": int(config.random_potential_switch_max_steps),
        "switch_seed": int(config.seed + seed_offset + 1_000_000),
        "shared_boundary": bool(shared),
        "prefix_steps_override": prefix_steps_override,
        "realized_switch_steps": np.asarray(jax.device_get(boundaries)).tolist(),
    }


def eval_skill_library(config: ExperimentConfig) -> jax.Array:
    """The skill set an evaluator sweeps.

    Continuous skills have no trained library (`SkillConfig` forces `num_skills=0`), so
    evaluating them needs a fixed probe set or the numbers are not comparable across
    checkpoints. Build the same seeded probe library the PQN evaluators use: `num_skills` unit
    vectors drawn once from `seed + 12345`, so every evaluation of a given seed -- and every
    checkpoint of that run -- sweeps identical directions.
    """

    count = max(int(config.num_skills), 1)
    rng = np.random.RandomState(int(config.seed) + 12345)
    rows = rng.normal(size=(count, int(config.skill_dim)))
    rows = rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-8)
    return jnp.asarray(rows, dtype=jnp.float32)


def load_run(run_dir: str | Path, *, checkpoint: str | Path | None = None) -> tuple[ExperimentConfig, Any, Any]:
    path = Path(run_dir).expanduser().resolve()
    config = ExperimentConfig.from_dict(json.loads((path / "config.json").read_text()))
    require_runtime(config)
    experiment = make_experiment(config)
    store = CheckpointStore(path / "checkpoints", keep=config.keep_checkpoints)
    manifest = store.latest_manifest()
    if manifest is not None and manifest.get("payload_kind") == "model":
        _, sac_key, skill_key, _ = jax.random.split(jax.random.PRNGKey(config.seed), 4)
        template = ModelCheckpoint(
            sac=experiment.sac.initialize(sac_key),
            skill=experiment.skill.initialize(skill_key),
            env_steps=jnp.asarray(0, dtype=jnp.int32),
        )
    else:
        template = experiment.trainer.initialize(jax.random.PRNGKey(config.seed))
    state = store.restore(template)
    if checkpoint is not None:
        selected = Path(checkpoint).expanduser().resolve()
        if not selected.is_dir() or not selected.name.startswith('step_'):
            raise ValueError('checkpoint must name a retained step directory')
        state = CheckpointStore(selected.parent).restore(
            state, int(selected.name.removeprefix('step_')))
    return config, experiment, state


def record_trace(
    run_dir: str | Path,
    output: str | Path,
    *,
    steps: int,
    lane: int = 0,
) -> Path:
    """Sample policy actions on device, then copy one lane once."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    config, experiment, training_state = load_run(run_dir)
    if not 0 <= lane < config.num_envs:
        raise ValueError(f"lane must be in [0, {config.num_envs})")
    random_schedule = config.random_potential_switch_max_steps > 0
    if hasattr(training_state, "collector") and not random_schedule:
        initial = training_state.collector.env_state
        anchors = training_state.collector.anchor_observations
        skills = training_state.collector.skills
    else:
        reset_key = jax.random.PRNGKey(config.seed + 60_000)
        initial = experiment.env.reset(jax.random.split(reset_key, config.num_envs))
        anchors = initial.obs
        # Continuous runs have no library; zeros would mean z=0 (the potential policy), which is
        # not a skill sweep, so use the same seeded probe directions the other evaluators use.
        skill_library = eval_skill_library(config)
        skill_indices = jnp.arange(config.num_envs) % skill_library.shape[0]
        skills = skill_library[skill_indices]
    switch_steps = evaluation_switch_steps(config, config.num_envs, 60_000)

    def scan_step(carry: tuple[Any, jax.Array, jax.Array], _: None):
        env_state, current_anchors, key = carry
        key, action_root = jax.random.split(key)
        action_keys = jax.random.split(action_root, config.num_envs)
        prefix = env_state.step_count < switch_steps if random_schedule else jnp.zeros(config.num_envs, dtype=jnp.bool_)
        effective_skills = jnp.where(prefix[:, None], jnp.zeros_like(skills), skills)
        actions = experiment.sac.sample_actions(
            training_state.sac,
            env_state.obs,
            current_anchors,
            effective_skills,
            action_keys,
            deterministic=False,
        )
        env_state = experiment.env.step(env_state, actions)
        current_anchors = jnp.where(prefix[:, None], env_state.obs, current_anchors)
        row = {
            "observations": env_state.obs[lane],
            "actions": actions[lane],
            "rewards": env_state.reward[lane],
            "terminated": env_state.terminated[lane],
            "truncated": env_state.truncated[lane],
        }
        if hasattr(env_state, "data"):
            row.update(
                qpos=env_state.data.qpos[lane],
                qvel=env_state.data.qvel[lane],
                ctrl=env_state.data.ctrl[lane],
                mocap_pos=env_state.data.mocap_pos[lane],
                mocap_quat=env_state.data.mocap_quat[lane],
            )
        return (env_state, current_anchors, key), row

    (_, _, _), trace = jax.jit(
        lambda state, key: jax.lax.scan(scan_step, (state, anchors, key), None, length=steps)
    )(initial, jax.random.PRNGKey(config.seed + 1))
    arrays = jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value)), trace)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        metadata=np.asarray(
            json.dumps(
                {
                    "environment": config.environment,
                    "algorithm": config.algorithm,
                    "control_dt": _control_dt(experiment.env),
                    "lane": lane,
                    "steps": steps,
                    "fresh_reset_population": random_schedule or not hasattr(training_state, "collector"),
                    "potential_schedule": potential_schedule_metadata(config, [switch_steps[lane]], 60_000)
                    if random_schedule else None,
                    "source_run": str(Path(run_dir).expanduser().resolve()),
                },
                sort_keys=True,
            )
        ),
    )
    return output_path


def _control_dt(env: Any) -> float | None:
    if hasattr(env, "config"):
        return float(env.config.ctrl_dt)
    return None
