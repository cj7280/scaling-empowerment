"""Recurrent Potential-PQN training, evaluation, and checkpoint restoration."""

from __future__ import annotations

import dataclasses
import json
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.agents.see_pqn import (
    make_encoder_update_burst,
    make_frozen_reward_fn,
)
from empowerment_warp.agents.skill_classifier import SkillLearner
from empowerment_warp.tracking import WandbMirror
from empowerment_warp.training.pqn.collector import (
    make_pqn_encoder_branch_sweep,
    initialize_potential_pqn_collector,
    make_potential_pqn_chunk,
)
from empowerment_warp.training.pqn.archive import PQNEncoderArchive
from empowerment_warp.training.pqn.checkpoint import (
    FULL_STATE_PREFIX,
    _check_resume_config,
    _latest_full_state,
    _latest_params_checkpoint,
    _save_checkpoint,
    _save_full_state,
    _tree_restore,
    _trim_logs_past,
)
from empowerment_warp.training.pqn.config import PotentialPQNRunConfig
from empowerment_warp.training.pqn.env import PQNEnv
from empowerment_warp.training.pqn.logging import (
    _craftax_achievement_keys,
    _reference_keys,
)


def _require_runtime(config: PotentialPQNRunConfig) -> None:
    platform = jax.devices()[0].platform
    if platform != "gpu" and not config.allow_cpu:
        raise RuntimeError(
            f"PQN training requires a GPU, but JAX selected {platform!r}; "
            "pass --allow-cpu only for tests"
        )


def _make_env(config: PotentialPQNRunConfig):
    if ":" not in config.environment:
        raise ValueError(
            f"PQN supports Craftax-Classic only; got {config.environment!r}"
        )
    return PQNEnv(
        config.environment, num_envs=config.num_envs,
        episode_length=config.episode_length, reset_ratio=config.optimistic_reset_ratio,
    )


def _jsonable(value):
    try:
        if hasattr(value, "item") and np.ndim(value) == 0:
            return value.item()
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def run_potential_pqn_training(config: PotentialPQNRunConfig, *, resume: bool = False) -> Path:
    _require_runtime(config)
    # Craftax trains on commanded skills only (relabel_unit is recorded as "segment"). The
    # dataclass defaults differ, so the pin lives here rather than in __post_init__.
    if config.achieved_skill_relabeling or config.relabel_unit != "segment":
        raise ValueError("Craftax hindsight relabeling is not supported; set achieved_skill_relabeling=False "
                         "and relabel_unit='segment'")
    if config.num_envs % config.pqn_num_minibatches != 0:
        raise ValueError("pqn_num_minibatches must divide num_envs exactly.")
    run_dir = Path(config.run_dir)
    if resume:
        # Exact continuation from checkpoints/full_state_*.pkl (agent + optimizer + encoder +
        # archive + collector + RNG); if a run predates full-state checkpoints, fall back to the
        # newest params_*.pkl (params, EMA copy and step counters only: optimizer, archive and
        # collector restart fresh) and say so in resume_log.json.
        if not (run_dir / "config.json").exists():
            raise FileNotFoundError(f"cannot resume: {run_dir / 'config.json'} is missing")
        frozen = _check_resume_config(run_dir, config)
        resume_from = _latest_full_state(run_dir) or _latest_params_checkpoint(run_dir)
        if resume_from is None:
            raise FileNotFoundError(f"cannot resume: no checkpoints under {run_dir / 'checkpoints'}")
        resume_exact = resume_from.name.startswith(FULL_STATE_PREFIX)
    elif run_dir.exists() and any(p.name not in {"recipe.json", "slurm_metadata.txt", "source_snapshot.json"}
                                  for p in run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}; pass --resume to continue")
    run_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env(config)
    obs_dim = int(env.observation_size)
    skill = SkillLearner(config.skill_config(obs_dim), ema_tau_one_shortcut=True)
    skill_state = skill.initialize(jax.random.PRNGKey(config.seed + 7))
    # Dense archive: capacity and readiness both count collected transitions.
    branch_encoder = config.encoder_data_source == "branches"
    branch_horizon = config.branch_horizon_steps
    encoder_archive = PQNEncoderArchive(
        num_envs=config.num_envs, capacity=config.representation_archive_size,
        episode_length=branch_horizon if branch_encoder else config.episode_length,
        future_gamma=config.future_gamma,
        max_future_steps=config.max_future_steps,
        fixed_anchors=branch_encoder,
    )
    archive_state = encoder_archive.initialize(
        config.num_envs, skill.config.observation_dim if branch_encoder else obs_dim,
        config.skill_dim)
    archive_insert = jax.jit(encoder_archive.insert)
    encoder_burst = (
        make_encoder_update_burst(skill, encoder_archive, batch_size=config.encoder_batch_size,
                                  update_steps=config.encoder_updates_per_chunk,
                                  preprocessed=branch_encoder)
        if config.encoder_updates_per_chunk > 0 else None
    )
    agent = config.agent_class().create(
        config.seed,
        jnp.zeros((1, obs_dim), jnp.float32),
        jnp.zeros((1, env.action_size), jnp.float32),
        config.agent_config(),
    )
    expected = (int(agent.config["pqn_expected_num_envs"]), int(agent.config["pqn_rollout_steps"]))
    if (config.num_envs, config.rollout_steps) != expected:
        raise ValueError(f"PQN width-lock: live {(config.num_envs, config.rollout_steps)} != {expected}.")

    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
    (run_dir / "reference_manifest.json").write_text(
        json.dumps(
            {
                "effective_recipe": config.effective_recipe(),
                "agent_config": {k: _jsonable(v) for k, v in agent.config.items()},
                "skill_config": {k: _jsonable(v) for k, v in dataclasses.asdict(skill.config).items()},
            },
            indent=2, sort_keys=True,
        )
        + "\n"
    )

    def sampler(key, count):
        return skill.sample_skills(key, count)

    # Standard craftax achievement keys (empty for non-craftax) -> logged from the
    # training rollout (exploration/*).
    achievement_keys = tuple(_craftax_achievement_keys(env, config.num_envs))
    ach_names = [a.split("/", 1)[1] for a in achievement_keys]

    collect = make_potential_pqn_chunk(
        env, agent, chunk_steps=config.rollout_steps, num_envs=config.num_envs,
        skill_dim=config.skill_dim, potential_steps=config.potential_steps, skill_sampler=sampler,
        potential_lane_fraction=config.potential_lane_fraction,
        achievement_keys=achievement_keys,
    )
    branch_sweep = (
        make_pqn_encoder_branch_sweep(
            env, agent, num_envs=config.num_envs, skill_dim=config.skill_dim,
            horizon=branch_horizon, skill_sampler=sampler,
            observation_fn=skill.encoder_observations,
        ) if branch_encoder else None
    )
    frozen_reward = make_frozen_reward_fn(skill, config.discount, zero_eps=1e-6)
    from empowerment_warp.training.pqn.branch_eval import (
        NativeHoldoutDriver, evaluate_agent, evaluation_skill_library,
    )
    eval_driver = None
    last_media_step = None
    eval_skill_library = evaluation_skill_library(config)
    last_eval_step, last_eval_metrics = None, {}

    def _run_eval(step, *, force_media=False):
        nonlocal eval_driver, last_media_step, last_eval_step, last_eval_metrics
        if last_eval_step == step and (not force_media or last_media_step == step):
            return last_eval_metrics
        if eval_driver is None:
            eval_driver = NativeHoldoutDriver(config, agent, eval_skill_library)
        media = bool(config.eval_video_episodes) and (
            force_media or last_media_step is None or config.eval_media_interval == 0
            or step - last_media_step >= config.eval_media_interval
        )
        print(f"[eval step {step}] shared-prefix contexts={config.eval_contexts} "
              f"skills/context={config.eval_skills_per_context} media={media}", flush=True)
        row = evaluate_agent(config, agent, eval_skill_library, env_steps=step,
                             output_dir=run_dir, driver=eval_driver, media=media)
        with (run_dir / "eval_reference.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        if media:
            last_media_step = step
            if wandb_mirror.run is not None:
                import wandb
                figure_dir = run_dir / "figures" / f"step_{step}"
                for name in ("craftax_branching_stills", "craftax_branching_stats"):
                    path = figure_dir / f"{name}.png"
                    if path.exists():
                        wandb_mirror.log({"env_steps": step, f"media/{name}": wandb.Image(str(path))})
                video = figure_dir / "eval_branching_skill_grid.mp4"
                if video.exists():
                    wandb_mirror.log_video({"env_steps": step}, key="media/craftax_branching_skill_grid",
                                           path=video, fps=row["figures"]["fps"])
        last_eval_step = step
        last_eval_metrics = {k: v for k, v in row.items() if k.startswith("evaluation/")}
        return last_eval_metrics

    rng = jax.random.PRNGKey(config.seed)
    rng, collect_key, eval_key = jax.random.split(rng, 3)
    carry = initialize_potential_pqn_collector(
        env, agent, collect_key, num_envs=config.num_envs, skill_dim=config.skill_dim,
        potential_steps=config.potential_steps, skill_sampler=sampler,
        potential_lane_fraction=config.potential_lane_fraction,
    )

    env_steps_per_chunk = config.num_envs * config.rollout_steps
    metrics_path = run_dir / "metrics.jsonl"
    resume_global_step, resume_segment, resume_note = 0, 0, None
    if resume:
        with resume_from.open("rb") as handle:
            payload = pickle.load(handle)
        if resume_exact:
            agent = _tree_restore(agent, payload["agent_leaves"])
            skill_state = _tree_restore(skill_state, payload["skill_leaves"])
            if "archive_leaves" in payload:
                archive_state = _tree_restore(archive_state, payload["archive_leaves"])
                archive_note = "archive restored"
            else:
                archive_note = "encoder archive restarted empty (not stored; refills in ~10M steps)"
            carry = _tree_restore(carry, payload["carry_leaves"])
            rng = jnp.asarray(payload["rng"])
            resume_global_step, resume_segment = int(payload["env_steps"]), int(payload["segment"])
            resume_note = f"exact resume from {resume_from.name} (agent, optimizer, encoder, collector, RNG; {archive_note})"
        else:
            agent = agent.replace(
                network=agent.network.replace(params=payload["params"]),
                batch_stats=payload["batch_stats"],
                env_steps=jnp.asarray(payload["env_steps"], dtype=jnp.asarray(agent.env_steps).dtype),
                n_updates=jnp.asarray(payload["n_updates"], dtype=jnp.asarray(agent.n_updates).dtype),
            )
            if "skill_params" in payload:
                skill_state = skill_state.replace(
                    params=payload["skill_params"], target_params=payload["skill_target_params"],
                    step=jnp.asarray(payload["skill_step"], dtype=jnp.int32),
                )
            resume_global_step = int(payload["env_steps"])
            resume_segment = resume_global_step // env_steps_per_chunk
            resume_note = (f"APPROXIMATE resume from {resume_from.name}: params, EMA copy and step counters "
                           f"restored; optimizer state, encoder archive and collector restarted fresh")
        trimmed = _trim_logs_past(run_dir, resume_global_step)
        (run_dir / "resume_log.json").write_text(json.dumps({
            "resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "from": str(resume_from), "exact": resume_exact,
            "env_steps": resume_global_step, "segment": resume_segment,
            "frozen_total_env_steps": frozen["total_env_steps"], "new_total_env_steps": config.total_env_steps,
            "schedule_env_steps": config.pqn_schedule_env_steps,
            "trimmed_log_rows_past_checkpoint": trimmed,
            "note": resume_note}, indent=2) + "\n")
        (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
        print(f"[resume] {resume_note}; env_steps={resume_global_step} -> budget {config.total_env_steps}", flush=True)
    wandb_mirror = WandbMirror.start(run_dir, config.to_dict(), resume=resume)
    print(
        f"[startup] {config.rl_algorithm}: env={config.environment} num_envs={config.num_envs} "
        f"rollout={config.rollout_steps} hidden={config.pqn_hidden_size} skill_dim={config.skill_dim} "
        f"K={config.num_skills} potential_steps={config.potential_steps} ref={config.potential_reward_reference} "
        f"encoder_objective={config.encoder_objective} "
        f"total_env_steps={config.total_env_steps}",
        flush=True,
    )

    global_step = resume_global_step
    segment = resume_segment
    window: dict[str, list[float]] = {}
    # Standard craftax training-rollout accumulators over the logging window
    # (completed episodes only): return, length, per-achievement unlocked, count.
    expl_acc = {"n": 0.0, "ret": 0.0, "len": 0.0, "ach": np.zeros(len(achievement_keys), np.float64)}
    if resume and resume_exact:
        saved_acc = payload.get("expl_acc") or {}
        expl_acc.update({k: (np.asarray(v, np.float64) if k == "ach" else float(v)) for k, v in saved_acc.items()})
        last_media_step, last_eval_step = payload.get("last_media_step"), payload.get("last_eval_step")
    training_started = time.perf_counter()
    stop_reason = "total_env_steps"
    try:
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            if config.eval_at_start and not resume:
                if config.checkpoint_every_segments:
                    _save_checkpoint(run_dir, 0, agent, skill_state=skill_state,
                                     keep_checkpoints=config.keep_checkpoints)
                initial_eval = {"env_steps": 0, **_run_eval(0)}
                metrics_file.write(json.dumps(initial_eval, sort_keys=True) + "\n")
                metrics_file.flush()
                wandb_mirror.log(initial_eval)
            while global_step < config.total_env_steps:
                initial_state = carry.recurrent_state
                epsilon = agent.exploration_epsilon()
                start = time.perf_counter()
                rng, reward_key, enc_key = jax.random.split(rng, 3)
                carry, batch, chunk_metrics = collect(
                    {"params": agent.network.params, "batch_stats": agent.batch_stats}, carry, epsilon
                )
                # Accumulate the standard craftax training-rollout metrics over completed
                # episodes in this chunk (reduced into exploration/* at log time).
                cm = {k: np.asarray(jax.device_get(v)) for k, v in chunk_metrics.items()}
                done_mask = cm["completed"] > 0.5
                expl_acc["n"] += float(done_mask.sum())
                expl_acc["ret"] += float(cm["completed_return"].sum())
                expl_acc["len"] += float(cm["completed_length"].sum())
                if achievement_keys:
                    # Collector flattens metrics to [T*B, ...]: achievements is
                    # [T*B, K] (0/100 unlocked flags), done_mask is [T*B]. Per
                    # achievement, count completed episodes that unlocked it.
                    unlocked = (cm["achievements"] > 0.0).astype(np.float64)
                    expl_acc["ach"] += (unlocked * done_mask[:, None]).sum(axis=0)
                # The frozen reward and the two-Q update read the commanded skills and anchors
                # (no Craftax recipe relabels them).
                update_batch = dict(batch)
                update_batch["pqn_command_boundary"] = (
                    jnp.linalg.norm(batch["next_skills"] - batch["skills"], axis=-1) > 1e-6
                ).astype(jnp.float32)
                # Freeze the empowerment reward BEFORE the updates.
                pqn_rewards, reward_info = frozen_reward(
                    agent, skill_state, update_batch["observations"], update_batch["next_observations"],
                    update_batch["anchor_observations"], update_batch["skills"], update_batch["timesteps"],
                    reward_key, update_batch, initial_state,
                    partition_weight=(config.reward_partition_weight(global_step + env_steps_per_chunk)
                                      if config.reward_partition_anneal_end_steps else None),
                )
                update_batch["pqn_rewards"] = pqn_rewards
                agent, info = agent.update_rollout(
                    update_batch, num_envs=config.num_envs, chunk_steps=config.rollout_steps,
                    initial_recurrent_state=initial_state,
                )
                # The live Q/reward path above is identical in both data-source modes.
                # A branch sweep is atomic: no in-progress or post-terminal padding
                # becomes eligible for an encoder draw.
                branch_steps = 0
                branch_valid_rows = 0
                if branch_sweep is not None:
                    previous_sweep = global_step // (config.num_envs * config.episode_length)
                    next_sweep = (global_step + env_steps_per_chunk) // (config.num_envs * config.episode_length)
                    if next_sweep > previous_sweep:
                        branch_key = jax.random.fold_in(enc_key, next_sweep)
                        branch_batch, valid_rows = branch_sweep(
                            {"params": agent.network.params, "batch_stats": agent.batch_stats},
                            carry.env_state, agent.exploration_epsilon(), branch_key,
                            jnp.asarray(next_sweep, jnp.int32),
                            carry.recurrent_state, carry.recurrent_last_done,
                        )
                        archive_state = archive_insert(archive_state, branch_batch)
                        branch_steps = branch_horizon * config.num_envs
                        branch_valid_rows = int(jax.device_get(valid_rows))
                else:
                    # Reference ordering: freeze reward -> Q epochs -> insert
                    # original transitions -> encoder updates.
                    archive_state = archive_insert(archive_state, batch)
                skill_metrics = None
                archive_info = {}
                archive_rows = int(jax.device_get(archive_state.size))
                encoder_ready = (
                    int(jax.device_get(archive_state.active_rows)) >= config.encoder_train_start_rows
                    if branch_encoder else archive_rows >= config.encoder_train_start_rows
                )
                encoder_steps = 0
                if encoder_ready and encoder_burst is not None:
                    skill_state, enc_key, skill_metrics, archive_info = encoder_burst(
                        skill_state, archive_state, enc_key
                    )
                    jax.block_until_ready(skill_state.step)
                    encoder_steps = config.encoder_updates_per_chunk
                jax.block_until_ready(agent.env_steps)
                elapsed = time.perf_counter() - start
                global_step += env_steps_per_chunk
                segment += 1

                row = {
                    "representation_archive/size": float(archive_rows),
                    "representation_archive/capacity": float(config.representation_archive_size),
                    "representation_archive/active_rows": float(archive_state.active_rows),
                    "representation_archive/total_inserted_rows": float(archive_state.total_inserted_rows),
                    "representation_archive/ready": float(encoder_ready),
                    "representation_archive/update_steps": float(encoder_steps),
                    "representation_archive/update_ratio": config.representation_archive_update_ratio,
                    "representation_archive/learner_update_steps": float(config.pqn_num_epochs * config.pqn_num_minibatches),
                    "representation_archive/total_encoder_steps": float(skill_state.step),
                    "encoder_branch/simulator_steps": float(branch_steps),
                    "encoder_branch/valid_rows": float(branch_valid_rows),
                    "encoder_branch/cumulative_simulator_steps": float(
                        ((global_step // (config.num_envs * config.episode_length))
                         * branch_horizon * config.num_envs) if branch_encoder else 0),
                    "pqn/done_frac": float(cm["done"].mean()),
                    "pqn/skill_on_frac": float(cm["skill_on"].mean()),
                    # Controller scalars from PQN.
                    **{k: float(v) for k, v in info.items()
                       if k in ("pqn/td_loss", "pqn/q_mean", "pqn/epsilon")},
                    "reward/partition_weight": float(reward_info["reward/partition_weight"]),
                    "reward/pqn_reward_mean": float(reward_info["reward/pqn_reward_mean"]),
                    "reward/empowerment_mean": float(reward_info["reward/empowerment_mean"]),
                    "reward/potential_value_mean": float(reward_info["reward/potential_value_mean"]),
                    "reward/potential_to_skill_ratio": float(reward_info["reward/potential_to_skill_ratio"]),
                    "sps": env_steps_per_chunk / max(elapsed, 1e-9),
                    **({k: float(v) for k, v in archive_info.items()} if archive_info else {}),
                }
                if skill_metrics is not None:
                    row.update({
                        "skill/loss": float(skill_metrics.loss),
                        "skill/accuracy": float(skill_metrics.accuracy),
                        "skill/mi_lower_bound": float(skill_metrics.mi_lower_bound),
                        "skill/alignment": float(skill_metrics.alignment),
                        "skill/log_partition": float(skill_metrics.log_partition),
                    })
                for key, value in row.items():
                    window.setdefault(key, []).append(value)

                reached_time_limit = (
                    config.max_train_seconds is not None
                    and (time.perf_counter() - training_started) >= config.max_train_seconds
                )
                reached_chunk_limit = config.max_chunks > 0 and segment >= resume_segment + config.max_chunks
                if (
                    segment == 1 or segment % config.log_every_segments == 0
                    or (config.eval_every_segments and segment % config.eval_every_segments == 0)
                    or global_step >= config.total_env_steps or reached_time_limit or reached_chunk_limit
                ):
                    logged = {
                        "segment": segment, "env_steps": global_step,
                        "train/wall_seconds": time.perf_counter() - training_started,
                        **{k: float(np.mean(v)) for k, v in window.items()},
                    }
                    # Ring sizes and cumulative counters describe this exact step.
                    for name in ("size", "capacity", "active_rows", "total_inserted_rows",
                                 "total_encoder_steps", "ready"):
                        key = f"representation_archive/{name}"
                        logged[key] = row[key]
                    window = {}
                    # Standard craftax training-rollout metrics over episodes that
                    # completed since the last log (exploration/* achievement counts
                    # and success rates).
                    if expl_acc["n"] > 0:
                        n = expl_acc["n"]
                        rates = expl_acc["ach"] / n
                        logged["exploration/episode.return"] = expl_acc["ret"] / n
                        logged["exploration/episode.length"] = expl_acc["len"] / n
                        if len(rates):
                            # exploration/* keys:
                            # count = mean #unlocked/ep, percent = mean %unlocked/ep,
                            # per-achievement = cross-episode success rate (percent).
                            logged["exploration/achievement_count"] = float(rates.sum())
                            logged["exploration/achievement_percent"] = float(100.0 * rates.mean())
                            for name, r in zip(ach_names, rates):
                                logged[f"exploration/achievement/{name}"] = float(100.0 * r)
                    expl_acc = {"n": 0.0, "ret": 0.0, "len": 0.0, "ach": np.zeros(len(achievement_keys), np.float64)}
                    if config.eval_every_segments and segment % config.eval_every_segments == 0:
                        logged.update(_run_eval(global_step))
                    logged = _reference_keys(logged)
                    metrics_file.write(json.dumps(logged, sort_keys=True) + "\n")
                    metrics_file.flush()
                    print(json.dumps(logged, sort_keys=True), flush=True)
                    wandb_mirror.log(logged)

                if config.checkpoint_every_segments and segment % config.checkpoint_every_segments == 0:
                    _save_checkpoint(run_dir, global_step, agent, skill_state=skill_state,
                                     keep_checkpoints=config.keep_checkpoints)
                if config.full_state_every_segments and segment % config.full_state_every_segments == 0:
                    _save_full_state(run_dir, global_step, agent=agent, skill_state=skill_state, archive_state=archive_state,
                                     carry=carry, rng=rng, global_step=global_step, segment=segment, expl_acc=expl_acc,
                                     last_media_step=last_media_step, last_eval_step=last_eval_step,
                                     include_archive=config.full_state_include_archive)
                if reached_time_limit or reached_chunk_limit:
                    stop_reason = "max_train_seconds" if reached_time_limit else "max_chunks"
                    break
        _save_checkpoint(run_dir, global_step, agent, skill_state=skill_state,
                         keep_checkpoints=config.keep_checkpoints)
        # The final full state is what a later resume continues from; kept forever (never rotated).
        _save_full_state(run_dir, global_step, agent=agent, skill_state=skill_state, archive_state=archive_state,
                         carry=carry, rng=rng, global_step=global_step, segment=segment, expl_acc=expl_acc,
                         last_media_step=last_media_step, last_eval_step=last_eval_step, keep=True,
                         include_archive=config.full_state_include_archive)
        final_eval = _run_eval(global_step, force_media=True)
        wandb_mirror.log({"env_steps": global_step, **_reference_keys(final_eval)})
        (run_dir / "train_summary.json").write_text(
            json.dumps(
                {"final_env_steps": global_step, "segments": segment, "stop_reason": stop_reason,
                 "encoder_optimizer_steps": int(skill_state.step),
                 "archive_size_transitions": int(archive_state.size),
                 "encoder_branch_simulator_steps": (
                     (global_step // (config.num_envs * config.episode_length))
                     * branch_horizon * config.num_envs if branch_encoder else 0),
                 "train_wall_seconds": time.perf_counter() - training_started, **_reference_keys(final_eval)},
                indent=2, sort_keys=True,
            )
            + "\n"
        )
        print(json.dumps({"final_eval": _reference_keys(final_eval)}, sort_keys=True), flush=True)
        return run_dir
    finally:
        wandb_mirror.finish()
