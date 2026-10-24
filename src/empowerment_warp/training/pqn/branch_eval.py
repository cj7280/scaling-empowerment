"""Fresh PQN switching evaluation and fixed-probe comparison media."""

from __future__ import annotations

import json
import hashlib
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.training.pqn import branch_protocol as reference
from empowerment_warp.agents.skill_classifier import sample_skills
from empowerment_warp.evaluation_protocol import evaluation_keys, POTENTIAL_STEPS
from empowerment_warp.training.pqn.env import PQNEnv


PROTOCOL = 'craftax-128-fresh-switching-v2'


def _make_recurrent_eval_actor(agent, rng):
    """Retain observed memory at command changes when the run enables it."""
    actor = reference._StatefulRecurrentEvalActor(agent, rng)
    # Configs written before this flag existed reset memory at the switch.
    if agent.config.get('pqn_preserve_memory_at_switch', False):
        actor._reset_on_skill_boundary = False
    return actor


def _player_floor_cells(game_state):
    """Craftax player cell as (floor, row, column); Classic has floor zero."""
    position = jnp.asarray(game_state.player_position, jnp.int32)
    floor = (jnp.asarray(game_state.player_level, jnp.int32)
             if hasattr(game_state, "player_level") else
             jnp.zeros(position.shape[:-1], jnp.int32))
    return jnp.concatenate((floor[..., None], position), axis=-1)


def make_switching_rollout(prefix_env, branch_env, agent, *, prefix_steps,
                           episode_length, skills_per_context, deterministic=False):
    """Fresh shared prefixes, independent skill draws, and first-episode outcomes."""
    if not 0 < prefix_steps < episode_length:
        raise ValueError("evaluation requires a potential prefix followed by a skill phase")
    contexts = prefix_env.num_envs
    count = branch_env.num_envs
    if count != contexts * skills_per_context:
        raise ValueError("branch width must equal contexts times skills per context")
    source = jnp.repeat(jnp.arange(contexts), skills_per_context)
    # Configs written before this flag existed reset memory at the switch.
    preserve_memory = bool(agent.config.get('pqn_preserve_memory_at_switch', False))

    @jax.jit
    def rollout(variables, epsilon, reset_key, action_key, skills):
        initial = prefix_env.reset(jax.random.split(reset_key, contexts))

        def scan(env, carry, commands, anchors, steps, potential):
            def step(carry, _):
                current, rnn, reset, alive, returns, lengths, rng = carry
                rng, key = jax.random.split(rng)
                _, actions, next_rnn = agent.sample_actions_recurrent_from_params(
                    variables, current.obs, commands, current.obs if potential else anchors,
                    rnn, reset, key, epsilon, deterministic=deterministic,
                )
                reached = env.step(current, actions)
                done = reached.terminated | reached.truncated
                def retain(old, new):
                    mask = alive.reshape((env.num_envs,) + (1,) * (old.ndim - 1))
                    return jnp.where(mask, new, old)
                # Include terminal unlocks; never restart a lane after its first death.
                kept = jax.tree_util.tree_map(retain, current, reached)
                next_carry = (kept, next_rnn, jnp.zeros_like(reset), alive & ~done,
                              returns + jnp.where(alive, reached.reward, 0),
                              lengths + alive.astype(jnp.int32), rng)
                return next_carry, (_player_floor_cells(kept.game_state), alive,
                                    alive & done, jnp.linalg.norm(commands, axis=-1) > 0)
            return jax.lax.scan(step, carry, None, length=steps)

        carry = (initial, agent.initial_recurrent_state(contexts),
                 jnp.ones(contexts, jnp.float32), jnp.ones(contexts, bool),
                 jnp.zeros(contexts, jnp.float32), jnp.zeros(contexts, jnp.int32), action_key)
        prefix, (prefix_cells, prefix_valid, prefix_done, prefix_skill_on) = scan(
            prefix_env, carry, jnp.zeros((contexts, agent.config['skill_dim'])),
            initial.obs, prefix_steps, True,
        )
        state, rnn, _, alive, returns, lengths, _ = prefix
        fork = jax.tree_util.tree_map(lambda value: value[source], state)
        fork = fork._replace(rng=jax.random.split(jax.random.fold_in(action_key, 1), count))
        # The packed state precedes fork.obs and includes both networks and the
        # previous action. Each branch consumes the switch observation once.
        switch_reset = jnp.full(count, not preserve_memory, jnp.float32)
        carry = (fork, rnn[source], switch_reset, alive[source],
                 returns[source], lengths[source], jax.random.fold_in(action_key, 2))
        final, (skill_cells, skill_valid, skill_done, skill_on) = scan(
            branch_env, carry, skills, fork.obs, episode_length - prefix_steps, False,
        )
        prefix_cells = jnp.concatenate((_player_floor_cells(initial.game_state)[None], prefix_cells))
        prefix_valid = jnp.concatenate((jnp.ones((1, contexts), bool), prefix_valid))
        return dict(
            cells=jnp.concatenate((prefix_cells[:, source], skill_cells)),
            valid=jnp.concatenate((prefix_valid[:, source], skill_valid)),
            prefix_cells=prefix_cells, prefix_valid=prefix_valid,
            initial_achievements=initial.game_state.achievements[source],
            prefix_achievements=state.game_state.achievements[source],
            final_achievements=final[0].game_state.achievements,
            episode_returns=final[4], episode_lengths=final[5],
            skills=skills, contexts=source,
            dones=jnp.concatenate((prefix_done[:, source], skill_done)),
            skill_on=jnp.concatenate((prefix_skill_on[:, source], skill_on)),
            reset_keys=jax.random.split(reset_key, contexts), initial_observations=initial.obs,
        )
    return rollout


def summarize_potential_policy_trace(cells, valid):
    """Mean per-world visitation and net grid displacement, including reset."""
    cells = np.asarray(cells, dtype=np.int32)
    valid = np.asarray(valid, dtype=bool)
    if cells.ndim != 3 or cells.shape[-1] != 3 or valid.shape != cells.shape[:2]:
        raise ValueError("expected cells [time, env, (floor,row,column)] and matching valid")
    if not np.all(valid[0]):
        raise ValueError("reset cell must be valid in every probe world")
    horizon = cells.shape[0] - 1
    num_envs = cells.shape[1]
    counts = valid[1:].sum(axis=0)
    visited = np.asarray([
        len({tuple(cell) for cell in cells[valid[:, lane], lane]})
        for lane in range(num_envs)
    ], dtype=np.float64)
    final = cells[counts, np.arange(num_envs)]
    same_floor = final[:, 0] == cells[0, :, 0]
    distance = np.linalg.norm((final[:, 1:] - cells[0, :, 1:]).astype(np.float64), axis=-1)
    return {
        "cells_visited_mean": float(visited.mean()) if num_envs else 0.0,
        "displacement_mean": float(distance[same_floor].mean()) if np.any(same_floor) else 0.0,
        "cell_size": 1.0,
        "horizon_steps": float(horizon),
        "num_envs": float(num_envs),
        "valid_steps_mean": float(counts.mean()) if num_envs else 0.0,
        "early_termination_fraction": float(np.mean(counts < horizon)) if num_envs else 0.0,
        "floor_changed_fraction": float(np.mean(~same_floor)) if num_envs else 0.0,
        "same_floor_fraction": float(np.mean(same_floor)) if num_envs else 0.0,
    }


class NativeScalarEvalEnv:
    """Expose the reference media protocol's small state/step/render interface."""

    def __init__(self, config):
        self._env = PQNEnv(config.environment, num_envs=1,
                           episode_length=config.eval_episode_length or config.episode_length,
                           reset_ratio=1)
        self._episode_limit = config.eval_episode_length or config.episode_length
        self._achievement_key_by_col = self._env.achievement_columns
        self._fixed_reset_seed = config.eval_fixed_reset_seed
        self._state = None
        self._step = jax.jit(self._env.step)
        from craftax.craftax_classic.renderer import render_craftax_pixels
        pixels = {"agent": 7, "img": 16, "human": 64}[config.eval_render_block_pixel_size]
        self._render = jax.jit(lambda state: render_craftax_pixels(state, block_pixel_size=pixels))

    def reset(self, *, seed):
        seed = self._fixed_reset_seed if self._fixed_reset_seed >= 0 else seed
        self._state = self._env.reset(jax.random.split(jax.random.PRNGKey(seed), 1))
        return np.asarray(self._state.obs[0]), {}

    def step(self, action):
        self._state = self._step(self._state, jnp.asarray([np.argmax(action)], jnp.int32))
        state = self._state
        return (np.asarray(state.obs[0]), float(state.reward[0]),
                bool(state.terminated[0]), bool(state.truncated[0]), {})

    def current_state(self):
        return jax.tree_util.tree_map(lambda value: value[0], self._state.game_state)

    def render(self):
        return np.asarray(self._render(self.current_state())).clip(0, 255).astype(np.uint8)


class NativeHoldoutDriver:
    """Cache compact metric scans separately from the fixed media probes."""

    def __init__(self, config, agent, library):
        self.config, self.agent, self.library = config, agent, jnp.asarray(library)
        self.num_envs = config.eval_contexts * config.eval_skills_per_context
        self.media_env = None
        self._rollouts = {}

    def evaluate(self, agent, *, seed, env_steps, contexts, skills_per_context,
                 deterministic=False):
        cache_key = (contexts, skills_per_context, deterministic)
        if cache_key not in self._rollouts:
            horizon = self.config.eval_episode_length or self.config.episode_length
            prefix_env = PQNEnv(self.config.environment, num_envs=contexts,
                                episode_length=horizon, reset_ratio=1)
            branch_env = PQNEnv(self.config.environment, num_envs=contexts * skills_per_context,
                                episode_length=horizon, reset_ratio=1)
            rollout = make_switching_rollout(
                prefix_env, branch_env, self.agent, prefix_steps=POTENTIAL_STEPS,
                episode_length=horizon, skills_per_context=skills_per_context,
                deterministic=deterministic,
            )
            self._rollouts[cache_key] = (rollout, prefix_env.achievement_columns,
                                       self.config.skill_config(prefix_env.observation_size))
        rollout, columns, skill_config = self._rollouts[cache_key]
        keys = evaluation_keys(self.config.environment, seed, env_steps)
        skills = sample_skills(skill_config, keys[2], contexts * skills_per_context)
        arrays = rollout(
            {'params': agent.network.params, 'batch_stats': agent.batch_stats},
            jnp.asarray(0.0 if deterministic else agent.exploration_epsilon()),
            keys[0], keys[1], skills,
        )
        return {k: np.asarray(v) for k, v in jax.device_get(arrays).items()}, columns, keys


def phase_achievement_counts(initial, prefix, final):
    """Count distinct unlocks; achievements already present at the switch are not new."""
    initial, prefix, final = (np.asarray(a, dtype=bool) for a in (initial, prefix, final))
    if np.any(initial & ~prefix) or np.any(prefix & ~final):
        raise ValueError('achievement masks must persist through the same episode')
    potential = (prefix & ~initial).sum(axis=-1)
    skill = (final & ~prefix).sum(axis=-1)
    total = (final & ~initial).sum(axis=-1)
    return dict(total=total, potential=potential, skill=skill)


def evaluate_agent(config, agent, library, *, env_steps, output_dir, driver=None,
                   media=False, seed=None, contexts=None, skills_per_context=None, checkpoint=None,
                   deterministic=False):
    """Score fresh switching trajectories; keep fixed probes for comparison videos."""
    from empowerment_warp.evaluation_checks import source_hashes
    driver = driver or NativeHoldoutDriver(config, agent, library)
    seed = config.seed if seed is None else seed
    contexts = config.eval_contexts if contexts is None else contexts
    skills_per_context = config.eval_skills_per_context if skills_per_context is None else skills_per_context
    print(f'[eval] fresh switching: {contexts} contexts x {skills_per_context} skills; '
          f'prefix={POTENTIAL_STEPS}; no separate potential-only probe', flush=True)
    start = time.perf_counter()
    arrays, columns, keys = driver.evaluate(
        agent, seed=seed, env_steps=env_steps, contexts=contexts,
        skills_per_context=skills_per_context, deterministic=deterministic,
    )
    wall_seconds = time.perf_counter() - start
    counts = phase_achievement_counts(arrays['initial_achievements'],
                                     arrays['prefix_achievements'], arrays['final_achievements'])
    total_flags = arrays['final_achievements'].astype(bool) & ~arrays['initial_achievements'].astype(bool)
    metrics = {'evaluation/' + name: value for name, value in reference.achievement_summary(
        {name: 100.0 * total_flags[:, column] for column, name in columns},
        episode_return_values=arrays['episode_returns']).items()}
    metrics.update({
        'evaluation/potential_policy/achievement_count': float(counts['potential'].mean()),
        'evaluation/skill_policy/new_achievement_count': float(counts['skill'].mean()),
        'evaluation/episode_return': float(arrays['episode_returns'].mean()),
        'evaluation/episode_length': float(arrays['episode_lengths'].mean()),
        'evaluation/branch_metric_backend_gpu': float(jax.devices()[0].platform == 'gpu'),
        'evaluation/branch_metric_contexts': float(contexts),
        'evaluation/branch_metric_skills_per_context': float(skills_per_context),
        'evaluation/completed_episodes': float(len(counts['total'])),
        'evaluation/stochastic_policy': float(not deterministic),
        'evaluation/epsilon': float(0.0 if deterministic else agent.exploration_epsilon()),
    })
    metrics.update({f'evaluation/potential_policy/player/{name}': value for name, value in
                   summarize_potential_policy_trace(arrays['prefix_cells'], arrays['prefix_valid']).items()})
    output_dir = Path(output_dir)
    step_dir = output_dir / 'figures' / f'step_{int(env_steps)}'
    step_dir.mkdir(parents=True, exist_ok=True)
    trace_path = step_dir / 'evaluation_trajectories.npz'
    np.savez_compressed(trace_path, **arrays, **{f'{k}_achievement_counts': v for k, v in counts.items()},
                        env_steps=np.asarray(env_steps), rng_keys=np.asarray(keys))
    out = {'env_steps': int(env_steps), 'checkpoint_step': int(agent.env_steps),
           'learner_updates': int(agent.n_updates), **metrics,
           'source_run_dir': str(Path(config.run_dir).resolve()),
           'checkpoint': str(Path(checkpoint).resolve()) if checkpoint is not None else None,
           'agent_config': dict(agent.config),
           'protocol': {'name': PROTOCOL, 'seed': int(seed), 'rng_keys': np.asarray(keys).tolist(),
                        'engine': 'pqn', 'environment': config.environment,
                        'backend': jax.devices()[0].platform,
                        'sampling': 'fresh_prior_and_resets_per_checkpoint',
                        'evaluation_source': source_hashes('pqn', config.environment),
                        'contexts': contexts, 'skills_per_context': skills_per_context,
                        'num_envs': contexts * skills_per_context, 'prefix_steps': POTENTIAL_STEPS,
                        'episode_length': config.eval_episode_length or config.episode_length,
                        # Configs written before this flag existed reset memory at the switch.
                        'pqn_preserve_memory_at_switch': bool(
                            agent.config.get('pqn_preserve_memory_at_switch', False)),
                        'deterministic': bool(deterministic), 'epsilon': metrics['evaluation/epsilon'],
                        'skills': 'Independent prior draws per continuation, fresh at every checkpoint',
                        'resets': 'Fresh per checkpoint; matched across methods at the same seed and step',
                        'termination': 'First episode only, terminal action included; no restart',
                        'phase_counts': 'Total = potential prefix + new unlocks during skill continuation',
                        'trace': str(trace_path), 'wall_seconds': wall_seconds,
                        'adapter_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    if media:
        print('[eval] collecting fixed media context and skill probes', flush=True)
        # Media keeps its fixed seed and library; it does not contribute metric rows.
        cfg = {**dict(agent.config), 'benchmark_backend': 'craftax',
               'max_episode_steps': config.eval_episode_length or config.episode_length,
               'potential_steps': POTENTIAL_STEPS, 'video_frame_skip': config.eval_video_frame_skip,
               'craftax_render_block_pixel_size': config.eval_render_block_pixel_size}
        if driver.media_env is None:
            driver.media_env = NativeScalarEvalEnv(config)
        rollouts = reference._collect_jax_game_branch_skill_rollouts(
            driver.media_env, _make_recurrent_eval_actor(agent, jax.random.PRNGKey(seed)),
            cfg, library, count=config.eval_media_skill_count, warmup_steps=POTENTIAL_STEPS,
            video_frame_skip=config.eval_video_frame_skip, eval_temperature=1.0,
            eval_deterministic=deterministic, supports_deterministic=True, seed=seed,
        )
        out['figures'] = write_branch_figures(output_dir, env_steps, rollouts, config,
                                            potential_steps=POTENTIAL_STEPS)
        out['figures'].update(checkpoint=out['checkpoint'], source_run_dir=out['source_run_dir'],
                              protocol={'name': 'fixed-media-probes', 'prefix_steps': POTENTIAL_STEPS,
                                        'reset_seed': config.eval_fixed_reset_seed,
                                        'contributes_to_metrics': False})
        (step_dir / 'trajectory_artifacts.json').write_text(json.dumps(out['figures'], indent=2) + '\n')
    (step_dir / 'evaluation.json').write_text(json.dumps(out, indent=2) + '\n')
    return out


def evaluation_skill_library(config):
    """Fixed probes for comparison media only; metric skills are drawn separately."""
    rng = np.random.RandomState(int(config.seed) + 12345)
    probes = rng.randn(int(config.num_skills), int(config.skill_dim))
    probes = probes / (np.linalg.norm(probes, axis=1, keepdims=True) + 1e-8)
    return jnp.asarray(probes, jnp.float32)


def load_run(run_dir: Path):
    """Restore the saved architecture, including old checkpoints' recurrence mode."""
    from empowerment_warp.training.pqn.config import PotentialPQNRunConfig
    config = PotentialPQNRunConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    probe = PQNEnv(config.environment, num_envs=1, episode_length=config.episode_length, reset_ratio=1)
    manifest = json.loads((run_dir / "reference_manifest.json").read_text())
    # Rebuild the saved potential-PQN controller before evaluating its checkpoint.
    agent = config.agent_class().create(config.seed, jnp.zeros((1, probe.observation_size)),
                                        jnp.zeros((1, probe.action_size)), manifest["agent_config"])
    library = evaluation_skill_library(config)
    return config, agent, library, NativeHoldoutDriver(config, agent, library)


def evaluate_checkpoint(run_dir: Path, checkpoint: Path, *, seed=None, contexts=None,
                        skills_per_context=None, media=False, loaded=None, output_dir=None):
    config, agent, library, driver = loaded if loaded is not None else load_run(run_dir)
    # Old checkpoints retain their training config; default scoring uses today's protocol.
    contexts = 128 if contexts is None else contexts
    skills_per_context = 1 if skills_per_context is None else skills_per_context
    with checkpoint.open("rb") as handle:
        payload = pickle.load(handle)
    agent = agent.replace(network=agent.network.replace(params=jax.device_put(payload["params"])),
                          batch_stats=jax.device_put(payload["batch_stats"]),
                          env_steps=jnp.asarray(payload["env_steps"], jnp.int32),
                          n_updates=jnp.asarray(payload["n_updates"], jnp.int32))
    out = evaluate_agent(config, agent, library, env_steps=payload["env_steps"],
                         output_dir=output_dir or run_dir / "eval_reference", driver=driver,
                         media=media, seed=seed, contexts=contexts, skills_per_context=skills_per_context,
                         checkpoint=checkpoint)
    return out


def write_branch_figures(run_dir: Path, env_steps: int, rollouts: list, config,
                        *, potential_steps=None) -> dict:
    """Reference figure set into <run_dir>/figures/step_<env_steps>/ (+ latest/)."""
    from empowerment_warp.plotting.craftax_branching import (
        create_craftax_branching_skill_grid_video_bundle, create_craftax_branching_stills_bundle,
    )
    cfg = {"num_skills": config.num_skills, "skill_dim": config.skill_dim,
           "potential_steps": config.potential_steps if potential_steps is None else potential_steps,
           "max_episode_steps": config.eval_episode_length or config.episode_length,
           "video_frame_skip": config.eval_video_frame_skip,
           "craftax_render_block_pixel_size": config.eval_render_block_pixel_size}
    dirs = [str(run_dir / "figures" / f"step_{env_steps}"), str(run_dir / "figures" / "latest")]
    stills = create_craftax_branching_stills_bundle(branch_rollouts=rollouts, output_dirs=dirs, config=cfg) or {}
    video = create_craftax_branching_skill_grid_video_bundle(
        branch_rollouts=rollouts, output_dirs=dirs, filename_prefix="eval_branching_skill_grid", config=cfg,
    ) or {}
    manifest = {"env_steps": int(env_steps), "prefix_seed_offset": 0,
                "fps": int(video.get("craftax_branching_skill_grid_video_fps", 24)),
                "skill_indices": [int(r["skill_index"]) for r in rollouts if not r["is_potential_continuation"]],
                "saved_paths": {**stills.get("saved_paths", {}), **video.get("saved_paths", {})}}
    (Path(dirs[0]) / "trajectory_artifacts.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ----------------------------------------------------------------------------- W&B push + backfill
# Panel names match the baseline runs' media keys.
WANDB_MEDIA_KEYS = {
    "craftax_branching_stills.png": "media/craftax_branching_stills",
    "craftax_branching_stats.png": "media/craftax_branching_stats",
    "eval_branching_skill_grid.mp4": "media/craftax_branching_skill_grid",
}


def push_to_wandb(run_dir: Path, row: dict, figures_dir: Path | None = None) -> str | None:
    """Log one eval row (+ media) into the run's own W&B run, keyed by env_steps."""
    import wandb
    meta = json.loads((run_dir / "wandb_run.json").read_text())
    # Companion run "<name>-eval" in the SAME group: the training process is a live
    # writer on the main run, and two concurrent writers on one W&B run is undefined.
    run = wandb.init(id=f"{meta['id']}-eval", name=f"{meta.get('name','run')}-eval", group=meta.get("group"),
                     project=meta.get("project"), entity=meta.get("entity"), job_type="branch_eval",
                     resume="allow", reinit=True, settings=wandb.Settings(silent=True),
                     config={"source_run": meta.get("url"), "protocol": "reference shared-prefix branched skill eval"})
    try:
        run.define_metric("env_steps"); run.define_metric("evaluation/*", step_metric="env_steps")
        payload = {k: v for k, v in row.items() if k == "env_steps" or k.startswith("evaluation/")}
        if figures_dir is not None and figures_dir.is_dir():
            for fname, key in WANDB_MEDIA_KEYS.items():
                f = figures_dir / fname
                if f.exists():
                    payload[key] = wandb.Video(str(f), fps=24, format="mp4") if f.suffix == ".mp4" else wandb.Image(str(f))
        run.log(payload)
        return run.url
    finally:
        run.finish()


def backfill_run(run_dir: Path, *, seed=None, contexts=None, skills_per_context=None,
                 media_every: int = 50_000_000, wandb_push: bool = False, checkpoints=None) -> list[dict]:
    """Backfill checkpoints separately, with the reference delta-gated media cadence."""
    if media_every < 0:
        raise ValueError("media_every must be nonnegative")
    target = run_dir / "eval_reference.jsonl"
    done = {}
    if target.exists():
        for line in target.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[int(row["env_steps"])] = row
    ckpts = checkpoints or sorted((run_dir / "checkpoints").glob("params_*.pkl"))
    loaded = load_run(run_dir)
    rows, last_media = [], None
    for ck in ckpts:
        step = int(ck.stem.split("_")[-1])
        if step in done:
            if done[step].get("figures"):
                last_media = step
            continue
        want_media = last_media is None or media_every == 0 or step - last_media >= media_every
        out = evaluate_checkpoint(run_dir, ck, seed=seed, contexts=contexts, skills_per_context=skills_per_context, media=want_media, loaded=loaded)
        if want_media:
            last_media = step
        with target.open("a") as fh:
            fh.write(json.dumps(out, sort_keys=True) + "\n")
        if wandb_push:
            out["wandb_url"] = push_to_wandb(run_dir, out, run_dir / "eval_reference" / "figures" / f"step_{step}" if want_media else None)
        print(json.dumps({k: v for k, v in out.items() if not k.startswith("evaluation/achievement/")}, sort_keys=True), flush=True)
        rows.append(out)
    return rows
