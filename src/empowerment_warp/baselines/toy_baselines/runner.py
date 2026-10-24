"""One host pipeline for GCE/VIM estimation and closed-loop PPO."""
from __future__ import annotations

import csv
import json
import os
import socket
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from empowerment_warp.baselines.toy_baselines.checkpoint import restore_checkpoint, save_checkpoint
from empowerment_warp.baselines.toy_baselines.config import (
    behavior_fingerprint, config_dict, effective_scale, event_updates, validate_config, write_config_artifacts)
from empowerment_warp.baselines.toy_baselines.evaluation import (
    evaluate_fields, evaluate_fixed_bank, make_anchor_diagnostic, make_controller_rewards,
    make_episode_rollout, make_estimate)
from empowerment_warp.baselines.toy_baselines.gce import GCE_METRICS, create_gce_train_state, make_gce_train_step
from empowerment_warp.baselines.toy_baselines.plotting import save_field_figure, save_rollout_media
from empowerment_warp.baselines.toy_baselines.ppo import (
    create_ppo_train_state, initialize_controller, make_collect_rollout, make_ppo_update)
from empowerment_warp.baselines.toy_baselines.probes import make_probe_collector
from empowerment_warp.baselines.toy_baselines.replay import (
    empty_probe_buffer, insert_probe_batch, make_replay_epoch, train_replay)
from empowerment_warp.baselines.toy_baselines.vim import VIM_METRICS, create_vim_train_states, make_vim_train_step
from empowerment_warp.baselines.toy_baselines.tracking import (
    record_wandb_failure, safe_wandb_finish, safe_wandb_log, write_wandb_status)


def _json_ready(value):
    if isinstance(value, (np.ndarray, jax.Array, np.generic)):
        return np.asarray(value).tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    return value


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


class MetricLogger:
    """Append stable-schema train/eval CSVs. Local writes precede W&B."""
    def __init__(self, output):
        self.output = output

    def log(self, split, metrics):
        row = {key: _json_ready(value) for key, value in metrics.items()}
        path = self.output / f'{split}.csv'
        exists = path.exists() and path.stat().st_size > 0
        fields = list(row)
        if exists:
            with path.open(newline='') as handle:
                fields = next(csv.reader(handle))
            if set(fields) != set(row):
                raise ValueError(f'{split} metric schema changed')
        with path.open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        safe_wandb_log(row, save_dir=str(self.output))


def _setup_wandb(config, output):
    identity = dict(wandb_mode=config.wandb_mode, local_run_dir=str(output),
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'))
    if config.wandb_mode != 'disabled':
        try:
            # Never couple the scientific run's lifetime to remote availability.
            run = wandb.init(project=config.wandb_project, group=config.wandb_group,
                             name=config.wandb_name or f'{config.method}-{config.environment.name}-{config.mode}-s{config.seed}',
                             job_type=config.mode, config=config_dict(config),
                             mode=config.wandb_mode, dir=str(output))
            run.define_metric('env_steps')
            run.define_metric('*', step_metric='env_steps')
            identity.update(wandb_entity=run.entity, wandb_project=run.project, wandb_run_id=run.id,
                            wandb_run_name=run.name, wandb_group=run.group, wandb_job_type=run.job_type,
                            wandb_url=run.url)
            (output / 'wandb_links.txt').write_text((run.url or '') + '\n')
        except Exception as exc:
            identity['failure'] = str(exc)
            record_wandb_failure('init', save_dir=str(output), exc=exc)
    _atomic_json(output / 'wandb_run.json', identity)
    write_wandb_status(str(output))


def _log_media(paths, step, output):
    if getattr(wandb, 'run', None) is None:
        return
    # Object construction can also fail; all referenced media is already local.
    try:
        payload = {'env_steps': step}
        for name, path in paths.items():
            if name.endswith('_png'):
                payload[f'media/{name[:-4]}'] = wandb.Image(str(path))
            elif name.endswith('_mp4'):
                payload[f'media/{name[:-4]}'] = wandb.Video(str(path), format='mp4')
        safe_wandb_log(payload, save_dir=str(output))
    except Exception as exc:
        record_wandb_failure('media', save_dir=str(output), exc=exc)
        with (output / 'wandb_pending_uploads.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(env_steps=step, paths={k: str(v) for k, v in paths.items()},
                                         error=str(exc))) + '\n')


def initialize_runtime(config):
    """Method-specific models/losses behind plain callbacks, not a plugin API."""
    keys = jax.random.split(jax.random.PRNGKey(config.seed), 9)
    source = decoder = None
    if config.method == 'gce':
        model, estimator = create_gce_train_state(keys[0], config.environment, config.estimator)
        step, metric_names = make_gce_train_step(model), GCE_METRICS
    else:
        source, source_state, decoder, decoder_state = create_vim_train_states(
            keys[0], keys[1], config.environment, config.estimator, config.vim,
            fixed_uniform=config.method == 'vim_uniform')
        model, estimator = source, (source_state, decoder_state)
        step, metric_names = make_vim_train_step(source, decoder), VIM_METRICS
    estimate = make_estimate(config, model)
    policy = ppo_state = controller = None
    controller_key = keys[5]
    if config.mode == 'controller':
        policy, ppo_state = create_ppo_train_state(keys[2], config.environment, config.ppo)
        controller, controller_key = initialize_controller(keys[5], config.environment, config.ppo)
    state = dict(
        estimator=estimator, buffer=empty_probe_buffer(config.environment, config.estimator),
        ppo_state=ppo_state, controller=controller,
        rng=dict(collection=keys[3], retention=keys[4], controller=controller_key,
                 training=keys[6], ppo=keys[7], evaluation=keys[8]),
        update=0, probe_steps=0, controller_steps=0, evaluation_steps=0,
        config_fingerprint=behavior_fingerprint(config))
    functions = dict(
        estimate=estimate, collect=make_probe_collector(config, source),
        epoch=make_replay_epoch(step, metric_names, config.estimator),
        diagnostic=make_anchor_diagnostic(config, model, estimate, decoder), policy=policy)
    if policy is not None:
        functions.update(
            collect_controller=make_collect_rollout(policy, config.environment, config.ppo),
            update_ppo=make_ppo_update(policy, config.environment, config.ppo),
            rewards=make_controller_rewards(config, estimate),
            rollouts={mode: make_episode_rollout(config, policy, estimate, deterministic=mode == 'deterministic')
                      for mode in ('deterministic', 'stochastic')})
    return state, functions


def estimator_params(config, state):
    estimator = state['estimator']
    return estimator.params if config.method == 'gce' else estimator[0].params


def training_update(config, state, functions):
    """Exactly one charged collection block followed by ten replay passes."""
    rng = state['rng']
    observations = jnp.empty((0, config.environment.observation_dim), jnp.float32)
    metrics = {}
    if config.mode == 'controller':
        state['controller'], rollout, rng['controller'] = functions['collect_controller'](
            state['ppo_state'], state['controller'], rng['controller'])
        observations = rollout.observations
        state['controller_steps'] += int(rollout.dones.size)
    batch, _, costs, rng['collection'] = functions['collect'](
        estimator_params(config, state), observations, rng['collection'])
    probe_steps = int(costs['probe_steps'])
    if probe_steps != config.probe_steps_per_update or len(batch.observations) != config.tuples_per_update:
        raise AssertionError('collector disagrees with resolved simulator/tuple budget')
    state['probe_steps'] += probe_steps
    rng['retention'], insert_key = jax.random.split(rng['retention'])
    state['buffer'] = insert_probe_batch(state['buffer'], batch, insert_key)
    state['estimator'], learner, rng['training'] = train_replay(
        functions['epoch'], state['estimator'], state['buffer'], rng['training'], config.estimator.replay_passes)
    if config.mode == 'controller':
        rewards, values = functions['rewards'](estimator_params(config, state), rollout, state['controller'])
        state['ppo_state'], ppo_metrics, _, _, rng['ppo'] = functions['update_ppo'](
            state['ppo_state'], rollout, rewards, state['controller'], rng['ppo'])
        metrics.update({f'training/ppo/{k}': v for k, v in ppo_metrics.items()})
        metrics['training/reward_mean'] = rewards.mean()
    state['update'] += 1
    metrics.update({f'training/estimator/{k}': v for k, v in learner.items()})
    metrics.update({f'training/collection/{k}': v for k, v in costs.items()})
    metrics.update({
        'env_steps': state['probe_steps'] + state['controller_steps'], 'update': state['update'],
        'training/probe_steps': state['probe_steps'], 'training/controller_steps': state['controller_steps'],
        'training/replay_count': state['buffer'].count, 'training/collected_tuples': len(batch.observations)})
    if metrics['env_steps'] != state['update'] * config.steps_per_update or metrics['env_steps'] > config.env_steps:
        raise AssertionError('training exceeded or miscounted its simulator-step budget')
    if not all(np.all(np.isfinite(np.asarray(v))) for v in metrics.values()):
        raise FloatingPointError('nonfinite training metric')
    return metrics, observations


def _save(config, state, name):
    return save_checkpoint(config.output_dir, name, state, controller_steps=state['controller_steps'],
                           probe_steps=state['probe_steps'])


def evaluate(config, state, functions, visitation, *, media):
    step = state['probe_steps'] + state['controller_steps']
    params = estimator_params(config, state)
    decoder_params = None if config.method == 'gce' else state['estimator'][1].params
    start = time.monotonic()
    # Fixed test randomness is independent of collection, SGD and event cadence.
    metrics, steps = functions['diagnostic'](params, decoder_params, state['rng']['evaluation'])
    paths, traces = {}, {}
    if config.mode == 'controller':
        behavior, traces, arrays, policy_steps = evaluate_fixed_bank(
            config, state['ppo_state'].params, params, state['controller'].observation_norm, functions['rollouts'])
        metrics.update(behavior)
        steps += policy_steps
        np.savez_compressed(Path(config.output_dir) / f'traces_step_{step:09d}.npz',
                            **{k: np.asarray(v) for k, v in arrays.items()})
    state['evaluation_steps'] += steps
    metrics.update(env_steps=step, update=state['update'], evaluation_steps=state['evaluation_steps'],
                   **{'time/evaluation_seconds': time.monotonic()-start})
    _save(config, state, f'step_{step:09d}')
    start = time.monotonic()
    if media:
        fields = evaluate_fields(
            config, functions['estimate'], params, visitation, functions['policy'],
            state['ppo_state'].params if config.mode == 'controller' else None,
            state['controller'].observation_norm if config.mode == 'controller' else None)
        directory = Path(config.output_dir) / 'media'
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / f'fields_step_{step:09d}.npz',
                            **{k: np.asarray(v) for k, v in fields.items()})
        paths.update(save_field_figure(config, fields, directory, step))
        for mode, trace in traces.items():
            paths.update({f'{mode}_{k}': v for k, v in save_rollout_media(
                config, trace, directory, step, method_label=config.method, mode=mode).items()})
    metrics['time/media_seconds'] = time.monotonic()-start
    if not all(np.all(np.isfinite(np.asarray(v))) for v in metrics.values()):
        raise FloatingPointError('nonfinite evaluation metric')
    return metrics, paths


def run_experiment(config, *, max_updates: int | None = None):
    validate_config(config)
    if max_updates is not None and max_updates < 1:
        raise ValueError('max_updates must be positive')
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'run_manifest.json').exists() and not config.resume:
        raise FileExistsError(f'{output} already contains a run; use a fresh output or --resume')
    state, functions = initialize_runtime(config)
    if config.resume:
        state = restore_checkpoint(config.resume, state)
        if state['config_fingerprint'] != behavior_fingerprint(config):
            raise ValueError('resume recipe differs from checkpoint')
        if state['update'] >= config.updates:
            raise ValueError('resume budget must extend beyond the checkpoint')
        for split in ('train', 'eval'):
            path = output / f'{split}.csv'
            if path.exists():
                with path.open(newline='') as handle:
                    ahead = any(int(row['update']) > state['update'] for row in csv.DictReader(handle))
                if ahead:
                    raise ValueError('output logs are ahead of the checkpoint; resume into a fresh output directory')
    stop_update = (config.updates if max_updates is None
                   else min(config.updates, state['update'] + max_updates))
    write_config_artifacts(config)
    manifest = dict(status='running', method=config.method, mode=config.mode, seed=config.seed,
                    purpose='Shared-step GCE/VIM comparison; smoke budgets validate execution, not convergence.',
                    hostname=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                    submit_script=os.environ.get('BASELINE_SUBMIT_SCRIPT'), source_root=os.environ.get('BASELINE_SNAPSHOT_ROOT'),
                    devices=[str(d) for d in jax.devices()], config_fingerprint=behavior_fingerprint(config),
                    effective_scale=effective_scale(config), resumed_from=config.resume or None)
    _atomic_json(output / 'run_manifest.json', manifest)
    _setup_wandb(config, output)
    logger = MetricLogger(output)
    started = time.monotonic()
    media_updates = set(event_updates(config, config.evaluation.media_interval_env_steps))
    eval_updates = set(event_updates(config, config.evaluation.interval_env_steps)) | media_updates
    visitation = jnp.empty((0, config.environment.observation_dim), jnp.float32)
    latest_eval = {}
    try:
        if not config.resume:
            latest_eval, paths = evaluate(config, state, functions, visitation, media=True)
            logger.log('eval', latest_eval)
            _log_media(paths, 0, output)
            _save(config, state, 'latest')
        while state['update'] < stop_update:
            tick = time.monotonic()
            metrics, visitation = training_update(config, state, functions)
            metrics['time/update_seconds'] = time.monotonic()-tick
            logger.log('train', metrics)
            print(json.dumps(_json_ready(metrics)), flush=True)
            if state['update'] in eval_updates:
                latest_eval, paths = evaluate(config, state, functions, visitation,
                                              media=state['update'] in media_updates)
                logger.log('eval', latest_eval)
                _log_media(paths, metrics['env_steps'], output)
                _save(config, state, 'latest')
        completed = state['update'] == config.updates
        _save(config, state, 'final' if completed else 'latest')
        summary = dict(status='completed' if completed else 'partial',
                       env_steps=state['probe_steps']+state['controller_steps'],
                       probe_steps=state['probe_steps'], controller_steps=state['controller_steps'],
                       evaluation_steps=state['evaluation_steps'], updates=state['update'],
                       replay_count=int(state['buffer'].count), elapsed_seconds=time.monotonic()-started,
                       effective_scale=effective_scale(config), final_evaluation=latest_eval)
        _atomic_json(output / 'summary.json', summary)
        _atomic_json(output / 'run_manifest.json', dict(manifest, status=summary['status']))
        return summary
    except Exception as exc:
        _atomic_json(output / 'run_manifest.json', dict(manifest, status='failed', failure=repr(exc),
                                                       update=state['update']))
        raise
    finally:
        safe_wandb_finish(save_dir=str(output))
