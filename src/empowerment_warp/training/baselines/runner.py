"""Minimal controller/probe/replay/PPO loop on the existing Warp environments."""
import csv
from dataclasses import asdict, replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import time

import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.checkpointing import CheckpointStore
from empowerment_warp.evaluation_protocol import evaluation_keys, make_rollout, save_evaluation
from empowerment_warp.training.sac.config import default_recipe
from empowerment_warp.core.rollout import coverage_counts, coverage_floor_masks
from empowerment_warp.envs.factory import make_environment
from empowerment_warp.tracking import WandbMirror
from empowerment_warp.agents.latent_gce import (
    VECTOR_GCE_METRICS, create_vector_gce_train_state, identity_moments, make_vector_gce_train_step,
    replay_moments, standardize, vector_capacity_from_observations, vector_losses,
)
from empowerment_warp.agents.ppo import (
    controller_learning_rate, create_ppo_train_state, make_ppo_update, normalize_observations,
)
from empowerment_warp.agents.vim import (
    VIM_METRICS, create_vim_train_states, empowerment_from_observations, make_vim_train_step, vim_diagnostics,
)
from .collector import (EXPLORATION_DISPLACEMENT_FIELDS, exploration_specs, initialize,
                        make_controller_collector, make_probe_collector)
from .config import GCEConfig, VIMModelConfig
from .replay import empty_probe_buffer, insert_probe_batch, make_replay_epoch, train_replay


def make_benchmark_environment(recipe, *, num_envs=None):
    # Resolve benchmark identity before resizing physics; SAC batching is unused.
    config = replace(default_recipe(recipe.environment), num_envs=recipe.num_envs if num_envs is None else num_envs,
                     episode_length=recipe.episode_steps, antsoccer_preset=recipe.reset_preset,
                     allow_reference_deviation=True,
                     allow_unpromoted_physics=True)
    return make_environment(config)


def make_standard_evaluator(recipe, policy, env=None):
    """Adapt the saved PPO actor and moments to the shared stochastic rollout."""
    if env is None:
        env = make_benchmark_environment(recipe, num_envs=128)

    def actions(parameters, observations, anchors, skills, action_keys):
        params, observation_norm = parameters
        if recipe.ppo.normalize_actor_observations:
            observations = normalize_observations(observation_norm, observations,
                epsilon=recipe.ppo.normalization_epsilon, clip=recipe.ppo.normalization_clip)
        distribution, _ = policy.apply({'params': params}, observations)
        mean, std = distribution.mean(), distribution.stddev()
        noise = jax.vmap(lambda key: jax.random.normal(key, (env.action_size,), dtype=mean.dtype))(action_keys)
        return jnp.tanh(mean + std * noise)

    return make_rollout(env, actions, horizon=100)


def build(recipe, env=None):
    if env is None:
        env = make_benchmark_environment(recipe)
    spec = recipe.model_environment
    if (env.observation_size, env.action_size) != (spec.observation_dim, spec.action_dim):
        raise ValueError('Warp observation/action dimensions differ from resolved recipe')
    actual_indices = env.encoder_observation_indices(
        {'ant':'full', 'ant_soccer':'ball_xy', 'builderbench_corner3':'object_pos'}[recipe.environment])
    if actual_indices != recipe.outcome_indices:
        raise ValueError('outcome view differs from Warp benchmark')
    keys = jax.random.split(jax.random.PRNGKey(recipe.seed), 9)
    source = None
    if recipe.method == 'gce_vector':
        model, estimator = create_vector_gce_train_state(
            keys[0], spec, recipe.effective_estimator, latent_dim=recipe.vector_gce_latent_dim,
            bound_action_decoder=recipe.vector_gce_bound_action_decoder)
        update, names = make_vector_gce_train_step(model), VECTOR_GCE_METRICS
        cfg = GCEConfig()
        def estimate(params, observations):
            return vector_capacity_from_observations(
                model, params, observations, power=cfg.channel_power,
                tolerance=cfg.water_filling_tolerance,
                max_iterations=cfg.water_filling_max_iterations)[0]
    else:
        source, source_state, decoder, decoder_state = create_vim_train_states(
            keys[0], keys[1], spec, recipe.estimator, VIMModelConfig())
        estimator = source_state, decoder_state
        update, names = make_vim_train_step(source, decoder), VIM_METRICS
        def estimate(params, observations):
            return empowerment_from_observations(source, params, observations, beta=1.)
    policy, ppo = create_ppo_train_state(keys[2], spec, recipe.ppo, total_updates=recipe.updates)
    state = dict(estimator=estimator, ppo=ppo, controller=initialize(env, keys[3]),
                 buffer=empty_probe_buffer(spec, recipe.estimator), update=jnp.asarray(0),
                 rng=dict(controller=keys[4], probes=keys[5], retention=keys[6],
                          training=keys[7], ppo=keys[8]))
    if recipe.vector_gce_normalize_inputs:
        state['estimator_norm'] = identity_moments(spec)
    functions = dict(collect=make_controller_collector(env, policy, recipe),
                     probe=make_probe_collector(env, recipe, source), estimate=jax.jit(estimate),
                     epoch=make_replay_epoch(update, names, recipe.estimator),
                     ppo_update=make_ppo_update(policy, spec, recipe.ppo),
                     standard_evaluation=make_standard_evaluator(recipe, policy))
    @jax.jit
    def diagnostic(estimator, moments):
        roots = env.reset(jax.random.split(jax.random.PRNGKey(93001), recipe.num_envs))
        params = estimator.params if recipe.method == 'gce_vector' else estimator[0].params
        batch, health, _ = functions['probe'](params, roots, jax.random.PRNGKey(93101))
        if recipe.method == 'gce_vector':
            observations, targets = standardize(moments, batch.observations, batch.targets)
            total, components = vector_losses(
                model, params, observations, batch.actions, targets,
                jnp.ones(recipe.num_envs, jnp.float32))
            # Raw-unit fit on the fixed probe bank, beside two trivial predictors.
            predicted = (model.apply({'params': params}, observations, batch.actions,
                                     method=model.predict_target)
                         * moments.target_std + moments.target_mean)
            permutation = jax.random.permutation(jax.random.PRNGKey(93201), recipe.num_envs)
            shuffled_prediction = (model.apply(
                {'params': params}, observations, batch.actions[permutation],
                method=model.predict_target) * moments.target_std + moments.target_mean)
            start = (batch.observations if recipe.outcome_indices is None
                     else batch.observations[:, jnp.asarray(recipe.outcome_indices)])
            channel = jnp.mean(jnp.square(predicted - batch.targets))
            mean_predictor = jnp.mean(jnp.square(batch.targets - batch.targets.mean(0)))
            values = dict(channel_mse=channel, latent_prediction_mse=components[0],
                          vector_total_loss=total,
                          shuffled_action_channel_mse=jnp.mean(jnp.square(
                              shuffled_prediction - batch.targets)),
                          mean_predictor_mse=mean_predictor,
                          no_change_predictor_mse=jnp.mean(jnp.square(batch.targets - start)),
                          channel_variance_explained=1. - channel / jnp.maximum(mean_predictor, 1e-12))
        else:
            values = vim_diagnostics(source, params, decoder, estimator[1].params,
                *batch, jnp.ones(recipe.num_envs), beta=1.)
        return {**values, **health}
    functions['diagnostic'] = diagnostic
    functions['environment_config'] = asdict(env.config)
    functions['coverage_specs'] = exploration_specs(env, state['controller'].env_state)
    functions['coverage_masks'] = coverage_floor_masks(env, functions['coverage_specs'])
    return state, functions


def parameters(recipe, state):
    return state['estimator'].params if recipe.method == 'gce_vector' else state['estimator'][0].params


def update(recipe, state, functions):
    rng = state['rng']
    frozen_controller = state['controller']
    state['controller'], roots, rollout, rng['controller'], exploration = functions['collect'](
        state['ppo'], state['controller'], rng['controller'])
    batch, collection, rng['probes'] = functions['probe'](parameters(recipe, state), roots, rng['probes'])
    if not bool(collection['probe_all_finite']):
        raise FloatingPointError('nonfinite counterfactual state; no tuple masking or silent reset applied')
    rng['retention'], key = jax.random.split(rng['retention'])
    state['buffer'] = insert_probe_batch(state['buffer'], batch, key)
    replay = state['buffer']
    if recipe.vector_gce_normalize_inputs:
        # Moments are frozen for this update's replay passes and reward query.
        state['estimator_norm'] = moments = replay_moments(replay)
        observations, targets = standardize(moments, replay.observations, replay.targets)
        replay = replay._replace(observations=observations, targets=targets)
    state['estimator'], metrics, rng['training'] = train_replay(
        functions['epoch'], state['estimator'], replay, rng['training'], recipe.estimator.replay_passes)
    query = rollout.observations if recipe.method == 'gce_vector' else rollout.terminal_observations
    query = query.reshape((-1, query.shape[-1]))
    if recipe.vector_gce_normalize_inputs:
        query = standardize(state['estimator_norm'], query)
    estimates = functions['estimate'](parameters(recipe, state), query)
    exponentiate = recipe.method == 'gce_vector' and recipe.exponentiate_empowerment
    rewards = (jnp.exp(estimates) if exponentiate else estimates).reshape(rollout.dones.shape)
    if not np.isfinite(np.asarray(rewards)).all():
        raise FloatingPointError('nonfinite baseline reward')
    learning_rate = controller_learning_rate(recipe.ppo, state['ppo'].step,
        recipe.updates*recipe.ppo.update_epochs*recipe.ppo.num_minibatches)
    state['ppo'], ppo_metrics, _, _, rng['ppo'] = functions['ppo_update'](
        state['ppo'], rollout, rewards, frozen_controller, rng['ppo'])
    state['update'] += 1
    count = int(state['update'])
    row = dict(env_steps=count*recipe.steps_per_update, update=count,
               controller_steps=count*recipe.num_envs*recipe.rollout_steps,
               probe_steps=count*recipe.num_envs*recipe.horizon,
               replay_count=state['buffer'].count, reward_mean=rewards.mean(),
               estimate_mean=estimates.mean(), estimate_std=estimates.std(), **collection)
    row.update({f'estimator/{k}':v for k,v in metrics.items()})
    row.update({f'ppo/{k}':v for k,v in ppo_metrics.items()})
    if recipe.ppo.normalize_actor_observations or recipe.ppo.anneal_learning_rate:
        row.update({'ppo/learning_rate': learning_rate, 'ppo/reward_std': rewards.std(),
                    'ppo/observation_rms': jnp.sqrt(jnp.mean(frozen_controller.observation_norm.variance))})
    # Standalone policies have no potential/skill split; publish only union keys.
    counts = coverage_counts(state['controller'].coverage_grids, functions['coverage_specs'],
                             functions['coverage_masks'])
    row.update({k:v for k,v in counts.items() if not k.endswith(('_potential_phase', '_skill_phase'))})
    completed = int(exploration['completed_episodes'])
    row['exploration/completed_episodes'] = completed
    if completed:
        row.update({f'exploration/{k}':v/completed for k,v in exploration.items()
                    if k != 'completed_episodes'})
    return {key:float(value) for key,value in row.items()}


def append_csv(path, row, *, optional_fields=()):
    exists = path.exists()
    with path.open('a', newline='') as handle:
        fields = [key for key in row if key not in optional_fields] + list(optional_fields)
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate(recipe, state, functions, output, *, source_run=None, target_step=None, mirror=None):
    step = int(state['update'])*recipe.steps_per_update
    target_step = step if target_step is None else target_step
    print(f'evaluating target={target_step} checkpoint={step}', flush=True)
    row = dict(env_steps=step, target_step=target_step)
    moments = state.get('estimator_norm', identity_moments(recipe.model_environment))
    row.update({f'heldout/{k}':float(v) for k,v in functions['diagnostic'](state['estimator'], moments).items()})
    reset_key, action_key, _ = evaluation_keys(recipe.environment, recipe.seed, target_step)
    arrays = jax.device_get(functions['standard_evaluation'](
        (state['ppo'].params, state['controller'].observation_norm), reset_key, action_key,
        jnp.zeros((128, 0), jnp.float32)))
    payload = save_evaluation(output / 'evaluations' / f'step_{target_step:012d}', arrays,
        environment=recipe.environment, seed=recipe.seed, target_step=target_step,
        checkpoint_step=step, reset_preset=recipe.reset_preset, source_run=source_run or output,
        skill_phase=False)
    metrics = payload['potential_policy']['metrics']
    row.update({f'potential/{name}':float(value) for name,value in metrics.items()})
    distance_names = {
        'agent_to_center_xy/distance_mean': 'agent_center_distance_xy_mean',
        'agent_to_ball_xy/distance_mean': 'agent_ball_distance_xy_mean',
        'gripper_to_nearest_block_xy/distance_mean': 'gripper_nearest_block_distance_xy_mean',
    }
    spatial = {name:float(metrics[key]) for key,name in distance_names.items() if key in metrics}
    endpoint_name = {'ant': 'agent_center_distance_xy_mean',
                     'ant_soccer': 'agent_ball_distance_xy_mean',
                     'builderbench_corner3': 'gripper_nearest_block_distance_xy_mean'}[recipe.environment]
    spatial.update(potential_endpoint_distance_xy_mean=spatial[endpoint_name],
                   potential_endpoint_horizon_actions=100,
                   potential_endpoint_early_termination_count=int((np.asarray(arrays['alive']).sum(0) < 100).sum()))
    row.update({f'stochastic/spatial/{name}':value for name,value in spatial.items()})
    row.update({f'potential/{name}':value for name,value in spatial.items()})
    row.update({f'stochastic/{name}_mean':float(values['mean'])
                for name,values in payload['summary'].items()})
    row['evaluation_worlds'] = 128
    row['evaluation_horizon_actions'] = 100
    row['evaluation_transitions'] = 128*100+recipe.num_envs*recipe.horizon
    if not all(np.isfinite(value) for value in row.values()) or not row['heldout/probe_all_finite']:
        raise FloatingPointError('nonfinite evaluation or held-out probe')
    append_csv(output / 'eval.csv', row)
    if mirror is not None:
        media = payload['media']
        mirror.log_video(row, key='eval/media/potential_policy_video',
                         path=media['video'], fps=media['fps'])
    return row


def evaluation_checkpoint(recipe, state):
    """Keep actor moments and estimator parameters without replay or optimizers."""
    estimator = (state['estimator'].params if recipe.method == 'gce_vector'
                 else tuple(component.params for component in state['estimator']))
    checkpoint = dict(ppo_params=state['ppo'].params,
                      observation_norm=state['controller'].observation_norm,
                      estimator_params=estimator,
                      env_steps=jnp.asarray(int(state['update'])*recipe.steps_per_update))
    if 'estimator_norm' in state:
        checkpoint['estimator_norm'] = state['estimator_norm']
    return checkpoint


def evaluation_targets(recipe, previous_step, step, *, final=False):
    """Evaluate each charged-transition boundary and the last available snapshot."""
    cadence = recipe.eval_every_env_steps
    targets = list(range((previous_step//cadence+1)*cadence, step+1, cadence))
    if final:
        if step not in targets:
            targets.append(step)
    return targets


def behavior_config(config):
    result = {k:v for k,v in config.items() if k not in
              ('total_env_steps','run_dir','resume','max_updates','config_only','effective_scale')}
    result.setdefault('vector_gce_bound_action_decoder', False)
    result.setdefault('exponentiate_empowerment', True)
    return result


def run(recipe):
    if jax.devices()[0].platform != 'gpu':
        raise RuntimeError('Warp baseline training requires a GPU')
    output = Path(recipe.run_dir).resolve()
    resolved = recipe.resolved()
    if output.exists() and any(p.name not in {"recipe.json", "slurm_metadata.txt", "source_snapshot.json"} for p in output.iterdir()) and not recipe.resume:
        raise FileExistsError('use a fresh run directory or explicit --resume')
    output.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).parents[2]
    source_files = sorted(Path(__file__).parent.glob('*.py'))
    source_files += [package / name for name in (
        'agents/latent_gce.py','agents/vim.py','agents/ppo.py','utils/networks.py',
        'evaluation_protocol.py','envs/factory.py','agents/see_sac.py','training/sac/config.py','checkpointing.py','envs/state.py','core/rollout.py',
        'envs/builderbench/env.py','envs/builderbench/config.py','envs/builderbench/model.py',
        'envs/locomotion/env.py','envs/locomotion/config.py','envs/locomotion/model.py')]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    previous_sources = None
    if recipe.resume:
        previous = json.loads((output / 'resolved_config.json').read_text())
        if recipe.ppo.anneal_learning_rate and previous['total_env_steps'] != recipe.total_env_steps:
            raise ValueError('resume changes the full PPO learning-rate annealing budget')
        if behavior_config(previous) != behavior_config(resolved):
            raise ValueError('resume changes the saved scientific recipe')
        previous_sources = json.loads((output / 'run_manifest.json').read_text())['sources']
        if previous_sources != hashes:
            with (output / 'resume_source_changes.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(previous_sources=previous_sources,
                                             resumed_sources=hashes)) + '\n')
    print(f'initializing {recipe.environment}/{recipe.method}, horizon={recipe.horizon}', flush=True)
    state, functions = build(recipe)
    print('initialized networks, replay and Warp state', flush=True)
    store = CheckpointStore(output / 'checkpoints', keep=recipe.checkpoint_keep)
    evaluation_store = CheckpointStore(output / 'evaluation_checkpoints')
    if recipe.resume:
        state = store.restore(state)
    (output / 'resolved_config.json').write_text(json.dumps(resolved, indent=2)+'\n')
    # Existing Warp offline sync reads this filename and the JSONL below.
    (output / 'config.json').write_text(json.dumps({**resolved, 'algorithm':recipe.method}, indent=2)+'\n')
    manifest = dict(hostname=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    device=jax.devices()[0].device_kind, evidence='smoke' if recipe.max_updates else 'benchmark',
                    environment_config=functions['environment_config'], sources=hashes,
                    versions={name:version(name) for name in ('jax','flax','optax','distrax','mujoco','mujoco-mjx')})
    if previous_sources is not None:
        manifest['checkpoint_sources'] = previous_sources
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    mirror = WandbMirror.start(output, {**resolved, 'algorithm':recipe.method}, resume=recipe.resume)
    stop = (min(recipe.updates, int(state['update']) + recipe.max_updates)
            if recipe.max_updates else recipe.updates)
    completed_targets = set()
    if (output / 'eval.csv').exists():
        with (output / 'eval.csv').open(newline='') as handle:
            completed_targets = {int(row.get('target_step', row['env_steps'])) for row in csv.DictReader(handle)}

    def checkpoint_and_evaluate(targets):
        targets = [target for target in targets if target not in completed_targets]
        if not targets:
            return {}
        step = int(state['update'])*recipe.steps_per_update
        metadata = {**resolved, 'evaluation_target_steps': targets}
        # Save recovery state before rollout/rendering, and retain compact actors separately.
        store.save(step, state, metadata)
        evaluation_store.save(step, evaluation_checkpoint(recipe, state), metadata, payload_kind='model')
        result = {}
        for target in targets:
            result = evaluate(recipe, state, functions, output, target_step=target, mirror=mirror)
            completed_targets.add(target)
        return result

    if not recipe.resume:
        initial = checkpoint_and_evaluate([0])
        initial['segment'] = 0
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(initial)+'\n')
        mirror.log(initial)
    else:
        # A failed evaluation resumes its saved actor before collecting another update.
        pending = store.latest_manifest()['config'].get('evaluation_target_steps', [])
        recovered = checkpoint_and_evaluate(pending)
        if recovered:
            recovered['segment'] = int(state['update'])
            with (output / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(recovered)+'\n')
            mirror.log(recovered)
    for _ in range(int(state['update']), stop):
        start = time.monotonic()
        row = update(recipe, state, functions)
        row['time/update_seconds'] = time.monotonic()-start
        if not all(np.isfinite(value) for value in row.values()):
            raise FloatingPointError('nonfinite learner metric')
        append_csv(output / 'train.csv', row, optional_fields=EXPLORATION_DISPLACEMENT_FIELDS)
        count = int(state['update'])
        targets = evaluation_targets(recipe, (count-1)*recipe.steps_per_update,
                                     count*recipe.steps_per_update, final=count == stop)
        if targets:
            row.update(checkpoint_and_evaluate(targets))
        row['segment'] = count
        with (output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        mirror.log(row)
        print(json.dumps(row), flush=True)
    summary = dict(update=int(state['update']), env_steps=int(state['update'])*recipe.steps_per_update,
                   requested_env_steps=recipe.total_env_steps,
                   evidence='smoke' if recipe.max_updates else 'single_seed_benchmark')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    mirror.finish()
