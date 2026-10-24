"""Shared fixed-bank diagnostics and policy evaluation; no training RNG is consumed."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from empowerment_warp.baselines.toy_baselines.behavior import behavior_metrics
from empowerment_warp.baselines.toy_baselines.envs import (
    observation_from_physical, physical_from_observation, reset_one, state_grid, step_one)
from empowerment_warp.baselines.toy_baselines.gce import capacity_from_observations, prediction_errors
from empowerment_warp.baselines.toy_baselines.ppo import normalize_observations
from empowerment_warp.baselines.toy_baselines.probes import draw_source_batch, rollout_actions, uniform_actions
from empowerment_warp.baselines.toy_baselines.vim import empowerment_from_observations, rollout_successor_observations


class EpisodeTrace(NamedTuple):
    observations: jax.Array
    physical_states: jax.Array
    actions: jax.Array
    estimates: jax.Array
    rewards: jax.Array


def estimate_key(config):
    return 'capacity_bits' if config.method == 'gce' else 'empowerment_nats'


def make_estimate(config, model):
    """Keep native units: GCE channel bits and VIM scalar nats are distinct."""
    @jax.jit
    def estimate(params, observations):
        if config.method == 'gce':
            return capacity_from_observations(
                model, params, observations, power=config.gce.channel_power,
                tolerance=config.gce.water_filling_tolerance,
                max_iterations=config.gce.water_filling_max_iterations)[0]
        return empowerment_from_observations(model, params, observations, beta=config.vim.beta)
    return estimate


def make_controller_rewards(config, estimate):
    """Preserve GCE current-state exp(bits), VIM reached-state raw nats."""
    @jax.jit
    def rewards(params, rollout, controller):
        observations = (rollout.observations if config.method == 'gce'
                        else rollout_successor_observations(rollout, controller))
        values = estimate(params, observations.reshape((-1, config.environment.observation_dim)))
        values = values.reshape(rollout.dones.shape)
        return (jnp.exp(values) if config.method == 'gce' else values), values
    return rewards


def _policy_observation(config, norm, observations):
    return (normalize_observations(norm, observations)
            if config.ppo.normalize_actor_observations else observations)


def make_episode_rollout(config, policy, estimate, *, deterministic):
    @jax.jit
    def rollout(policy_params, estimator_params, norm, reset_key, action_key):
        observation, state = reset_one(config.environment, reset_key)

        def step(carry, _):
            state, obs, key = carry
            distribution, _ = policy.apply(
                {'params': policy_params}, _policy_observation(config, norm, obs[None]))
            key, sample_key, reset_rng = jax.random.split(key, 3)
            action = distribution.mean()[0] if deterministic else distribution.sample(seed=sample_key)[0]
            next_obs, state, _, terminal = step_one(config.environment, state, action, reset_rng)
            query = obs if config.method == 'gce' else terminal
            value = estimate(estimator_params, query[None])[0]
            reward = jnp.exp(value) if config.method == 'gce' else value
            row = (obs, physical_from_observation(config.environment.name, terminal), action, value, reward)
            return (state, next_obs, key), row

        _, (observations, physical, actions, values, rewards) = jax.lax.scan(
            step, (state, observation, action_key), None, length=config.environment.episode_steps)
        initial = physical_from_observation(config.environment.name, observation)[None]
        return EpisodeTrace(observations, jnp.concatenate((initial, physical)), actions, values, rewards)
    return rollout


def evaluate_fixed_bank(config, policy_params, estimator_params, norm, rollouts):
    metrics, first_traces, arrays = {}, {}, {}
    for mode, reset_seed, action_seed, count in (
        ('deterministic', 91001, 91101, config.evaluation.deterministic_episodes),
        ('stochastic', 92001, 92101, config.evaluation.stochastic_episodes),
    ):
        traces = [rollouts[mode](
            policy_params, estimator_params, norm,
            jax.random.fold_in(jax.random.PRNGKey(reset_seed), i),
            jax.random.fold_in(jax.random.PRNGKey(action_seed), i)) for i in range(count)]
        physical = jnp.stack([t.physical_states for t in traces])
        rewards = jnp.stack([t.rewards for t in traces]).sum(axis=1)
        values = jnp.stack([t.estimates for t in traces])
        prefix = f'evaluation/{mode}'
        metrics.update({
            f'{prefix}/episode_reward_mean': float(rewards.mean()),
            f'{prefix}/episode_reward_std': float(rewards.std()),
            f'{prefix}/{estimate_key(config)}_mean': float(values.mean()),
            **{f'{prefix}/{key}': value for key, value in
               behavior_metrics(config.environment.name, physical).items()},
        })
        first_traces[mode] = traces[0]
        arrays[f'{mode}_physical_states'] = physical
        arrays[f'{mode}_actions'] = jnp.stack([t.actions for t in traces])
    steps = (config.evaluation.deterministic_episodes + config.evaluation.stochastic_episodes
             ) * config.environment.episode_steps
    return metrics, first_traces, arrays, steps


def anchor_states(environment):
    if environment.name == 'ball_in_box':
        names = ('center', 'left', 'right', 'bottom', 'top')
        physical = jnp.array([[0., 0.], [-8.5, 0.], [8.5, 0.], [0., -8.5], [0., 8.5]])
    else:
        names = ('upright', 'down', 'left', 'right', 'upright_minus4', 'upright_plus4', 'down_minus4', 'down_plus4')
        physical = jnp.array([[0., 0.], [jnp.pi, 0.], [-jnp.pi/2, 0.], [jnp.pi/2, 0.],
                              [0., -4.], [0., 4.], [jnp.pi, -4.], [jnp.pi, 4.]])
    return names, observation_from_physical(environment.name, physical)


def make_anchor_diagnostic(config, model, estimate, decoder=None):
    names, anchors = anchor_states(config.environment)
    count = config.evaluation.samples_per_anchor
    starts = jnp.repeat(anchors, count, axis=0)

    @jax.jit
    def diagnostic(params, decoder_params, key):
        values = estimate(params, anchors)
        if config.method == 'gce':
            actions = uniform_actions(key, (len(starts), config.environment.horizon, config.environment.action_dim))
            targets = rollout_actions(config.environment, starts, actions)[:, -1]
            errors = prediction_errors(model, params, starts, actions, targets).reshape((len(names), count))
            return values, dict(endpoint_rmse=jnp.sqrt(errors.mean(axis=1)))
        batch, log_h = draw_source_batch(config.environment, model, params, starts, key)
        log_q = decoder.apply({'params': decoder_params}, batch.observations, batch.targets,
                             batch.actions, batch.pre_tanh_actions)
        bounds = (log_q - log_h).reshape((len(names), count))
        bound = bounds.mean(axis=1)
        return values, dict(
            fresh_bound_nats=bound, fresh_bound_standard_error_nats=bounds.std(axis=1, ddof=1) / jnp.sqrt(count),
            scalar_minus_fresh_bound_nats=values-bound,
            source_entropy_nats=-log_h.reshape((len(names), count)).mean(axis=1),
            decoder_nll=-log_q.reshape((len(names), count)).mean(axis=1))

    def evaluate(params, decoder_params, key):
        values, diagnostics = diagnostic(params, decoder_params, key)
        metrics = {}
        for i, name in enumerate(names):
            metrics[f'evaluation/anchors/{name}/{estimate_key(config)}'] = float(values[i])
            metrics.update({f'evaluation/anchors/{name}/{k}': float(v[i]) for k, v in diagnostics.items()})
        return metrics, len(starts) * config.environment.horizon
    return evaluate


def evaluate_fields(config, estimate, params, visitation, policy=None, policy_params=None, norm=None):
    x, y, observations = state_grid(config.environment, config.evaluation.grid_bins)
    fields = dict(x=x, y=y, observations=observations, estimate=estimate(params, observations),
                  visitation_observations=visitation.reshape((-1, config.environment.observation_dim)))
    if policy is not None:
        distribution, _ = policy.apply({'params': policy_params}, _policy_observation(config, norm, observations))
        fields['policy_actions'] = jnp.clip(distribution.mean(), -config.environment.action_limit,
                                           config.environment.action_limit)
    return fields
