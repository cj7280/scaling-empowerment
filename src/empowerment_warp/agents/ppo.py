# Extracted from empowerment; see training/baselines/reference_manifest.json and LICENSE.
"""Tanh-squashed PPO controller shared by the VIM and LatentGCE baselines.

The actor/critic network is ``ActorCritic`` in ``utils/networks.py``.
"""
from __future__ import annotations

from functools import partial

from typing import NamedTuple

import jax

import jax.numpy as jnp

import optax

from flax.training.train_state import TrainState

from empowerment_warp.training.baselines.config import EnvironmentConfig, PPOConfig

from empowerment_warp.utils.networks import ActorCritic

class ObservationNormState(NamedTuple):
    """Running moments used only by the optional actor/critic observation path."""

    mean: jax.Array
    variance: jax.Array
    count: jax.Array

def controller_learning_rate(config, step, total_steps):
    fraction = jnp.maximum(0., 1. - step / max(1, total_steps))
    return config.learning_rate * fraction if getattr(config, 'anneal_learning_rate', False) else config.learning_rate

class RolloutBatch(NamedTuple):
    """One on-policy rollout before the GCE reward is attached."""

    observations: jax.Array
    policy_observations: jax.Array
    pre_tanh_actions: jax.Array
    log_probabilities: jax.Array
    values: jax.Array
    dones: jax.Array
    terminal_observations: jax.Array
    terminated: jax.Array

class ControllerState(NamedTuple):
    """Environment and normalization state carried between PPO updates."""

    env_state: object
    observation: jax.Array
    observation_norm: ObservationNormState
    coverage_grids: jax.Array | None = None
    episode_start_positions: tuple[jax.Array, ...] = ()


def tanh_log_det(pre_tanh: jax.Array) -> jax.Array:
    """Log absolute Jacobian determinant, stable even when tanh rounds to +/-1."""
    return jnp.sum(2.0 * (jnp.log(2.0) - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh)), axis=-1)


def squashed_log_prob(distribution, pre_tanh: jax.Array) -> jax.Array:
    return distribution.log_prob(pre_tanh) - tanh_log_det(pre_tanh)


def squashed_entropy(distribution, key: jax.Array) -> jax.Array:
    """H[tanh(U)] = H[U] + E log|J|; one reparameterized sample per state."""
    return distribution.entropy() + tanh_log_det(distribution.sample(seed=key))

def create_ppo_train_state(
    key: jax.Array,
    environment: EnvironmentConfig,
    config: PPOConfig,
    *,
    total_updates: int | None = None,
) -> tuple[ActorCritic, TrainState]:
    """Initialize the actor/critic and clipped Adam optimizer."""

    network = ActorCritic(
        action_dim=environment.action_dim,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
    )
    variables = network.init(key, jnp.zeros((1, environment.observation_dim), dtype=jnp.float32))
    learning_rate = config.learning_rate
    if getattr(config, 'anneal_learning_rate', False):
        if total_updates is None or total_updates < 1:
            raise ValueError('annealed PPO requires the full controller update budget')
        total_steps = total_updates * config.update_epochs * config.num_minibatches
        learning_rate = lambda step: controller_learning_rate(config, step, total_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate, eps=config.adam_epsilon),
    )
    return network, TrainState.create(apply_fn=network.apply, params=variables['params'], tx=optimizer)

def empty_observation_norm(observation_dim: int) -> ObservationNormState:
    """Return numerically stable empty running moments."""

    return ObservationNormState(
        mean=jnp.zeros((observation_dim,), dtype=jnp.float32),
        variance=jnp.ones((observation_dim,), dtype=jnp.float32),
        count=jnp.asarray(1e-4, dtype=jnp.float32),
    )

def _update_observation_norm(state: ObservationNormState, observations: jax.Array) -> ObservationNormState:
    batch_mean = jnp.mean(observations, axis=0)
    batch_variance = jnp.var(observations, axis=0)
    batch_count = observations.shape[0]
    delta = batch_mean - state.mean
    total_count = state.count + batch_count
    mean = state.mean + delta * batch_count / total_count
    m_a = state.variance * state.count
    m_b = batch_variance * batch_count
    m2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / total_count
    return ObservationNormState(mean=mean, variance=m2 / total_count, count=total_count)

def normalize_observations(state: ObservationNormState, observations: jax.Array,
                           *, epsilon=1e-8, clip=10.) -> jax.Array:
    return jnp.clip((observations - state.mean) / jnp.sqrt(state.variance + epsilon), -clip, clip)

@partial(jax.jit, static_argnames=('gamma', 'gae_lambda'))
def calculate_gae(
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminated: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Bootstrap timeouts from pre-reset values, but never trace across a reset."""

    def reverse_step(gae, transition):
        reward, done, value, next_value, terminal = transition
        delta = reward + gamma * jnp.where(terminal, 0.0, next_value) - value
        gae = delta + gamma * gae_lambda * jnp.where(done, 0.0, gae)
        return gae, gae

    _, advantages = jax.lax.scan(
        reverse_step,
        jnp.zeros_like(values[0]),
        (rewards, dones, values, next_values, terminated),
        reverse=True,
    )
    return advantages, advantages + values

def make_ppo_update(network: ActorCritic, environment: EnvironmentConfig, config: PPOConfig):
    """Build Zhao-configured clipped PPO updates around the PureJaxRL network."""

    batch_size = config.num_envs * config.rollout_steps
    minibatch_size = batch_size // config.num_minibatches

    @jax.jit
    def update(
        train_state: TrainState,
        batch: RolloutBatch,
        rewards: jax.Array,
        controller_state: ControllerState,
        key: jax.Array,
    ) -> tuple[TrainState, dict[str, jax.Array], jax.Array, jax.Array, jax.Array]:
        # Genuine terminations can contain nonfinite simulator observations.
        next_observations = jnp.where(batch.terminated[..., None], 0.0,
                                      batch.terminal_observations)
        if config.normalize_actor_observations:
            next_observations = normalize_observations(
                controller_state.observation_norm,
                next_observations,
                epsilon=getattr(config, 'normalization_epsilon', 1e-8),
                clip=getattr(config, 'normalization_clip', 10.),
            )
        _, next_values = network.apply({'params': train_state.params}, next_observations)
        advantages, returns = calculate_gae(
            rewards,
            batch.dones,
            batch.values,
            next_values,
            batch.terminated,
            gamma=environment.gamma,
            gae_lambda=config.gae_lambda,
        )
        flat_batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
        flat_advantages = advantages.reshape((batch_size,))
        flat_returns = returns.reshape((batch_size,))

        def update_epoch(carry, _):
            state, rng = carry
            rng, permutation_key = jax.random.split(rng)
            permutation = jax.random.permutation(permutation_key, batch_size)
            shuffled_batch = jax.tree_util.tree_map(lambda x: x[permutation], flat_batch)
            shuffled_advantages = flat_advantages[permutation]
            shuffled_returns = flat_returns[permutation]
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((config.num_minibatches, minibatch_size) + x.shape[1:]),
                (shuffled_batch, shuffled_advantages, shuffled_returns),
            )

            def update_minibatch(carry, minibatch):
                current_state, rng = carry
                rng, entropy_key = jax.random.split(rng)
                transition, minibatch_advantages, minibatch_returns = minibatch

                def loss_fn(params):
                    distribution, value = network.apply({'params': params}, transition.policy_observations)
                    log_probability = squashed_log_prob(distribution, transition.pre_tanh_actions)
                    ratio = jnp.exp(log_probability - transition.log_probabilities)
                    normalized_advantages = (
                        minibatch_advantages - jnp.mean(minibatch_advantages)
                    ) / (jnp.std(minibatch_advantages) + 1e-8)
                    actor_loss = -jnp.mean(
                        jnp.minimum(
                            ratio * normalized_advantages,
                            jnp.clip(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
                            * normalized_advantages,
                        )
                    )
                    clipped_value = transition.values + jnp.clip(
                        value - transition.values,
                        -config.clip_epsilon,
                        config.clip_epsilon,
                    )
                    value_loss = 0.5 * jnp.mean(
                        jnp.maximum(
                            jnp.square(value - minibatch_returns),
                            jnp.square(clipped_value - minibatch_returns),
                        )
                    )
                    entropy = jnp.mean(squashed_entropy(distribution, entropy_key))
                    total_loss = (
                        actor_loss
                        + config.value_coefficient * value_loss
                        - config.entropy_coefficient * entropy
                    )
                    approx_kl = jnp.mean((ratio - 1.0) - jnp.log(ratio))
                    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > config.clip_epsilon).astype(jnp.float32))
                    return total_loss, (actor_loss, value_loss, entropy, approx_kl, clip_fraction)

                (total_loss, auxiliary), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_state.params)
                grad_norm = optax.global_norm(grads)
                current_state = current_state.apply_gradients(grads=grads)
                return (current_state, rng), jnp.asarray((*auxiliary, total_loss, grad_norm))

            (state, rng), metrics = jax.lax.scan(update_minibatch, (state, rng), minibatches)
            return (state, rng), metrics

        (train_state, key), epoch_metrics = jax.lax.scan(
            update_epoch,
            (train_state, key),
            None,
            length=config.update_epochs,
        )
        means = jnp.mean(epoch_metrics, axis=(0, 1))
        metrics = {
            'actor_loss': means[0],
            'value_loss': means[1],
            'entropy': means[2],
            'approx_kl': means[3],
            'clip_fraction': means[4],
            'total_loss': means[5],
            'gradient_norm_before_clip': means[6],
            'advantage_mean': jnp.mean(advantages),
            'advantage_std': jnp.std(advantages),
            'value_mean': jnp.mean(batch.values),
            'action_saturation_fraction': jnp.mean(
                (jnp.abs(jnp.tanh(batch.pre_tanh_actions)) >= 0.99).astype(jnp.float32)),
        }
        return train_state, metrics, advantages, returns, key

    return update

@partial(jax.jit, static_argnames=('network', 'deterministic'))
def policy_action(
    network: ActorCritic,
    params,
    observations: jax.Array,
    key: jax.Array,
    *,
    deterministic: bool,
) -> tuple[jax.Array, jax.Array]:
    """Return tanh(mean) or sampled bounded actions, plus sampled squashed entropy."""

    distribution, _ = network.apply({'params': params}, observations)
    action_key, entropy_key = jax.random.split(key)
    pre_tanh = distribution.mean() if deterministic else distribution.sample(seed=action_key)
    return jnp.tanh(pre_tanh), squashed_entropy(distribution, entropy_key)
