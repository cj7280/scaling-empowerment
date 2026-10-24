"""Continuous-action PPO adapted from PureJaxRL with Zhao's PPO2 recipe."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from empowerment_warp.baselines.toy_baselines.config import EnvironmentConfig, PPOConfig
from empowerment_warp.baselines.toy_baselines.envs import EnvState, reset_batch, step_batch


class ObservationNormState(NamedTuple):
    """Running moments used only by the optional actor/critic observation path."""

    mean: jax.Array
    variance: jax.Array
    count: jax.Array


class RolloutBatch(NamedTuple):
    """One on-policy rollout before the GCE reward is attached."""

    observations: jax.Array
    policy_observations: jax.Array
    actions: jax.Array
    log_probabilities: jax.Array
    values: jax.Array
    dones: jax.Array
    terminal_observations: jax.Array


class ControllerState(NamedTuple):
    """Environment and normalization state carried between PPO updates."""

    env_state: EnvState
    observation: jax.Array
    observation_norm: ObservationNormState


class ActorCritic(nn.Module):
    """PureJaxRL's separate 2x256 tanh Gaussian actor and critic."""

    action_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 2

    @nn.compact
    def __call__(self, observations: jax.Array):
        actor = observations
        for _ in range(self.hidden_layers):
            actor = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2.0)),
                    bias_init=constant(0.0),
                )(actor)
            )
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        log_std = self.param('log_std', nn.initializers.zeros_init(), (self.action_dim,))

        critic = observations
        for _ in range(self.hidden_layers):
            critic = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2.0)),
                    bias_init=constant(0.0),
                )(critic)
            )
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)
        return distrax.MultivariateNormalDiag(actor_mean, jnp.exp(log_std)), jnp.squeeze(value, axis=-1)


def create_ppo_train_state(
    key: jax.Array,
    environment: EnvironmentConfig,
    config: PPOConfig,
) -> tuple[ActorCritic, TrainState]:
    """Initialize the actor/critic and clipped Adam optimizer."""

    network = ActorCritic(
        action_dim=environment.action_dim,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
    )
    variables = network.init(key, jnp.zeros((1, environment.observation_dim), dtype=jnp.float32))
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=config.adam_epsilon),
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


def normalize_observations(state: ObservationNormState, observations: jax.Array) -> jax.Array:
    return jnp.clip((observations - state.mean) / jnp.sqrt(state.variance + 1e-8), -10.0, 10.0)


def initialize_controller(
    key: jax.Array,
    environment: EnvironmentConfig,
    config: PPOConfig,
) -> tuple[ControllerState, jax.Array]:
    """Reset the exact number of parallel controller environments."""

    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, config.num_envs)
    observations, env_state = reset_batch(environment, reset_keys)
    return ControllerState(
        env_state=env_state,
        observation=observations,
        observation_norm=empty_observation_norm(environment.observation_dim),
    ), key


def make_collect_rollout(
    network: ActorCritic,
    environment: EnvironmentConfig,
    config: PPOConfig,
):
    """Build the compiled on-policy rollout collector."""

    @jax.jit
    def collect(
        train_state: TrainState,
        controller_state: ControllerState,
        key: jax.Array,
    ) -> tuple[ControllerState, RolloutBatch, jax.Array]:
        def step(carry, _):
            state, rng = carry
            norm_state = state.observation_norm
            if config.normalize_actor_observations:
                norm_state = _update_observation_norm(norm_state, state.observation)
                policy_observation = normalize_observations(norm_state, state.observation)
            else:
                policy_observation = state.observation
            distribution, value = network.apply({'params': train_state.params}, policy_observation)
            rng, action_key, reset_key = jax.random.split(rng, 3)
            action = distribution.sample(seed=action_key)
            log_probability = distribution.log_prob(action)
            reset_keys = jax.random.split(reset_key, config.num_envs)
            next_observation, next_env_state, done, terminal_observation = step_batch(
                environment,
                state.env_state,
                action,
                reset_keys,
            )
            transition = RolloutBatch(
                observations=state.observation,
                policy_observations=policy_observation,
                actions=action,
                log_probabilities=log_probability,
                values=value,
                dones=done,
                terminal_observations=terminal_observation,
            )
            next_state = ControllerState(
                env_state=next_env_state,
                observation=next_observation,
                observation_norm=norm_state,
            )
            return (next_state, rng), transition

        (controller_state, key), batch = jax.lax.scan(
            step,
            (controller_state, key),
            None,
            length=config.rollout_steps,
        )
        return controller_state, batch, key

    return collect


@partial(jax.jit, static_argnames=('gamma', 'gae_lambda'))
def calculate_gae(
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    last_value: jax.Array,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Calculate GAE with TimeLimit boundaries aligned to the acted transition."""

    def reverse_step(carry, transition):
        gae, next_value = carry
        reward, done, value = transition
        nonterminal = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_value * nonterminal - value
        gae = delta + gamma * gae_lambda * nonterminal * gae
        return (gae, value), gae

    (_, _), advantages = jax.lax.scan(
        reverse_step,
        (jnp.zeros_like(last_value), last_value),
        (rewards, dones, values),
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
        if config.normalize_actor_observations:
            last_policy_observation = normalize_observations(
                controller_state.observation_norm,
                controller_state.observation,
            )
        else:
            last_policy_observation = controller_state.observation
        _, last_value = network.apply({'params': train_state.params}, last_policy_observation)
        advantages, returns = calculate_gae(
            rewards,
            batch.dones,
            batch.values,
            last_value,
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

            def update_minibatch(current_state, minibatch):
                transition, minibatch_advantages, minibatch_returns = minibatch

                def loss_fn(params):
                    distribution, value = network.apply({'params': params}, transition.policy_observations)
                    log_probability = distribution.log_prob(transition.actions)
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
                    entropy = jnp.mean(distribution.entropy())
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
                return current_state, jnp.asarray((*auxiliary, total_loss, grad_norm))

            state, metrics = jax.lax.scan(update_minibatch, state, minibatches)
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
    """Return Gaussian-mean or sampled actions for evaluation."""

    distribution, _ = network.apply({'params': params}, observations)
    actions = distribution.mean() if deterministic else distribution.sample(seed=key)
    return actions, distribution.entropy()
