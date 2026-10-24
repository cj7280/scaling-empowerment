"""Pure-JAX versions of Zhao's Ball-in-Box and state Pendulum environments."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from empowerment_warp.baselines.toy_baselines.config import EnvironmentConfig


class EnvState(NamedTuple):
    """Physical state plus the TimeLimit counter for one environment."""

    physical: jax.Array
    episode_step: jax.Array


def observation_from_physical(env_name: str, physical: jax.Array) -> jax.Array:
    """Convert authoritative physical state to the policy/GCE observation."""

    if env_name == 'ball_in_box':
        return physical
    if env_name == 'pendulum':
        theta, theta_dot = physical[..., 0], physical[..., 1]
        return jnp.stack((jnp.cos(theta), jnp.sin(theta), theta_dot), axis=-1)
    raise ValueError(f'unknown Latent-GCE environment: {env_name}')


def physical_from_observation(env_name: str, observation: jax.Array) -> jax.Array:
    """Recover the physical state used to seed a counterfactual probe."""

    if env_name == 'ball_in_box':
        return observation
    if env_name == 'pendulum':
        theta = jnp.arctan2(observation[..., 1], observation[..., 0])
        return jnp.stack((theta, observation[..., 2]), axis=-1)
    raise ValueError(f'unknown Latent-GCE environment: {env_name}')


def _angle_normalize(theta: jax.Array) -> jax.Array:
    return (theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def pendulum_unclipped_angular_velocity(
    config: EnvironmentConfig,
    physical: jax.Array,
    action: jax.Array,
) -> jax.Array:
    """Return Zhao's next angular velocity before the speed-limit clip."""

    if config.name != 'pendulum':
        raise ValueError('pendulum_unclipped_angular_velocity requires the pendulum config.')
    clipped_action = jnp.clip(action, -config.action_limit, config.action_limit)
    theta, theta_dot = physical[0], physical[1]
    torque = clipped_action[0]
    return theta_dot + (
        -3.0 * 10.0 / 2.0 * jnp.sin(theta + jnp.pi) + 3.0 * torque
    ) * 0.1


def reset_one(config: EnvironmentConfig, key: jax.Array) -> tuple[jax.Array, EnvState]:
    """Reset one environment with Zhao's exact reset distribution."""

    if config.name == 'ball_in_box':
        physical = jax.random.uniform(key, (2,), minval=-9.0, maxval=9.0)
    elif config.name == 'pendulum':
        offsets = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        physical = offsets.at[0].add(jnp.pi)
    else:
        raise ValueError(f'unknown Latent-GCE environment: {config.name}')
    state = EnvState(physical=physical, episode_step=jnp.asarray(0, dtype=jnp.int32))
    return observation_from_physical(config.name, physical), state


def transition_physical(config: EnvironmentConfig, physical: jax.Array, action: jax.Array) -> jax.Array:
    """Apply one uncluttered physical transition without resets or rewards."""

    clipped_action = jnp.clip(action, -config.action_limit, config.action_limit)
    if config.name == 'ball_in_box':
        return jnp.clip(physical + clipped_action, -9.0, 9.0)
    if config.name == 'pendulum':
        theta = physical[0]
        new_theta_dot = pendulum_unclipped_angular_velocity(
            config,
            physical,
            clipped_action,
        )
        new_theta = _angle_normalize(theta + new_theta_dot * 0.1)
        new_theta_dot = jnp.clip(new_theta_dot, -8.0, 8.0)
        return jnp.stack((new_theta, new_theta_dot))
    raise ValueError(f'unknown Latent-GCE environment: {config.name}')


def step_one(
    config: EnvironmentConfig,
    state: EnvState,
    action: jax.Array,
    reset_key: jax.Array,
) -> tuple[jax.Array, EnvState, jax.Array, jax.Array]:
    """Step once and apply the TimeLimit reset used by the vector environment."""

    next_physical = transition_physical(config, state.physical, action)
    terminal_observation = observation_from_physical(config.name, next_physical)
    next_episode_step = state.episode_step + 1
    done = next_episode_step >= config.episode_steps
    continued_state = EnvState(physical=next_physical, episode_step=next_episode_step)
    reset_observation, reset_state = reset_one(config, reset_key)
    next_state = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), reset_state, continued_state)
    next_observation = jnp.where(done, reset_observation, terminal_observation)
    return next_observation, next_state, done, terminal_observation


def reset_batch(config: EnvironmentConfig, keys: jax.Array) -> tuple[jax.Array, EnvState]:
    """Vectorized reset."""

    return jax.vmap(lambda key: reset_one(config, key))(keys)


def step_batch(
    config: EnvironmentConfig,
    states: EnvState,
    actions: jax.Array,
    reset_keys: jax.Array,
) -> tuple[jax.Array, EnvState, jax.Array, jax.Array]:
    """Vectorized controller step."""

    return jax.vmap(lambda state, action, key: step_one(config, state, action, key))(
        states,
        actions,
        reset_keys,
    )


def state_grid(config: EnvironmentConfig, bins: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return grid axes and flattened observations for field evaluation."""

    if config.name == 'ball_in_box':
        x = jnp.linspace(-9.0, 9.0, bins)
        y = jnp.linspace(-9.0, 9.0, bins)
        xx, yy = jnp.meshgrid(x, y)
        observations = jnp.stack((xx, yy), axis=-1).reshape((-1, 2))
        return x, y, observations
    theta = jnp.linspace(-jnp.pi, jnp.pi, bins)
    theta_dot = jnp.linspace(-8.0, 8.0, bins)
    tt, vv = jnp.meshgrid(theta, theta_dot)
    observations = jnp.stack((jnp.cos(tt), jnp.sin(tt), vv), axis=-1).reshape((-1, 3))
    return theta, theta_dot, observations
