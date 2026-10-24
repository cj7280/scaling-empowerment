"""Shared physical behavior metrics, independent of the intrinsic reward."""

import jax.numpy as jnp


def behavior_metrics(env_name: str, physical_states) -> dict[str, float]:
    """Summarize (..., episode_steps + 1, 2) traces, excluding reset states."""

    reached = jnp.asarray(physical_states)[..., 1:, :]
    if env_name == 'ball_in_box':
        clearance = 9.0 - jnp.max(jnp.abs(reached), axis=-1)
        return {
            'wall_fraction': float(jnp.mean((clearance <= 1.0).astype(jnp.float32))),
            'wall_distance_mean': float(jnp.mean(clearance)),
        }
    if env_name == 'pendulum':
        angle = (reached[..., 0] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
        speed = jnp.abs(reached[..., 1])
        upright = jnp.abs(angle) <= jnp.pi / 6.0
        return {
            'upright_fraction': float(jnp.mean(upright.astype(jnp.float32))),
            'absolute_angular_velocity_mean': float(jnp.mean(speed)),
            'upright_and_slow_fraction': float(jnp.mean((upright & (speed < 1.0)).astype(jnp.float32))),
        }
    raise ValueError(f'unsupported behavior environment: {env_name}')
