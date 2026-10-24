"""Small device-side adapters shared by JAX and MJX-Warp environments."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class FlatObservationEnv:
    """Expose arbitrary array observations as float32 vectors on device."""

    def __init__(self, env: Any):
        self.env = env
        self.num_envs = env.num_envs
        self.action_size = env.action_size
        keys = jax.ShapeDtypeStruct((self.num_envs, 2), jnp.uint32)
        state_shape = jax.eval_shape(env.reset, keys)
        self.observation_size = int(np.prod(state_shape.obs.shape[1:]))

    @staticmethod
    def _flatten(obs: jax.Array) -> jax.Array:
        value = obs.astype(jnp.float32)
        if obs.dtype == jnp.uint8:
            value = value / 255.0
        return value.reshape((value.shape[0], -1))

    def reset(self, keys: jax.Array) -> Any:
        state = self.env.reset(keys)
        return state._replace(obs=self._flatten(state.obs))

    def step(self, state: Any, actions: jax.Array) -> Any:
        state = self.env.step(state, actions)
        return state._replace(obs=self._flatten(state.obs))

    def where_done(self, state: Any, reset_state: Any) -> Any:
        return self.env.where_done(state, reset_state)
