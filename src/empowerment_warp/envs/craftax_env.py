"""Batched Craftax-Classic with explicit resets and caller-visible terminal observations."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnvNoAutoReset

ENVIRONMENTS = {"craftax:classic-symbolic": CraftaxClassicSymbolicEnvNoAutoReset}


class State(NamedTuple):
    game_state: Any
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    step_count: jax.Array
    rng: jax.Array
    info: Any


class CraftaxBatchEnv:
    """Vectorized Craftax; the caller resets finished lanes with `where_done`."""

    def __init__(self, name: str, *, num_envs: int, episode_length: int | None = None):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if episode_length is not None and episode_length <= 0:
            raise ValueError("episode_length must be positive")
        self.name = name
        self.num_envs = int(num_envs)
        self.episode_length = episode_length
        if name not in ENVIRONMENTS:
            raise ValueError(f"unsupported environment {name!r}; available: {', '.join(ENVIRONMENTS)}")
        self.game = ENVIRONMENTS[name]()
        self.params = self.game.default_params
        self.action_size = int(self.game.num_actions)
        self.reset_many = jax.vmap(lambda key: self.game.reset(key, self.params))
        self.step_many = jax.vmap(lambda key, state, action: self.game.step(key, state, action, self.params))
        shape_key = jax.ShapeDtypeStruct((2,), jnp.uint32)

        def scalar_info(key: jax.Array) -> Any:
            _, game_state = self.game.reset(key, self.params)
            return self.game.step(
                key,
                game_state,
                jnp.asarray(0, dtype=jnp.int32),
                self.params,
            )[4]

        info_shapes = jax.eval_shape(scalar_info, shape_key)
        self._empty_info = jax.tree_util.tree_map(
            lambda value: jnp.zeros(value.shape, value.dtype),
            info_shapes,
        )
        self._reset_batched = jax.jit(self._reset_impl)
        self._step_batched = jax.jit(self._step_impl)

    def _reset_impl(self, keys: jax.Array) -> State:
        next_keys, reset_keys = jax.vmap(jax.random.split)(keys).transpose((1, 0, 2))
        obs, game_state = self.reset_many(reset_keys)
        return State(
            game_state=game_state,
            obs=obs,
            reward=jnp.zeros((self.num_envs,), dtype=jnp.float32),
            terminated=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
            truncated=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
            step_count=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            rng=next_keys,
            info=jax.tree_util.tree_map(
                lambda value: jnp.broadcast_to(value, (self.num_envs,) + value.shape),
                self._empty_info,
            ),
        )

    def _step_impl(self, state: State, actions: jax.Array) -> State:
        split_keys = jax.vmap(jax.random.split)(state.rng)
        next_keys = split_keys[:, 0]
        step_keys = split_keys[:, 1]
        obs, game_state, reward, done, info = self.step_many(step_keys, state.game_state, actions)
        step_count = state.step_count + 1
        if self.episode_length is None:
            truncated = jnp.zeros_like(done, dtype=jnp.bool_)
        else:
            # time limit shared with the physics envs, so `episode_length` means one thing
            truncated = step_count >= self.episode_length
        return State(
            game_state=game_state,
            obs=obs,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            terminated=jnp.asarray(done, dtype=jnp.bool_),
            truncated=truncated,
            step_count=step_count,
            rng=next_keys,
            info=info,
        )

    def reset(self, keys: jax.Array) -> State:
        if keys.shape != (self.num_envs, 2):
            raise ValueError(f"expected reset keys with shape {(self.num_envs, 2)}, got {keys.shape}")
        return self._reset_batched(keys)

    def step(self, state: State, actions: jax.Array) -> State:
        if actions.shape != (self.num_envs,):
            raise ValueError(f"expected actions with shape {(self.num_envs,)}, got {actions.shape}")
        return self._step_batched(state, actions)

    @staticmethod
    @jax.jit
    def where_done(state: State, reset_state: State) -> State:
        done = state.terminated | state.truncated

        def choose(current: Any, reset: Any) -> Any:
            if not hasattr(current, "shape"):
                return current
            mask = done.reshape(done.shape + (1,) * (current.ndim - done.ndim))
            return jnp.where(mask, reset, current)

        return State(
            game_state=jax.tree_util.tree_map(choose, state.game_state, reset_state.game_state),
            obs=choose(state.obs, reset_state.obs),
            reward=choose(state.reward, reset_state.reward),
            terminated=choose(state.terminated, reset_state.terminated),
            truncated=choose(state.truncated, reset_state.truncated),
            step_count=choose(state.step_count, reset_state.step_count),
            rng=choose(state.rng, reset_state.rng),
            info=jax.tree_util.tree_map(choose, state.info, reset_state.info),
        )
