"""PQN-only Craftax reset and episode-metric parity with the reference backend."""

import jax
import jax.numpy as jnp

from empowerment_warp.envs.craftax_env import CraftaxBatchEnv
from empowerment_warp.envs.wrappers import FlatObservationEnv


class PQNEnv(FlatObservationEnv):
    def __init__(self, name, *, num_envs, episode_length, reset_ratio=16):
        if reset_ratio < 1:
            raise ValueError("reset_ratio must be positive")
        super().__init__(CraftaxBatchEnv(name, num_envs=num_envs, episode_length=episode_length))
        self.reset_ratio = int(reset_ratio) if name.startswith('craftax:') else 1
        while self.reset_ratio > 1 and num_envs % self.reset_ratio:
            self.reset_ratio -= 1
        self.achievement_columns = ()
        if name.startswith('craftax:'):
            from craftax.craftax_classic.constants import Achievement
            self.achievement_columns = tuple(
                (int(a.value), f'Achievements/{a.name.lower()}') for a in Achievement
            )

    def step(self, state, actions):
        state = super().step(state, actions)
        if self.achievement_columns:
            # Craftax emits achievements only at game-over. The reference also
            # reports them at the wrapper's shorter time limit (1000 steps).
            done = state.terminated | state.truncated
            achievements = state.game_state.achievements.astype(jnp.float32)
            info = dict(state.info)
            for column, key in self.achievement_columns:
                info[key] = 100.0 * achievements[:, column] * done.astype(jnp.float32)
            state = state._replace(info=info)
        return state

    def reset_done(self, state, keys):
        if self.reset_ratio == 1:
            return self.where_done(state, self.reset(keys))
        # Reference CraftaxGymVectorEnv::_resolve_resets: world pool of N/R,
        # unique pool entries assigned to sampled done lanes; overflow shares the
        # lane's default pool entry. Only world/state generation is shared: future
        # dynamics RNG remains independent per lane, like reference step_keys.
        count = self.num_envs // self.reset_ratio
        split = jax.vmap(jax.random.split)(keys)
        reset_keys = split[:, 1]
        obs, game_state = self.env.reset_many(reset_keys[:count])
        take = jnp.minimum(jnp.arange(self.num_envs) // self.reset_ratio, count - 1)
        done = state.terminated | state.truncated
        choice_key = jax.random.fold_in(reset_keys[0], 0xC0FFEE)
        chosen = jax.random.choice(choice_key, self.num_envs, shape=(count,),
                                   replace=False, p=done.astype(jnp.float32))
        take = take.at[chosen].set(jnp.arange(count))
        reset = state._replace(
            game_state=jax.tree_util.tree_map(lambda value: value[take], game_state),
            obs=self._flatten(obs[take]), reward=jnp.zeros_like(state.reward),
            terminated=jnp.zeros_like(state.terminated), truncated=jnp.zeros_like(state.truncated),
            step_count=jnp.zeros_like(state.step_count), rng=split[:, 0],
            info=jax.tree_util.tree_map(jnp.zeros_like, state.info),
        )
        return self.where_done(state, reset)


def reset_finished(env, state, keys):
    """Use the PQN reset pool when bound; retain generic game test adapters."""
    if isinstance(env, PQNEnv):
        return env.reset_done(state, keys)
    return env.where_done(state, env.reset(keys))
