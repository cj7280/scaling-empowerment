"""Windowed uniform probes and correctly conditioned fresh learned-source branches."""
import jax
import jax.numpy as jnp

from empowerment_warp.baselines.toy_baselines.envs import (
    observation_from_physical, physical_from_observation, transition_physical)
from empowerment_warp.baselines.toy_baselines.replay import ProbeBatch


def uniform_actions(key, shape, dtype=jnp.float32):
    interior = jnp.nextafter(jnp.asarray(1., dtype), jnp.asarray(0., dtype))
    return jnp.clip(jax.random.uniform(key, shape, dtype=dtype, minval=-1., maxval=1.), -interior, interior)


def sample_roots(environment, key, count):
    """Uniform physical roots; evolved window starts need not remain uniform."""
    if environment.name == 'ball_in_box':
        return jax.random.uniform(key, (count, 2), minval=-9., maxval=9.)
    limits = jnp.array([jnp.pi, 8.])
    physical = jax.random.uniform(key, (count, 2), minval=-limits, maxval=limits)
    return observation_from_physical(environment.name, physical)


def rollout_actions(environment, observations, actions):
    """Return root plus all reached observations, without episode resets."""
    physical = physical_from_observation(environment.name, observations)
    def step(states, normalized):
        states = jax.vmap(lambda s, a: transition_physical(environment, s, a))(
            states, normalized * environment.action_limit)
        return states, observation_from_physical(environment.name, states)
    _, future = jax.lax.scan(step, physical, jnp.swapaxes(actions, 0, 1))
    return jnp.concatenate((observations[:, None], jnp.swapaxes(future, 0, 1)), axis=1)


def draw_source_batch(environment, source, params, observations, key):
    actions, z, log_h = source.apply({'params': params}, observations, key, method=source.sample)
    targets = rollout_actions(environment, observations, actions)[:, -1]
    return ProbeBatch(observations, actions, z, targets), log_h


def make_probe_collector(config, source=None):
    env = config.environment
    roots, horizon, windows = config.roots_per_update, env.horizon, config.windows_per_trace

    @jax.jit
    def collect(params, controller_observations, key):
        key, root_key, path_key, action_key = jax.random.split(key, 4)
        if config.mode == 'estimator':
            observations = sample_roots(env, root_key, roots)
        else:
            pool = controller_observations.reshape((-1, env.observation_dim))
            indices = jax.random.choice(root_key, len(pool), shape=(roots,), replace=False)
            observations = pool[indices]
        path_actions = uniform_actions(path_key, (roots, env.probe_trace_steps, env.action_dim))
        if config.method == 'vim':
            # Only a start walk is shared. A continuation from a different
            # source context is NOT a fresh sample from h(actions | this start).
            paths = rollout_actions(env, observations, path_actions[:, :windows - 1])
            starts = paths.reshape((-1, env.observation_dim))
            batch, log_h = draw_source_batch(env, source, params, starts, action_key)
            walk_steps = roots * (windows - 1)
            sequence_steps = len(starts) * horizon
        else:
            paths = rollout_actions(env, observations, path_actions)
            starts = paths[:, :windows].reshape((-1, env.observation_dim))
            targets = paths[:, horizon:].reshape((-1, env.observation_dim))
            actions = jnp.stack([path_actions[:, i:i+horizon] for i in range(windows)], axis=1)
            actions = actions.reshape((-1, horizon, env.action_dim))
            batch = ProbeBatch(starts, actions, jnp.arctanh(actions), targets)
            log_h = jnp.full(len(starts), -horizon * env.action_dim * jnp.log(2.))
            walk_steps, sequence_steps = 0, roots * env.probe_trace_steps
        return batch, log_h, dict(start_walk_steps=jnp.asarray(walk_steps),
                                 sequence_steps=jnp.asarray(sequence_steps),
                                 probe_steps=jnp.asarray(walk_steps + sequence_steps)), key
    return collect
