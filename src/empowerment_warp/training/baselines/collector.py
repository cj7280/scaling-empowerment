"""Warp state branching and collection; the baseline learner is environment-independent."""
import jax
import jax.numpy as jnp

from empowerment_warp.core.rollout import (
    _locomotion_xy, _mark_coverage, coverage_track_specs, initial_coverage_grids,
)
from empowerment_warp.agents.ppo import (
    ControllerState, RolloutBatch, empty_observation_norm, squashed_log_prob,
    normalize_observations, _update_observation_norm,
)
from .probes import uniform_actions
from .replay import ProbeBatch


EXPLORATION_DISPLACEMENT_FIELDS = (
    'exploration/episode_agent_disp_xy', 'exploration/episode_ball_disp_xy',
    'exploration/episode_cube_disp_xyz', 'exploration/episode_cube_disp_max_xyz',
    'exploration/episode_cube_disp_xy', 'exploration/episode_cube_disp_max_xy',
    'exploration/obj_moved',
)


def exploration_positions(env, state):
    if hasattr(env, '_object_positions'):
        objects = jax.vmap(env._object_positions)(state.data)
        return (objects[..., :2], objects[:, 0], objects[:, 1], objects[:, 2], state.obs[:, :3])
    return _locomotion_xy(state, env.config.num_envs)[:2 if env.soccer else 1]


def exploration_specs(env, state):
    specs = coverage_track_specs(env, state)
    return specs[:1] if hasattr(env, 'soccer') and not env.soccer else specs


def episode_displacements(env, reached, starts, positions):
    if hasattr(env, '_object_positions'):
        delta = jnp.stack(positions[1:4], axis=1) - reached.initial_object_pos
        xyz, xy = jnp.linalg.norm(delta, axis=-1), jnp.linalg.norm(delta[..., :2], axis=-1)
        return dict(episode_cube_disp_xyz=xyz.mean(-1), episode_cube_disp_max_xyz=xyz.max(-1),
                    episode_cube_disp_xy=xy.mean(-1), episode_cube_disp_max_xy=xy.max(-1),
                    obj_moved=reached.metrics.obj_moved)
    return {f'episode_{name}_disp_xy':jnp.linalg.norm(end-start, axis=-1)
            for name, start, end in zip(('agent', 'ball'), starts, positions)}


def select_lanes(mask, current, replacement):
    """Select complete simulator lanes, including hidden MJX and actuator state."""
    def choose(old, new):
        expanded = mask.reshape(mask.shape + (1,) * (old.ndim - mask.ndim))
        return jnp.where(expanded, new, old)
    fields = {}
    for name in current._fields:
        old, new = getattr(current, name), getattr(replacement, name)
        fields[name] = (jax.vmap(lambda a, b, m: a.where(m, b))(old, new, mask)
                        if name == 'data' else jax.tree.map(choose, old, new))
    return current._replace(**fields)


def initialize(env, key):
    state = env.reset(jax.random.split(key, env.config.num_envs))
    return ControllerState(state, state.obs, empty_observation_norm(env.observation_size),
                           initial_coverage_grids(exploration_specs(env, state)),
                           exploration_positions(env, state))


def make_controller_collector(env, policy, recipe):
    @jax.jit
    def collect(train_state, controller, key):
        key, root_key = jax.random.split(key)
        root_times = jax.random.randint(root_key, (recipe.num_envs,), 0, recipe.rollout_steps)
        frozen_norm = controller.observation_norm

        def step(carry, time):
            state, roots, rng = carry
            roots = select_lanes(root_times == time, roots, state.env_state)
            policy_obs = state.observation
            if recipe.ppo.normalize_actor_observations:
                policy_obs = normalize_observations(frozen_norm, policy_obs,
                    epsilon=recipe.ppo.normalization_epsilon, clip=recipe.ppo.normalization_clip)
            distribution, value = policy.apply({'params': train_state.params}, policy_obs)
            rng, action_key, reset_key = jax.random.split(rng, 3)
            raw_action = distribution.sample(seed=action_key)
            reached = env.step(state.env_state, jnp.tanh(raw_action))
            done = reached.terminated | reached.truncated
            reset = env.reset(jax.random.split(reset_key, recipe.num_envs))
            continued = env.where_done(reached, reset)
            batch = RolloutBatch(state.observation, policy_obs, raw_action,
                                 squashed_log_prob(distribution, raw_action), value, done, reached.obs,
                                 reached.terminated)
            positions = exploration_positions(env, reached)
            coverage = _mark_coverage(state.coverage_grids, exploration_specs(env, reached),
                                      positions, jnp.zeros_like(done))
            displacement = episode_displacements(env, reached, state.episode_start_positions, positions)
            metrics = {name:jnp.sum(jnp.where(done, value, 0.0)) for name,value in displacement.items()}
            metrics['completed_episodes'] = jnp.sum(done)
            reset_positions = exploration_positions(env, continued)
            starts = tuple(jnp.where(done.reshape(done.shape + (1,)*(old.ndim-1)), new, old)
                           for old,new in zip(state.episode_start_positions, reset_positions))
            controller = ControllerState(continued, continued.obs, state.observation_norm, coverage, starts)
            return (controller, roots, rng), (batch, metrics)

        (controller, roots, key), (batch, metrics) = jax.lax.scan(
            step, (controller, controller.env_state, key), jnp.arange(recipe.rollout_steps))
        if recipe.ppo.normalize_actor_observations:
            moments = _update_observation_norm(frozen_norm, batch.observations.reshape(
                (-1, batch.observations.shape[-1])))
            controller = controller._replace(observation_norm=moments)
        return controller, roots, batch, key, jax.tree.map(jnp.sum, metrics)
    return collect


def outcome(recipe, observations):
    return observations if recipe.outcome_indices is None else observations[..., jnp.array(recipe.outcome_indices)]


def make_probe_collector(env, recipe, source):
    @jax.jit
    def collect(params, roots, key):
        key, action_key = jax.random.split(key)
        if recipe.method == 'vim':
            actions, pre_tanh, _ = source.apply({'params': params}, roots.obs, action_key,
                                                 method=source.sample)
        else:
            actions = uniform_actions(action_key, (recipe.num_envs, recipe.horizon, env.action_size))
            pre_tanh = jnp.arctanh(actions)
        def step(state, action):
            reached = env.step(state, action)
            return reached, (jnp.all(jnp.isfinite(reached.obs)), reached.terminated.mean())
        terminal, (finite, terminated) = jax.lax.scan(step, roots, jnp.swapaxes(actions, 0, 1))
        batch = ProbeBatch(roots.obs, actions, pre_tanh, outcome(recipe, terminal.obs))
        return batch, {'probe_all_finite': jnp.all(finite),
                       'probe_terminated_fraction': terminated.mean()}, key
    return collect


def make_evaluator(env, policy, recipe, *, deterministic):
    @jax.jit
    def evaluate(params, observation_norm=None):
        initial = env.reset(jax.random.split(jax.random.PRNGKey(91001), recipe.num_envs))
        def step(carry, _):
            state, key = carry
            key, action_key, reset_key = jax.random.split(key, 3)
            policy_obs = state.obs
            if recipe.ppo.normalize_actor_observations:
                policy_obs = normalize_observations(observation_norm, policy_obs,
                    epsilon=recipe.ppo.normalization_epsilon, clip=recipe.ppo.normalization_clip)
            distribution, _ = policy.apply({'params': params}, policy_obs)
            action = distribution.mean() if deterministic else distribution.sample(seed=action_key)
            action = jnp.tanh(action)
            reached = env.step(state, action)
            reset = env.reset(jax.random.split(reset_key, recipe.num_envs))
            continued = env.where_done(reached, reset)
            render_state = {name:getattr(reached.data,name)[0] for name in
                            ('qpos','qvel','ctrl','mocap_pos','mocap_quat')}
            row = (state.obs, reached.obs, action, reached.reward,
                   reached.terminated | reached.truncated, reached.metrics, render_state)
            return (continued, key), row
        _, rows = jax.lax.scan(step, (initial, jax.random.PRNGKey(91101)), None,
                               length=recipe.episode_steps)
        return rows
    return evaluate
