"""Craftax branch evaluation protocol.

The NumPy classifier is local and frames come from the native renderer. With
``potential_steps=0`` (skill only) branches start directly at reset and media omit
the nonexistent potential continuation. No Gymnasium dependency.
"""
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np

ACHIEVEMENT_PREFIX = 'Achievements/'
_JAX_GAME_BRANCH_DEFAULT_STEPS = 6
_JAX_GAME_BRANCH_SNAPSHOT_ATTRS = (
    '_state', '_raw_obs', '_rng', '_episode_return', '_episode_length', '_episode_length_jax',
)


def profile_render_context(env, config):
    return nullcontext()



def is_achievement_key(key):
    return str(key).startswith(ACHIEVEMENT_PREFIX)


def achievement_name(key):
    return str(key).split('/', 1)[1]


def _achievement_columns(mapping, mask=None):
    """Return (column_arrays, names) for the Achievements/* keys, in sorted order."""
    mask_arr = None if mask is None else np.asarray(mask, dtype=bool).reshape(-1)
    columns = []
    names = []
    for key, value in sorted((mapping or {}).items()):
        if not is_achievement_key(key):
            continue
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError, IndexError):
            continue
        if arr.size == 0:
            continue
        if mask_arr is not None:
            if arr.shape != mask_arr.shape:
                continue
            arr = arr[mask_arr]
        columns.append(arr)
        names.append(achievement_name(key))
    return columns, names


def achievement_matrix_and_names(mapping, *, mask=None):
    """[N, achievements] terminal matrix plus the per-column achievement names."""
    columns, names = _achievement_columns(mapping, mask=mask)
    if not columns:
        return None, []
    min_len = min(arr.shape[0] for arr in columns)
    if min_len <= 0:
        return None, []
    return np.stack([arr[:min_len] for arr in columns], axis=-1), names


def achievement_score_from_values(values):
    """Craftax-style total achievement score from 0/100 achievement values."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    return np.exp(np.mean(np.log1p(np.maximum(values, 0.0)), axis=-1)) - 1.0


def achievement_summary(mapping, *, mask=None, score_values=None, episode_return_values=None):
    """Aggregate Craftax achievements into the few scalar curves used in W&B.

    Reports breadth (`achievement_count`, `achievement_percent`) and the per-achievement
    value `achievement/<name>` (0/100); averaged across episodes downstream the latter
    becomes the per-achievement success rate %. No geometric-mean score is emitted.
    """
    del score_values
    matrix, names = achievement_matrix_and_names(mapping, mask=mask)
    if matrix is None:
        return {}

    active = matrix > 0.0
    cumulative = np.sum(active, axis=-1, dtype=np.float32)
    percent = np.mean(active, axis=-1, dtype=np.float32) * 100.0

    summary = {
        'achievement_count': float(np.mean(cumulative)),
        'achievement_percent': float(np.mean(percent)),
    }
    for name, value in zip(names, np.mean(matrix, axis=0)):
        summary[f'achievement/{name}'] = float(value)

    if episode_return_values is not None:
        try:
            return_arr = np.asarray(episode_return_values, dtype=np.float32).reshape(-1)
            if mask is not None and return_arr.shape == np.asarray(mask, dtype=bool).reshape(-1).shape:
                return_arr = return_arr[np.asarray(mask, dtype=bool).reshape(-1)]
            if return_arr.size:
                summary['total_reward'] = float(np.mean(return_arr))
        except (TypeError, ValueError, IndexError):
            pass
    return summary


def state_achievement_values(env):
    """Return current Craftax achievements scaled like terminal info, or None."""
    base_env = getattr(env, 'unwrapped', env)
    current_state = getattr(base_env, 'current_state', None)
    state = current_state() if callable(current_state) else getattr(base_env, '_state', None)
    achievements = getattr(state, 'achievements', None)
    if achievements is None:
        return None
    try:
        values = np.asarray(achievements, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, IndexError):
        return None
    if values.size == 0:
        return None
    if np.nanmax(values) <= 1.0:
        values = values * 100.0
    return values


def state_achievement_names(env, count):
    """Return Craftax achievement names aligned to the current achievement vector."""
    base_env = getattr(env, 'unwrapped', env)
    names = [f'achievement_{idx}' for idx in range(int(count))]
    for col, key in getattr(base_env, '_achievement_key_by_col', ()) or ():
        col = int(col)
        if 0 <= col < len(names):
            names[col] = achievement_name(key)
    return names


def trajectory_record_from_state(env, *, step, reward=0.0, total_reward=0.0):
    values = state_achievement_values(env)
    if values is None:
        count = 0.0
        score = 0.0
        names = []
        unlocked = []
    else:
        count = float(np.sum(values > 0.0))
        score_values = achievement_score_from_values(values)
        score = float(score_values[0]) if np.asarray(score_values).size else 0.0
        names = state_achievement_names(env, int(values.size))
        unlocked = [str(name) for name, value in zip(names, values) if float(value) > 0.0]
    record = {
        'step': int(step),
        'reward': float(np.asarray(reward, dtype=np.float32).reshape(-1)[0]),
        'total_reward': float(total_reward),
        'num_achievements': count,
        'cumulative_achievements': count,
        'achievement_score': score,
        'total_achievement_score': score,
        'unlocked_achievements': unlocked,
    }
    if values is not None:
        record['achievement_names'] = [str(name) for name in names]
        record['achievement_values'] = [float(value) for value in values]
    return record


class _StatefulRecurrentEvalActor:
    """Carry one recurrent policy state across scalar Craftax eval steps."""

    def __init__(self, agent, rng=None):
        self._sample_actions = agent.sample_actions_recurrent
        self._sample_actions_with_perception = getattr(
            agent,
            'sample_actions_recurrent_with_perception',
            None,
        )
        self._active_perception = bool(
            getattr(agent, 'config', {}).get('craftax_active_perception', False)
        )
        self._reset_on_skill_boundary = bool(
            getattr(agent, 'config', {}).get('pqn_recurrent', False)
        )
        self._initial_recurrent_state = agent.initial_recurrent_state
        self._rng = jax.random.PRNGKey(0) if rng is None else rng
        self._recurrent_state = None
        self._reset_before_next_observation = True
        self._last_skills = None
        self.last_gaze = None
        self.last_policy_observation = None
        self.last_raw_observation = None

    def reset_recurrent_state(self):
        self._recurrent_state = None
        self._reset_before_next_observation = True
        self._last_skills = None
        self.last_gaze = None
        self.last_policy_observation = None
        self.last_raw_observation = None

    def snapshot_recurrent_state(self):
        return (
            self._recurrent_state,
            self._reset_before_next_observation,
            self._last_skills,
            self.last_gaze,
            self.last_policy_observation,
            self.last_raw_observation,
        )

    def restore_recurrent_state(self, snapshot):
        if len(snapshot) == 2:
            self._recurrent_state, self._reset_before_next_observation = snapshot
            self._last_skills = None
            self.last_gaze = None
            self.last_policy_observation = None
            self.last_raw_observation = None
            return
        if len(snapshot) == 5:
            (
                self._recurrent_state,
                self._reset_before_next_observation,
                self.last_gaze,
                self.last_policy_observation,
                self.last_raw_observation,
            ) = snapshot
            self._last_skills = None
            return
        (
            self._recurrent_state,
            self._reset_before_next_observation,
            self._last_skills,
            self.last_gaze,
            self.last_policy_observation,
            self.last_raw_observation,
        ) = snapshot

    def __call__(
        self,
        observations,
        skills=None,
        anchor_observations=None,
        temperature=1.0,
        deterministic=False,
    ):
        observations = jnp.asarray(observations, dtype=jnp.float32)
        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations[None, :]

        def optional_batch(value):
            if value is None:
                return None
            value = jnp.asarray(value, dtype=jnp.float32)
            return value[None, :] if unbatched else value

        skills = optional_batch(skills)
        anchor_observations = optional_batch(anchor_observations)

        skill_boundary = False
        if self._reset_on_skill_boundary and skills is not None and self._last_skills is not None:
            skill_boundary = bool(
                np.any(np.asarray(skills) != np.asarray(self._last_skills))
            )
        reset = jnp.full(
            (observations.shape[0],),
            self._reset_before_next_observation or skill_boundary,
            dtype=bool,
        )
        if self._recurrent_state is None:
            self._recurrent_state = self._initial_recurrent_state(
                observations.shape[0],
                dtype=observations.dtype,
            )
        self._rng, key = jax.random.split(self._rng)
        call_kwargs = dict(
            observations=observations,
            skills=skills,
            anchor_observations=anchor_observations,
            recurrent_state=self._recurrent_state,
            reset=reset,
            seed=key,
            temperature=temperature,
            deterministic=deterministic,
        )
        if self._active_perception:
            if not callable(self._sample_actions_with_perception):
                raise ValueError(
                    'Active-perception evaluation requires '
                    'sample_actions_recurrent_with_perception.'
                )
            (
                actions,
                self._recurrent_state,
                gaze,
                policy_observation,
            ) = self._sample_actions_with_perception(**call_kwargs)
            self.last_gaze = gaze[0] if unbatched else gaze
            self.last_policy_observation = (
                policy_observation[0] if unbatched else policy_observation
            )
            self.last_raw_observation = observations[0] if unbatched else observations
        else:
            actions, self._recurrent_state = self._sample_actions(**call_kwargs)
        self._reset_before_next_observation = False
        self._last_skills = skills
        return actions[0] if unbatched else actions


def _reset_recurrent_actor(actor_fn):
    reset_fn = getattr(actor_fn, 'reset_recurrent_state', None)
    if callable(reset_fn):
        reset_fn()


def _snapshot_recurrent_actor(actor_fn):
    snapshot_fn = getattr(actor_fn, 'snapshot_recurrent_state', None)
    return snapshot_fn() if callable(snapshot_fn) else None


def _restore_recurrent_actor(actor_fn, snapshot):
    restore_fn = getattr(actor_fn, 'restore_recurrent_state', None)
    if callable(restore_fn):
        restore_fn(snapshot)


def _is_craftax_config(config):
    config = {} if config is None else config
    return config.get('benchmark_backend') == 'craftax' or config.get('env_profile') == 'craftax'


def _render_eval_frame(env, config, current_skill, *, add_skill_legend=True):
    with profile_render_context(env, config):
        frame = env.render().copy()

    return frame


def _skill_subset_indices(skill_library, count):
    """Evenly-spaced subset of skill-library indices of size <= count."""
    n = int(len(skill_library))
    count = max(1, int(count))
    if count >= n:
        return list(range(n))
    return [int(round(i * (n - 1) / (count - 1))) for i in range(count)] if count > 1 else [0]


def _copy_jax_game_leaf(value):
    if hasattr(value, 'shape'):
        return np.array(jax.device_get(value), copy=True)
    return value


def _restore_jax_game_leaf(value):
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    return value


def _copy_jax_game_tree(tree):
    return jax.tree_util.tree_map(_copy_jax_game_leaf, tree)


def _restore_jax_game_tree(tree):
    return jax.tree_util.tree_map(_restore_jax_game_leaf, tree)


def _capture_jax_game_env_snapshot(env):
    base_env = getattr(env, 'unwrapped', env)
    if getattr(base_env, '_state', None) is None:
        return None
    snapshot = {}
    for attr in _JAX_GAME_BRANCH_SNAPSHOT_ATTRS:
        if hasattr(base_env, attr):
            snapshot[attr] = _copy_jax_game_tree(getattr(base_env, attr))
    return snapshot


def _restore_jax_game_env_snapshot(env, snapshot):
    if snapshot is None:
        raise ValueError('JAX-game branch snapshot is required for restore.')
    base_env = getattr(env, 'unwrapped', env)
    for attr, value in snapshot.items():
        if attr in {'_episode_return', '_episode_length'}:
            restored = np.array(value, copy=True)
        else:
            restored = _restore_jax_game_tree(value)
        setattr(base_env, attr, restored)


def _jax_game_branch_prefix_steps(config, warmup_steps, max_steps):
    potential_steps = config.get('potential_steps', None)
    if potential_steps is not None and int(potential_steps) == 0:
        # Skill-only training commands a skill on the reset observation. Branch
        # directly from that reset instead of fabricating a zero-skill action.
        return 0
    for key in ('eval_skill_grid_branch_steps', 'eval_branch_warmup_steps', 'skill_grid_branch_steps'):
        value = config.get(key, None)
        if value is not None and int(value) > 0:
            return max(1, min(int(value), max(1, int(max_steps) - 1)))
    if potential_steps is not None and int(potential_steps) > 0:
        return max(1, min(int(potential_steps), max(1, int(max_steps) - 1)))
    if int(warmup_steps) > 0:
        return max(1, min(int(warmup_steps), max(1, int(max_steps) - 1)))
    return max(1, min(_JAX_GAME_BRANCH_DEFAULT_STEPS, max(1, int(max_steps) - 1)))


def _skill_library_dim(skill_library, fallback):
    if skill_library is not None and len(skill_library) > 0:
        return int(np.asarray(skill_library[0]).reshape(-1).size)
    return int(fallback)


def _copy_branch_records(records):
    return [dict(record) for record in records]


def _collect_jax_game_branch_skill_rollouts(
    env,
    actor_fn,
    config,
    skill_library,
    *,
    count,
    warmup_steps,
    video_frame_skip,
    eval_temperature,
    eval_deterministic,
    supports_deterministic,
    seed,
):
    """Collect skill branches after the configured potential-policy prefix."""
    if (
        skill_library is None
        or len(skill_library) == 0
        or int(config.get('skill_dim', 0) or 0) <= 0
    ):
        return []
    if not _is_craftax_config(config):
        return []

    max_steps = int(getattr(env, '_episode_limit', config.get('max_episode_steps', 0) or 0))
    if max_steps <= 1:
        return []
    prefix_steps = _jax_game_branch_prefix_steps(config, warmup_steps, max_steps)
    skill_indices = _skill_subset_indices(skill_library, count)
    frame_skip = max(1, int(video_frame_skip))
    skill_dim = _skill_library_dim(skill_library, config.get('skill_dim', 0) or 0)
    zero_skill = np.zeros(skill_dim, dtype=np.float32)
    entry_snapshot = _capture_jax_game_env_snapshot(env)
    entry_actor_snapshot = _snapshot_recurrent_actor(actor_fn)

    observation, _ = env.reset(seed=int(seed) + 97_000)
    _reset_recurrent_actor(actor_fn)
    anchor_obs = observation
    done = False
    step = 0
    prefix_frames = []
    prefix_records = []
    prefix_frames.append(_render_eval_frame(env, config, zero_skill, add_skill_legend=False))
    prefix_records.append(trajectory_record_from_state(env, step=0))

    while not done and step < prefix_steps:
        action_kwargs = dict(
            observations=observation,
            skills=zero_skill,
            anchor_observations=anchor_obs,
            temperature=eval_temperature,
        )
        if eval_deterministic and supports_deterministic:
            action_kwargs['deterministic'] = True
        action = np.array(actor_fn(**action_kwargs))
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        anchor_obs = observation

        total_reward = prefix_records[-1]['total_reward'] if prefix_records else 0.0
        total_reward += float(np.asarray(reward, dtype=np.float32).reshape(-1)[0])
        prefix_records.append(
            trajectory_record_from_state(
                env,
                step=step,
                reward=reward,
                total_reward=total_reward,
            )
        )
        if step % frame_skip == 0 or done or step == prefix_steps:
            prefix_frames.append(_render_eval_frame(env, config, zero_skill, add_skill_legend=False))

    branch_snapshot = _capture_jax_game_env_snapshot(env)
    branch_actor_snapshot = _snapshot_recurrent_actor(actor_fn)
    if branch_snapshot is None:
        if entry_snapshot is not None:
            _restore_jax_game_env_snapshot(env, entry_snapshot)
        _restore_recurrent_actor(actor_fn, entry_actor_snapshot)
        return []
    branch_observation = np.array(observation, copy=True)
    branch_anchor_obs = np.array(anchor_obs, copy=True)
    # With no prefix, the reset frame already belongs to the commanded skill.
    activation_frame = -1 if prefix_steps == 0 else max(0, len(prefix_frames) - 1)

    def collect_branch_rollout(skill_idx, current_skill, *, is_potential_continuation=False):
        _restore_jax_game_env_snapshot(env, branch_snapshot)
        _restore_recurrent_actor(actor_fn, branch_actor_snapshot)
        observation = np.array(branch_observation, copy=True)
        anchor_obs = np.array(branch_anchor_obs, copy=True)
        current_skill = np.array(current_skill, copy=True)
        done_local = bool(done)
        step_local = int(step)

        frames = [np.array(frame, copy=True) for frame in prefix_frames]
        records = _copy_branch_records(prefix_records)
        total_reward = records[-1]['total_reward'] if records else 0.0

        branch_start_step = int(step)
        while not done_local and step_local < max_steps:
            action_kwargs = dict(
                observations=observation,
                skills=current_skill,
                anchor_observations=anchor_obs,
                temperature=eval_temperature,
            )
            if eval_deterministic and supports_deterministic:
                action_kwargs['deterministic'] = True
            action = np.array(actor_fn(**action_kwargs))
            observation, reward, terminated, truncated, info = env.step(action)
            done_local = terminated or truncated
            step_local += 1
            if np.linalg.norm(current_skill) <= 1e-6:
                anchor_obs = np.array(observation, copy=True)

            total_reward += float(np.asarray(reward, dtype=np.float32).reshape(-1)[0])
            records.append(
                trajectory_record_from_state(
                    env,
                    step=step_local,
                    reward=reward,
                    total_reward=total_reward,
                )
            )
            if (
                step_local == branch_start_step + 1 or step_local % frame_skip == 0 or done_local
            ):
                frames.append(_render_eval_frame(env, config, current_skill, add_skill_legend=False))

        if frames:
            return {
                'skill_index': int(skill_idx),
                'skill': np.array(current_skill, copy=True),
                'frames': np.stack(frames, axis=0),
                'records': records,
                'branch_prefix_steps': int(prefix_steps),
                'activation_frame': int(activation_frame),
                'is_potential_continuation': bool(is_potential_continuation),
            }
        return None

    rollouts = []
    if prefix_steps > 0:
        potential_rollout = collect_branch_rollout(
            -1,
            zero_skill,
            is_potential_continuation=True,
        )
        if potential_rollout is not None:
            rollouts.append(potential_rollout)
    for skill_idx in skill_indices:
        rollout = collect_branch_rollout(skill_idx, skill_library[skill_idx])
        if rollout is not None:
            rollouts.append(rollout)

    if entry_snapshot is not None:
        _restore_jax_game_env_snapshot(env, entry_snapshot)
    _restore_recurrent_actor(actor_fn, entry_actor_snapshot)
    return rollouts
