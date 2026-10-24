"""Warp-native recurrent on-policy chunk collector for PQN.

It produces the batch ``PotentialPQNAgent.update_rollout`` consumes, built on the
env interface (``reset``/``step``/``where_done``).

Recurrence bookkeeping:
  * carry holds ``recurrent_state`` S_t (the packed LSTM (c,h)+last_action the
    agent's sampler returns) and ``recurrent_last_done`` d_{t-1};
  * step t samples with reset d_{t-1}, records ``rnn_resets[t]=d_{t-1}`` and
    ``terminals[t]=done_t``, then carries S_{t+1} and d_t;
  * the chunk's ``initial_recurrent_state`` for the loss recurrence is S_0 (the
    state carried in from the previous chunk), so ``update_rollout`` re-runs the
    exact same reset/last-action sequence over the frozen chunk.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from empowerment_warp.training.pqn.env import reset_finished as reset_finished_env
from flax import struct

SkillSampler = Callable[[jax.Array, int], jax.Array]


def make_pqn_encoder_branch_sweep(
    env: Any,
    agent: Any,
    *,
    num_envs: int,
    skill_dim: int,
    horizon: int,
    skill_sampler: SkillSampler,
    observation_fn: Callable[[jax.Array], jax.Array],
) -> Callable[..., Any]:
    """Run an encoder-only commanded branch from copied live physical states.

    The branch wrapper clock starts at zero, as in SAC's marginal branch rollout;
    the game state and its physical clocks remain copied from the live collector.
    Rows are returned only after the entire sweep has executed. A death ends that
    lane's valid prefix, while the time-major shape retains the archive stride.
    """
    if horizon < 2:
        raise ValueError("encoder branch horizon must be at least two")

    @jax.jit
    def sweep(policy_variables, live_env_state, epsilon, key, sweep_id,
              live_recurrent_state=None, live_recurrent_reset=None):
        skill_key, rng_key, rollout_key = jax.random.split(key, 3)
        skills = skill_sampler(skill_key, num_envs)
        # Re-key branch stochastic dynamics without mutating the live state.
        branch_rng = jax.random.split(rng_key, num_envs)
        state = live_env_state._replace(
            reward=jnp.zeros_like(live_env_state.reward),
            terminated=jnp.zeros_like(live_env_state.terminated),
            truncated=jnp.zeros_like(live_env_state.truncated),
            step_count=jnp.zeros_like(live_env_state.step_count),
            rng=branch_rng,
        )
        policy_anchor = state.obs
        encoder_anchor = observation_fn(policy_anchor)
        # Configs written before this flag existed reset memory at the switch.
        if agent.config.get("pqn_preserve_memory_at_switch", False):
            if live_recurrent_state is None or live_recurrent_reset is None:
                raise ValueError("memory-preserving encoder branches require the live recurrent state and reset mask")
            initial_rnn, initial_reset = live_recurrent_state, live_recurrent_reset
        else:
            initial_rnn = agent.initial_recurrent_state(num_envs)
            initial_reset = jnp.ones((num_envs,), jnp.float32)

        def one_step(carry, t):
            current, rnn, reset, alive, rng = carry
            rng, action_key = jax.random.split(rng)
            _, action_idx, next_rnn = agent.sample_actions_recurrent_from_params(
                policy_variables, current.obs, skills, policy_anchor,
                rnn, reset, action_key, epsilon, deterministic=False,
            )
            stepped = env.step(current, action_idx)
            finite = jnp.all(jnp.isfinite(stepped.obs).reshape((num_envs, -1)), axis=-1)
            valid = alive & finite & ~stepped.terminated & ~stepped.truncated
            def retain(current_leaf, stepped_leaf):
                if not hasattr(current_leaf, "shape"):
                    return current_leaf
                mask = valid.reshape((num_envs,) + (1,) * (current_leaf.ndim - 1))
                return jnp.where(mask, stepped_leaf, current_leaf)

            next_state = jax.tree_util.tree_map(retain, current, stepped)
            outcome = jnp.where(valid[:, None], stepped.obs, current.obs)
            next_carry = (next_state, next_rnn, jnp.zeros_like(reset), valid, rng)
            return next_carry, (observation_fn(outcome), valid)

        _, (outcomes, valid) = jax.lax.scan(
            one_step,
            (state, initial_rnn, initial_reset,
             jnp.ones((num_envs,), jnp.bool_), rollout_key),
            jnp.arange(horizon, dtype=jnp.int32),
        )
        episode_ids = sweep_id * num_envs + jnp.arange(num_envs, dtype=jnp.int32)
        broadcast = lambda x: jnp.broadcast_to(x, (horizon,) + x.shape)
        flatten = lambda x: x.reshape((horizon * num_envs,) + x.shape[2:])
        batch = {
            "observations": flatten(outcomes),
            "anchor_observations": flatten(broadcast(encoder_anchor)),
            "skills": flatten(jnp.where(valid[..., None], broadcast(skills), 0.0)),
            "episode_ids": flatten(broadcast(episode_ids)),
            "timesteps": flatten(jnp.broadcast_to(
                jnp.arange(horizon, dtype=jnp.int32)[:, None], (horizon, num_envs))),
        }
        return batch, jnp.sum(valid.astype(jnp.int32))

    return sweep


@struct.dataclass
class PotentialPQNCollectorState:
    """Recurrent carry for potential-PQN: adds the skill schedule + anchor + episode id."""

    env_state: Any
    recurrent_state: jax.Array
    recurrent_last_done: jax.Array
    skills: jax.Array  # [num_envs, skill_dim]; zero during the potential prefix
    anchor_observations: jax.Array  # [num_envs, obs]; the state where the live skill activated
    episode_ids: jax.Array  # [num_envs] int32
    next_episode_id: jax.Array  # scalar int32
    episode_return: jax.Array
    episode_length: jax.Array  # [num_envs] steps elapsed in the live episode
    rng: jax.Array
    # [num_envs] bool; "separate lanes" ablation: a potential lane runs z=0 for its WHOLE
    # episode (never activates a skill), redrawn at every reset with probability
    # `potential_lane_fraction`. All-False (two-phase base) leaves every rule below unchanged.
    # None (states built by hand, e.g. the eval protocol) = no potential lanes.
    potential_lane: jax.Array | None = None


def _draw_potential_lanes(key: jax.Array, num_envs: int, fraction: float) -> jax.Array:
    if float(fraction) <= 0.0:
        return jnp.zeros((num_envs,), dtype=jnp.bool_)
    # fold_in keeps the existing split streams (action/skill/reset keys) byte-identical.
    return jax.random.uniform(jax.random.fold_in(key, 7), (num_envs,)) < float(fraction)


def initialize_potential_pqn_collector(
    env: Any,
    agent: Any,
    key: jax.Array,
    *,
    num_envs: int,
    skill_dim: int,
    potential_steps: int,
    skill_sampler: SkillSampler,
    potential_lane_fraction: float = 0.0,
) -> PotentialPQNCollectorState:
    """Reset lanes; lanes past the zero-skill prefix (step 0 >= potential_steps, i.e.
    potential_steps == 0) start with a live skill anchored at their reset obs.
    `potential_lane_fraction` > 0 (separate-lanes ablation) marks that fraction of lanes
    as whole-episode z=0 lanes."""
    reset_root, skill_key, next_key = jax.random.split(key, 3)
    reset_keys = jax.random.split(reset_root, num_envs)
    env_state = env.reset(reset_keys)
    sampled = skill_sampler(skill_key, num_envs)
    potential_lane = _draw_potential_lanes(skill_key, num_envs, potential_lane_fraction)
    active_at_start = (jnp.asarray(env_state.step_count, jnp.int32) >= int(potential_steps)) & (~potential_lane)
    skills = jnp.where(active_at_start[:, None], sampled, jnp.zeros((num_envs, skill_dim), jnp.float32))
    return PotentialPQNCollectorState(
        env_state=env_state,
        recurrent_state=agent.initial_recurrent_state(num_envs),
        recurrent_last_done=jnp.ones((num_envs,), dtype=jnp.float32),
        skills=skills,
        anchor_observations=env_state.obs,
        episode_ids=jnp.arange(num_envs, dtype=jnp.int32),
        next_episode_id=jnp.asarray(num_envs, dtype=jnp.int32),
        episode_return=jnp.zeros((num_envs,), dtype=jnp.float32),
        episode_length=jnp.zeros((num_envs,), dtype=jnp.float32),
        rng=next_key,
        potential_lane=potential_lane,
    )


def make_potential_pqn_chunk(
    env: Any,
    agent: Any,
    *,
    chunk_steps: int,
    num_envs: int,
    skill_dim: int,
    potential_steps: int,
    skill_sampler: SkillSampler,
    zero_eps: float = 1e-6,
    deterministic: bool = False,
    achievement_keys: tuple[str, ...] = (),
    potential_lane_fraction: float = 0.0,
) -> Callable[..., Any]:
    """Compile one recurrent potential-PQN chunk (skills/anchors + skill-boundary reset).

    `potential_lane_fraction` > 0 = the "separate lanes" ablation: at every reset a lane is
    a whole-episode z=0 (potential) lane with that probability, otherwise a skill lane
    (skill active from its first step under potential_steps == 0). 0.0 = two-phase method.

    Skill schedule mirrors the SAC collector (`core/rollout.py:one_step`): a
    zero-skill prefix of `potential_steps`, then one sampled skill for the rest of
    the episode, anchored at activation. The LSTM reset carried to the next step is
    `done | commanded_skill_boundary` (reference `recurrent_pqn`), so the recurrence
    restarts at episode boundaries and, by default, at skill switches. With
    preserved switch memory only the TD done mask includes command boundaries.
    Emits a flat T-major batch with the fields the frozen-reward assembly needs.
    """
    if chunk_steps < 1:
        raise ValueError("PQN collection requires at least one step.")

    @jax.jit
    def collect(policy_variables, carry: PotentialPQNCollectorState, epsilon):
        epsilon = jnp.asarray(epsilon, dtype=jnp.float32)

        def step_fn(carry: PotentialPQNCollectorState, _):
            rng, action_key, skill_key, reset_key = jax.random.split(carry.rng, 4)
            obs = carry.env_state.obs
            actions, action_idx, next_recurrent_state = (
                agent.sample_actions_recurrent_from_params(
                    policy_variables,
                    obs,
                    carry.skills,
                    carry.anchor_observations,
                    carry.recurrent_state,
                    carry.recurrent_last_done,
                    action_key,
                    jnp.asarray(0.0) if deterministic else epsilon,
                    deterministic=deterministic,
                )
            )
            next_env_state = env.step(carry.env_state, action_idx)
            done = (next_env_state.terminated | next_env_state.truncated)
            done_f = done.astype(jnp.float32)
            reward = jnp.asarray(next_env_state.reward, dtype=jnp.float32)
            next_obs = next_env_state.obs  # true s' (terminal obs on done, pre-reset)

            # Skill schedule (warp rollout.py:one_step parity).
            sampled_skills = skill_sampler(skill_key, num_envs)
            skill_on = jnp.linalg.norm(carry.skills, axis=-1) > zero_eps
            has_lanes = carry.potential_lane is not None
            if float(potential_lane_fraction) > 0.0 and not has_lanes:
                raise ValueError("potential_lane_fraction > 0 needs a collector state with potential_lane set")
            activate = (~skill_on) & (jnp.asarray(next_env_state.step_count, jnp.int32) >= int(potential_steps))
            if has_lanes:
                activate = activate & (~carry.potential_lane)
            next_skills = jnp.where(activate[:, None], sampled_skills, carry.skills)
            next_skill_on = jnp.linalg.norm(next_skills, axis=-1) > zero_eps
            anchor_reset = ((~skill_on) & next_skill_on) | (~next_skill_on)
            transition_next_anchors = jnp.where(
                anchor_reset[:, None], next_obs, carry.anchor_observations
            )

            transition = dict(
                observations=obs,
                actions=jnp.asarray(actions, dtype=jnp.float32),
                rewards=reward,
                terminals=done_f,
                rnn_resets=carry.recurrent_last_done,
                skills=carry.skills,
                next_skills=next_skills,
                anchor_observations=carry.anchor_observations,
                next_anchor_observations=transition_next_anchors,
                next_observations=next_obs,
                episode_ids=carry.episode_ids.astype(jnp.float32),
                # step_count of the CURRENT obs (pre-step), for the gamma^delay flip
                # discount in the skill-0 K-way replacement value (Ant parity).
                timesteps=jnp.asarray(carry.env_state.step_count, jnp.float32),
            )

            episode_return = carry.episode_return + reward
            episode_length = carry.episode_length + 1.0
            completed = done
            step_metrics = dict(
                reward=reward,
                done=done_f,
                skill_on=skill_on.astype(jnp.float32),
                completed_return=jnp.where(completed, episode_return, 0.0),
                completed_length=jnp.where(completed, episode_length, 0.0),
                completed=completed.astype(jnp.float32),
            )
            if achievement_keys:
                # Achievements/* are persistent 0/100 per-episode unlocked flags (NOT
                # cumulative counts); the runner reduces (flag>0) over completed
                # episodes into the standard craftax metrics. Only done rows are
                # meaningful (masked downstream).
                step_metrics["achievements"] = jnp.stack(
                    [jnp.asarray(next_env_state.info[a], jnp.float32) for a in achievement_keys],
                    axis=-1,
                )

            # Reward phases still end at the switch; memory optionally continues.
            commanded_boundary = jnp.linalg.norm(next_skills - carry.skills, axis=-1) > zero_eps
            # Configs written before this flag existed reset memory at the switch.
            memory_boundary = (jnp.zeros_like(commanded_boundary)
                               if agent.config.get("pqn_preserve_memory_at_switch", False)
                               else commanded_boundary)
            next_recurrent_reset = (done | memory_boundary).astype(jnp.float32)

            # Auto-reset finished lanes.
            def reset_finished(state: Any) -> Any:
                reset_keys = jax.random.split(reset_key, num_envs)
                return reset_finished_env(env, state, reset_keys)

            env_state = jax.lax.cond(
                jnp.any(done), reset_finished, lambda state: state, next_env_state
            )
            # On done: reset the skill (zero prefix unless potential_steps==0), anchor, episode id.
            reset_skills = sampled_skills if int(potential_steps) == 0 else jnp.zeros_like(sampled_skills)
            if has_lanes:
                new_potential_lane = _draw_potential_lanes(reset_key, num_envs, potential_lane_fraction)
                potential_lane = jnp.where(done, new_potential_lane, carry.potential_lane)
                reset_skills = jnp.where(new_potential_lane[:, None], jnp.zeros_like(reset_skills), reset_skills)
            else:
                potential_lane = None
            skills = jnp.where(done[:, None], reset_skills, next_skills)
            anchor_observations = jnp.where(done[:, None], env_state.obs, transition_next_anchors)
            reset_offset = jnp.cumsum(done.astype(jnp.int32)) - 1
            new_ids = carry.next_episode_id + reset_offset
            episode_ids = jnp.where(done, new_ids, carry.episode_ids)
            next_episode_id = carry.next_episode_id + jnp.sum(done.astype(jnp.int32))

            next_carry = PotentialPQNCollectorState(
                env_state=env_state,
                recurrent_state=next_recurrent_state,
                recurrent_last_done=next_recurrent_reset,
                skills=skills,
                anchor_observations=anchor_observations,
                episode_ids=episode_ids,
                next_episode_id=next_episode_id,
                episode_return=jnp.where(done, 0.0, episode_return),
                episode_length=jnp.where(done, 0.0, episode_length),
                rng=rng,
                potential_lane=potential_lane,
            )
            return next_carry, (transition, step_metrics)

        carry, (transitions, metrics) = jax.lax.scan(step_fn, carry, xs=None, length=chunk_steps)

        def flatten(value):
            value = jnp.asarray(value)
            return value.reshape((chunk_steps * num_envs,) + value.shape[2:])

        transitions = {key: flatten(value) for key, value in transitions.items()}
        metrics = {key: flatten(value) for key, value in metrics.items()}
        return carry, transitions, metrics

    return collect
