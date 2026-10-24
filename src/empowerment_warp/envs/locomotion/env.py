"""Batched Ant and AntSoccer transitions on MJX-Warp."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx.third_party.warp._src.jax.ffi import GraphMode

from empowerment_warp.envs.locomotion.config import AntConfig, AntSoccerConfig, SOCCER_TASKS
from empowerment_warp.envs.locomotion.model import arena_cell_xy, build_ant_model
from empowerment_warp.envs.state import gather_env_state_lanes


class LocomotionMetrics(NamedTuple):
    success: jax.Array
    agent_x: jax.Array
    agent_y: jax.Array
    ball_x: jax.Array
    ball_y: jax.Array
    agent_ball_distance: jax.Array
    ball_goal_distance: jax.Array


class State(NamedTuple):
    data: Any
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    step_count: jax.Array
    rng: jax.Array
    goal_xy: jax.Array
    metrics: LocomotionMetrics


def _graph_mode(name: str) -> GraphMode:
    return {
        "none": GraphMode.NONE,
        "warp": GraphMode.WARP,
        "warp_staged": GraphMode.WARP_STAGED,
        "warp_staged_ex": GraphMode.WARP_STAGED_EX,
    }[name]


class AntWarpEnv:
    """OGBench locomaze Ant with raw qpos/qvel observations."""

    action_size = 8
    soccer = False

    def __init__(self, config: AntConfig):
        self.config = config
        self.native_model = self._build_native_model(config)
        self.model = mjx.put_model(
            self.native_model,
            impl="warp",
            graph_mode=_graph_mode(config.graph_mode),
        )
        self._data_template = mjx.make_data(
            self.native_model,
            impl="warp",
            naconmax=config.naconmax,
            njmax=config.njmax_per_env,
        )
        self._qpos0 = jnp.asarray(self.native_model.qpos0)
        self._qvel0 = jnp.zeros((self.native_model.nv,), dtype=jnp.float32)
        self._ctrl_low = jnp.asarray(self.native_model.actuator_ctrlrange[:, 0])
        self._ctrl_high = jnp.asarray(self.native_model.actuator_ctrlrange[:, 1])
        self._reset_batched = jax.jit(jax.vmap(self._reset_one))
        self._step_batched = jax.jit(jax.vmap(self._step_one))

    def _build_native_model(self, config: AntConfig):
        return build_ant_model(config, soccer=self.soccer)

    @property
    def observation_size(self) -> int:
        return self.native_model.nq + self.native_model.nv

    def encoder_observation_indices(self, mode: str) -> tuple[int, ...] | None:
        if mode == "full":
            return None
        if mode == "agent_xy":
            # The ant's torso x-y (qpos[0:2]): the locomotion analogue of AntSoccer's `ball_xy`
            # view. With self-derived encoder labels (encoder_skill_source=achieved_discounted)
            # the full 42-state view lets the encoder define "skills" out of joint wiggles; this
            # view leaves only planar displacement for it to represent.
            return (0, 1)
        raise ValueError(f"Ant does not implement encoder_obs_mode={mode!r}")

    def _observation(self, data: Any) -> jax.Array:
        return jnp.concatenate((data.qpos, data.qvel))

    def _spawn(self, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        noise = self.config.reset_noise_scale
        qpos_key, qvel_key, next_key = jax.random.split(key, 3)
        qpos = self._qpos0 + jax.random.uniform(
            qpos_key,
            self._qpos0.shape,
            minval=-noise,
            maxval=noise,
        )
        qvel = self._qvel0 + noise * jax.random.normal(qvel_key, self._qvel0.shape)
        return qpos, qvel, jnp.zeros((2,), dtype=jnp.float32), next_key

    def _metrics(self, data: Any, goal_xy: jax.Array) -> LocomotionMetrics:
        agent_xy = data.qpos[:2]
        if self.soccer:
            ball_xy = data.qpos[-7:-5]
            ball_goal_distance = jnp.linalg.norm(ball_xy - goal_xy)
            agent_ball_distance = jnp.linalg.norm(agent_xy - ball_xy)
            success = ball_goal_distance <= self.config.goal_tolerance
        else:
            ball_xy = jnp.zeros((2,), dtype=data.qpos.dtype)
            ball_goal_distance = jnp.asarray(0.0, dtype=data.qpos.dtype)
            agent_ball_distance = jnp.asarray(0.0, dtype=data.qpos.dtype)
            success = jnp.asarray(False)
        return LocomotionMetrics(
            success=success.astype(jnp.float32),
            agent_x=agent_xy[0],
            agent_y=agent_xy[1],
            ball_x=ball_xy[0],
            ball_y=ball_xy[1],
            agent_ball_distance=agent_ball_distance,
            ball_goal_distance=ball_goal_distance,
        )

    def _reward(self, metrics: LocomotionMetrics) -> jax.Array:
        if self.soccer:
            return metrics.success - 1.0
        return jnp.asarray(0.0, dtype=jnp.float32)

    def _reset_one(self, key: jax.Array) -> State:
        qpos, qvel, goal_xy, next_key = self._spawn(key)
        data = self._data_template.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=jnp.zeros((self.action_size,), dtype=jnp.float32),
        )
        data = mjx.forward(self.model, data)
        metrics = self._metrics(data, goal_xy)
        return State(
            data=data,
            obs=self._observation(data),
            reward=self._reward(metrics),
            terminated=jnp.asarray(False),
            truncated=jnp.asarray(False),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            rng=next_key,
            goal_xy=goal_xy,
            metrics=metrics,
        )

    def _step_one(self, state: State, action: jax.Array) -> State:
        pre_metrics = self._metrics(state.data, state.goal_xy)
        ctrl = jnp.clip(action, self._ctrl_low, self._ctrl_high)
        data = state.data.replace(ctrl=ctrl)
        data = jax.lax.fori_loop(0, self.config.substeps, lambda _, value: mjx.step(self.model, value), data)
        finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
        terminated = (~finite) & self.config.terminate_on_nonfinite
        step_count = state.step_count + 1
        truncated = step_count >= self.config.episode_length
        metrics = self._metrics(data, state.goal_xy)
        if self.soccer and self.config.success_timing == "pre":
            metrics = metrics._replace(success=pre_metrics.success)
        return State(
            data=data,
            obs=self._observation(data),
            reward=self._reward(metrics),
            terminated=terminated,
            truncated=truncated,
            step_count=step_count,
            rng=state.rng,
            goal_xy=state.goal_xy,
            metrics=metrics,
        )

    def reset(self, keys: jax.Array) -> State:
        if keys.shape != (self.config.num_envs, 2):
            raise ValueError(f"expected reset keys with shape {(self.config.num_envs, 2)}, got {keys.shape}")
        return self._reset_batched(keys)

    def step(self, state: State, actions: jax.Array) -> State:
        if actions.shape != (self.config.num_envs, self.action_size):
            raise ValueError(
                f"expected actions with shape {(self.config.num_envs, self.action_size)}, got {actions.shape}"
            )
        return self._step_batched(state, actions)

    @staticmethod
    def gather(state: State, lane_indices: jax.Array) -> State:
        """Gather complete simulator lanes, including MJX-Warp internals."""

        return gather_env_state_lanes(state, state, lane_indices)

    @staticmethod
    def where(mask: jax.Array, state: State, replacement: State) -> State:
        """Replace selected lanes, including every batched MJX-Warp data field."""

        mask = jnp.asarray(mask, dtype=jnp.bool_)

        def choose(current: jax.Array, new: jax.Array) -> jax.Array:
            expanded = mask.reshape(mask.shape + (1,) * (current.ndim - mask.ndim))
            return jnp.where(expanded, new, current)

        data = jax.vmap(lambda current, new, selected: current.where(selected, new))(
            state.data,
            replacement.data,
            mask,
        )
        metrics = LocomotionMetrics(
            *(
                choose(current, new)
                for current, new in zip(
                    state.metrics,
                    replacement.metrics,
                    strict=True,
                )
            )
        )
        return State(
            data=data,
            obs=choose(state.obs, replacement.obs),
            reward=choose(state.reward, replacement.reward),
            terminated=choose(state.terminated, replacement.terminated),
            truncated=choose(state.truncated, replacement.truncated),
            step_count=choose(state.step_count, replacement.step_count),
            rng=choose(state.rng, replacement.rng),
            goal_xy=choose(state.goal_xy, replacement.goal_xy),
            metrics=metrics,
        )

    @classmethod
    def where_done(cls, state: State, reset_state: State) -> State:
        return cls.where(state.terminated | state.truncated, state, reset_state)


# How far past the arena center the standard-preset ball sits on the
# ant->ball line, in arena units (the maze cell is 4.0).
SOCCER_STANDARD_BALL_PAST_CENTER = 1.5


class AntSoccerWarpEnv(AntWarpEnv):
    """OGBench arena AntSoccer with task-4 and spawn-preset semantics."""

    soccer = True

    def __init__(self, config: AntSoccerConfig):
        super().__init__(config)

    def encoder_observation_indices(self, mode: str) -> tuple[int, ...] | None:
        if mode == "ball_xy":
            return (15, 16)
        return super().encoder_observation_indices(mode)

    def _spawn(self, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        config = self.config
        key, ant_key, agent_key, ball_key, goal_key = jax.random.split(key, 5)
        qpos, qvel, _, _ = super()._spawn(ant_key)
        agent_cell, ball_cell, goal_cell = SOCCER_TASKS[config.task_id - 1]
        agent_xy = jnp.asarray(arena_cell_xy(*agent_cell))
        ball_xy = jnp.asarray(arena_cell_xy(*ball_cell))
        goal_xy = jnp.asarray(arena_cell_xy(*goal_cell))
        radius = config.spawn_noise_scale
        agent_xy = agent_xy + jax.random.uniform(agent_key, (2,), minval=-radius, maxval=radius)
        ball_xy = ball_xy + jax.random.uniform(ball_key, (2,), minval=-radius, maxval=radius)
        if config.add_noise_to_goal:
            goal_xy = goal_xy + jax.random.uniform(goal_key, (2,), minval=-radius, maxval=radius)

        center_xy = jnp.asarray([10.0, 10.0], dtype=jnp.float32)
        if config.spawn_mode == "past_center_ball":
            # The ball waits just past the arena center on the ant->ball line
            # rather than at the far task cell, so the approach direction is
            # unchanged but the meeting point is central. It is a separate preset
            # because "standard" is shared with the base Ant comparison env.
            direction = ball_xy - agent_xy
            direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-6)
            ball_xy = center_xy + SOCCER_STANDARD_BALL_PAST_CENTER * direction
        elif config.spawn_mode == "closer_ball":
            agent_xy = 0.2 * agent_xy + 0.8 * center_xy
            ball_xy = center_xy
        elif config.spawn_mode == "center_ball":
            agent_xy = center_xy
            ball_xy = center_xy

        qpos = qpos.at[:2].set(agent_xy)
        qpos = qpos.at[-7:-5].set(ball_xy)
        return qpos, qvel, goal_xy, key
