"""Batched BuilderBench Corner3 dynamics on MJX-Warp."""

from __future__ import annotations

from itertools import permutations
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from mujoco.mjx.third_party.warp._src.jax.ffi import GraphMode

from empowerment_warp.envs.builderbench.config import BuilderBenchConfig
from empowerment_warp.envs.builderbench.model import ModelBundle, build_model


class Metrics(NamedTuple):
    success: jax.Array
    easy_success: jax.Array
    out_of_bounds: jax.Array
    obj_lifted: jax.Array
    obj_moved: jax.Array
    obj_goal_dist: jax.Array
    max_cube_penetration: jax.Array


class State(NamedTuple):
    data: Any
    ctrl: jax.Array
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    step_count: jax.Array
    rng: jax.Array
    initial_object_pos: jax.Array
    target_goal: jax.Array
    metrics: Metrics
    cube_density: jax.Array
    density_choice_pending: jax.Array


def _quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = jnp.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = jnp.moveaxis(q2, -1, 0)
    return jnp.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _graph_mode(name: str) -> GraphMode:
    modes = {
        "none": GraphMode.NONE,
        "warp": GraphMode.WARP,
        "warp_staged": GraphMode.WARP_STAGED,
        "warp_staged_ex": GraphMode.WARP_STAGED_EX,
    }
    try:
        return modes[name]
    except KeyError as exc:
        raise ValueError(f"unsupported Warp graph mode: {name}") from exc


class BuilderBenchWarpEnv:
    """Corner3 with explicit batched resets and no CPU transition callback."""

    action_size = 5
    num_cubes = 3
    cube_half_extent = 0.02

    def __init__(self, config: BuilderBenchConfig, *, _physics_impl: str = "warp"):
        if _physics_impl not in {"warp", "jax"}:
            raise ValueError(f"unsupported MJX implementation: {_physics_impl}")
        self.config = config
        self._physics_impl = _physics_impl
        self.action_size = 6 if config.density_choice else 5
        self.bundle: ModelBundle = build_model(config)
        put_model_kwargs = {"impl": _physics_impl}
        if _physics_impl == "warp":
            put_model_kwargs["graph_mode"] = _graph_mode(config.graph_mode)
        self.model = mjx.put_model(self.bundle.model, **put_model_kwargs)
        cube_bodies = np.asarray([self.bundle.model.body(f"block_{i}").id for i in range(3)])
        # These independent, unactuated free bodies have no descendants. Their
        # mass-derived constants scale exactly with their shared density.
        if np.any(self.bundle.model.body_parentid[cube_bodies] != 0):
            raise ValueError("density scaling requires independent world-attached cubes")
        self._cube_body_mask = jnp.asarray(np.isin(np.arange(self.model.nbody), cube_bodies))
        self._cube_dof_mask = jnp.asarray(np.isin(self.bundle.model.dof_bodyid, cube_bodies))
        self._cube_mass_total = jnp.sum(jnp.where(self._cube_body_mask, self.model.body_mass, 0.0))
        self._cube_inertia_mean = jnp.sum(jnp.where(self._cube_dof_mask, self.model.dof_M0, 0.0)) / self.model.nv
        naconmax = config.naconmax if _physics_impl == "warp" else config.naconmax_per_env
        # naconmax is batch-total in MJX-Warp, while njmax is per world.
        njmax = config.njmax_per_env
        self._data_template = mjx.make_data(
            self.bundle.model,
            impl=_physics_impl,
            naconmax=naconmax,
            njmax=njmax,
        )

        self._init_qpos = jnp.asarray(self.bundle.init_qpos)
        self._init_qvel = jnp.asarray(self.bundle.init_qvel)
        self._init_ctrl = jnp.asarray(self.bundle.init_ctrl)
        self._object_qpos_adr = jnp.asarray(self.bundle.object_qpos_adr)
        self._object_qvel_adr = jnp.asarray(self.bundle.object_qvel_adr)
        self._finger_qpos_adr = jnp.asarray(self.bundle.finger_qpos_adr)
        self._mocap_ids = jnp.asarray(self.bundle.mocap_ids)
        self._target_goal_offsets = jnp.asarray(self.bundle.target_goal_offsets)
        self._workspace_bounds = jnp.asarray(self.bundle.workspace_bounds)
        self._ctrl_bounds = jnp.asarray(self.bundle.ctrl_bounds)
        self._action_scale = jnp.asarray([255.0, 0.05, 0.05, 0.05, 0.05])
        self._gripper_quat_inverse = jnp.asarray([0.0, 0.0, -1.0, 0.0])
        self._assignment_permutations = jnp.asarray(tuple(permutations(range(3))), dtype=jnp.int32)

        self._reset_batched = jax.jit(jax.vmap(self._reset_one))
        self._step_batched = jax.jit(jax.vmap(self._step_one))

    @property
    def observation_size(self) -> int:
        return 53 if self.config.density_choice else 51

    def model_for_density(self, density: jax.Array) -> Any:
        """Resolve one lane's mass, inertia and solver constants on device."""
        if not self.config.density_choice:
            return self.model
        ratio = density / self.config.cube_density
        body_scale = jnp.where(self._cube_body_mask, ratio, 1.0)
        dof_scale = jnp.where(self._cube_dof_mask, ratio, 1.0)
        subtree = self.model.body_subtreemass * body_scale
        subtree = subtree.at[0].add((ratio - 1.0) * self._cube_mass_total)
        return self.model.replace(
            body_mass=self.model.body_mass * body_scale,
            body_inertia=self.model.body_inertia * body_scale[:, None],
            body_subtreemass=subtree,
            body_invweight0=self.model.body_invweight0 / body_scale[:, None],
            dof_invweight0=self.model.dof_invweight0 / dof_scale,
            dof_M0=self.model.dof_M0 * dof_scale,
            stat=self.model.stat.replace(
                meaninertia=self.model.stat.meaninertia + (ratio - 1.0) * self._cube_inertia_mean
            ),
        )

    def _density_observation(self, data: Any, density: jax.Array, pending: jax.Array) -> jax.Array:
        obs = self._observation(data)
        if self.config.density_choice:
            normalized = 2.0 * jnp.log(density / self.config.density_min) / jnp.log(
                self.config.density_max / self.config.density_min
            ) - 1.0
            obs = jnp.concatenate((obs, jnp.stack((normalized, pending.astype(jnp.float32)))))
        return obs

    @classmethod
    def encoder_observation_indices(cls, mode: str) -> tuple[int, ...] | None:
        """Return the encoder's intrinsic view without changing policy observations."""

        if mode == "auto":
            mode = "object_pos_effector"
        cube_positions = tuple(range(10, 10 + 3 * cls.num_cubes))
        views = {
            "full": None,
            "object_pos": cube_positions,
            "object_pos_effector": tuple(range(3)) + cube_positions,
            "object_z": tuple(10 + 3 * cube + 2 for cube in range(cls.num_cubes)),
        }
        if mode not in views:
            choices = ", ".join(sorted(views))
            raise ValueError(f"unsupported BuilderBench encoder_obs_mode {mode!r}; choose {choices}")
        return views[mode]

    def _object_positions(self, data: Any) -> jax.Array:
        return data.qpos[self._object_qpos_adr[:, None] + jnp.arange(3)]

    def _object_quaternions(self, data: Any) -> jax.Array:
        return data.qpos[(self._object_qpos_adr + 3)[:, None] + jnp.arange(4)]

    def _object_linear_velocities(self, data: Any) -> jax.Array:
        return data.qvel[self._object_qvel_adr[:, None] + jnp.arange(3)]

    def _object_angular_velocities(self, data: Any) -> jax.Array:
        return data.qvel[(self._object_qvel_adr + 3)[:, None] + jnp.arange(3)]

    def _observation(self, data: Any) -> jax.Array:
        gripper_pos = data.sensordata[self.bundle.gripper_pos_slice]
        gripper_quat = _quat_mul(
            data.sensordata[self.bundle.gripper_quat_slice],
            self._gripper_quat_inverse,
        )
        gripper_linvel = data.sensordata[self.bundle.gripper_linvel_slice]
        return jnp.concatenate(
            (
                gripper_pos,
                gripper_quat,
                gripper_linvel,
                self._object_positions(data).reshape(-1),
                self._object_quaternions(data).reshape(-1),
                self._object_linear_velocities(data).reshape(-1),
                self._object_angular_velocities(data).reshape(-1),
                data.qpos[self._finger_qpos_adr],
            )
        )

    def _reward_metrics(
        self,
        data: Any,
        initial_object_pos: jax.Array,
        target_goal: jax.Array,
        wall_hit: jax.Array,
    ) -> tuple[jax.Array, Metrics]:
        object_pos = self._object_positions(data)
        if self.config.permutation_invariant_reward:
            pairwise_sq = jnp.sum((target_goal[:, None, :] - object_pos[None, :, :]) ** 2, axis=-1)
            target_ids = jnp.arange(3)
            permutation_cost = jnp.sum(pairwise_sq[target_ids, self._assignment_permutations], axis=-1)
            assignment = self._assignment_permutations[jnp.argmin(permutation_cost)]
            position_error = jnp.sqrt(pairwise_sq[target_ids, assignment])
        else:
            position_error = jnp.linalg.norm(target_goal - object_pos, axis=-1)

        gripper_pos = data.sensordata[self.bundle.gripper_pos_slice]
        lift_height = object_pos[:, 2] - initial_object_pos[:, 2]
        gripper_distance = jnp.linalg.norm(object_pos - gripper_pos[None, :], axis=-1)
        lifted = jnp.any(
            (object_pos[:, 2] > 0.05)
            & (lift_height > 0.015)
            & (gripper_distance < self.cube_half_extent * 3.0)
        )
        moved = jnp.any(jnp.linalg.norm(object_pos - initial_object_pos, axis=-1) > 0.01)
        penetration = jnp.maximum(0.0, self.cube_half_extent - object_pos[:, 2])

        reward = jnp.sum(1.0 - jnp.tanh(5.0 * position_error))
        metrics = Metrics(
            success=jnp.all(position_error < 0.02).astype(jnp.float32),
            easy_success=jnp.all(position_error < 0.05).astype(jnp.float32),
            out_of_bounds=wall_hit.astype(jnp.float32),
            obj_lifted=lifted.astype(jnp.float32),
            obj_moved=moved.astype(jnp.float32),
            obj_goal_dist=jnp.sum(position_error),
            max_cube_penetration=jnp.max(penetration),
        )
        return reward, metrics

    def _reset_one(self, key: jax.Array) -> State:
        key, target_key = jax.random.split(key)
        anchor = jnp.asarray([self.config.corner_x, self.config.corner_y])
        offsets = jnp.asarray(
            [[0.0, 0.0], [-self.config.corner_spacing, 0.0], [0.0, -self.config.corner_spacing]]
        )
        xy_low = self._workspace_bounds[0, :2] + self.cube_half_extent
        xy_high = self._workspace_bounds[1, :2] - self.cube_half_extent
        object_xy = jnp.clip(jnp.clip(anchor, xy_low, xy_high) + offsets, xy_low, xy_high)
        object_pos = jnp.concatenate((object_xy, jnp.full((3, 1), 0.02)), axis=-1)
        if self.config.centered_start:
            object_pos = jnp.asarray([[0.2, -0.055, 0.02], [0.2, 0.0, 0.02], [0.2, 0.055, 0.02]])

        target_base = jax.random.uniform(
            target_key,
            (1, 3),
            minval=jnp.asarray([0.22, -0.1, 0.02]),
            maxval=jnp.asarray([0.32, 0.1, 0.02]),
        )
        target_goal = target_base + self._target_goal_offsets

        qpos = self._init_qpos
        qpos = qpos.at[self._object_qpos_adr[:, None] + jnp.arange(3)].set(object_pos)
        qpos = qpos.at[(self._object_qpos_adr + 3)[:, None] + jnp.arange(4)].set(
            jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0]), (3, 1))
        )
        mocap_pos = self._data_template.mocap_pos.at[self._mocap_ids].set(target_goal)
        mocap_quat = self._data_template.mocap_quat.at[self._mocap_ids].set(
            jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0]), (3, 1))
        )
        data = self._data_template.replace(
            qpos=qpos,
            qvel=self._init_qvel,
            ctrl=self._init_ctrl,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
        )
        data = mjx.forward(self.model, data)
        reward, metrics = self._reward_metrics(data, object_pos, target_goal, jnp.asarray(False))
        density = jnp.asarray(self.config.cube_density, dtype=jnp.float32)
        pending = jnp.asarray(self.config.density_choice)
        return State(
            data=data,
            ctrl=self._init_ctrl,
            obs=self._density_observation(data, density, pending),
            reward=reward,
            terminated=jnp.asarray(False),
            truncated=jnp.asarray(False),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            rng=key,
            initial_object_pos=object_pos,
            target_goal=target_goal,
            metrics=metrics,
            cube_density=density,
            density_choice_pending=pending,
        )

    def _apply_wall_bounce(self, data: Any) -> tuple[Any, jax.Array]:
        object_pos = self._object_positions(data)
        object_linvel = self._object_linear_velocities(data)
        xy_low = self._workspace_bounds[0, :2] + self.cube_half_extent
        xy_high = self._workspace_bounds[1, :2] - self.cube_half_extent
        hit_mask = (object_pos[:, :2] < xy_low) | (object_pos[:, :2] > xy_high)
        object_pos = object_pos.at[:, :2].set(jnp.clip(object_pos[:, :2], xy_low, xy_high))
        object_linvel = object_linvel.at[:, :2].set(
            jnp.where(hit_mask, -self.config.wall_restitution * object_linvel[:, :2], object_linvel[:, :2])
        )
        qpos = data.qpos.at[self._object_qpos_adr[:, None] + jnp.arange(3)].set(object_pos)
        qvel = data.qvel.at[self._object_qvel_adr[:, None] + jnp.arange(3)].set(object_linvel)
        return data.replace(qpos=qpos, qvel=qvel), jnp.any(hit_mask)

    def _step_one(self, state: State, action: jax.Array) -> State:
        pending = state.density_choice_pending
        density = state.cube_density
        if self.config.density_choice:
            unit = (jnp.clip(action[-1], -1.0, 1.0) + 1.0) * 0.5
            chosen = self.config.density_min * jnp.exp(
                unit * jnp.log(self.config.density_max / self.config.density_min)
            )
            density = jnp.where(pending, chosen, density)
        model = self.model_for_density(density)
        ctrl = jnp.clip(
            state.ctrl + action[:5] * self._action_scale,
            self._ctrl_bounds[0],
            self._ctrl_bounds[1],
        )
        data = state.data.replace(ctrl=ctrl)
        data = jax.lax.fori_loop(0, self.config.substeps, lambda _, value: mjx.step(model, value), data)
        data, wall_hit = self._apply_wall_bounce(data)
        object_pos = self._object_positions(data)
        xy_low = self._workspace_bounds[0, :2] + self.cube_half_extent
        xy_high = self._workspace_bounds[1, :2] - self.cube_half_extent
        xy_out = jnp.any((object_pos[:, :2] < xy_low) | (object_pos[:, :2] > xy_high))
        z_out = jnp.any(
            (object_pos[:, 2] < self._workspace_bounds[0, 2])
            | (object_pos[:, 2] > self._workspace_bounds[1, 2])
        )
        nonfinite = ~jnp.all(jnp.isfinite(data.qpos)) | ~jnp.all(jnp.isfinite(data.qvel))
        terminated = (z_out | nonfinite) & self.config.early_termination
        step_count = state.step_count + 1
        truncated = step_count >= self.config.episode_length
        reward, metrics = self._reward_metrics(data, state.initial_object_pos, state.target_goal, wall_hit | xy_out)
        if self.config.density_choice:
            # Batched Warp lanes can be in different phases. Discard every
            # physical field for setup lanes, including solver warm-start data.
            data = data.where(pending, state.data)
            ctrl = jnp.where(pending, state.ctrl, ctrl)
            step_count = jnp.where(pending, state.step_count, step_count)
            terminated = jnp.where(pending, False, terminated)
            truncated = jnp.where(pending, False, truncated)
            reward = jnp.where(pending, 0.0, reward)
            metrics = Metrics(*(jnp.where(pending, old, new) for old, new in zip(state.metrics, metrics, strict=True)))
        return State(
            data=data,
            ctrl=ctrl,
            obs=self._density_observation(data, density, jnp.asarray(False)),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            step_count=step_count,
            rng=state.rng,
            initial_object_pos=state.initial_object_pos,
            target_goal=state.target_goal,
            metrics=metrics,
            cube_density=density,
            density_choice_pending=jnp.asarray(False),
        )

    def reset(self, keys: jax.Array) -> State:
        """Reset one environment per key; keys has shape [num_envs, 2]."""

        if keys.shape != (self.config.num_envs, 2):
            raise ValueError(f"expected reset keys with shape {(self.config.num_envs, 2)}, got {keys.shape}")
        return self._reset_batched(keys)

    def step(self, state: State, actions: jax.Array) -> State:
        """Advance every environment by one 10 ms control step."""

        if actions.shape != (self.config.num_envs, self.action_size):
            raise ValueError(
                f"expected actions with shape {(self.config.num_envs, self.action_size)}, got {actions.shape}"
            )
        return self._step_batched(state, actions)

    @staticmethod
    def where(mask: jax.Array, state: State, replacement: State) -> State:
        """Replace selected lanes, including every batched MJX-Warp data field (the branch
        collector's lane mask; same contract as the locomotion env)."""

        mask = jnp.asarray(mask, dtype=jnp.bool_)

        def choose(current: jax.Array, new: jax.Array) -> jax.Array:
            expanded = mask.reshape(mask.shape + (1,) * (current.ndim - mask.ndim))
            return jnp.where(expanded, new, current)

        data = jax.vmap(lambda current, new, selected: current.where(selected, new))(
            state.data, replacement.data, mask
        )
        metrics = Metrics(*(choose(x, y) for x, y in zip(state.metrics, replacement.metrics, strict=True)))
        rest = {
            f: choose(getattr(state, f), getattr(replacement, f))
            for f in State._fields if f not in ("data", "metrics")
        }
        return State(data=data, metrics=metrics, **rest)

    @staticmethod
    def gather(state: State, lane_indices: jax.Array) -> State:
        """Gather complete simulator lanes, including MJX-Warp internals."""

        from empowerment_warp.envs.state import gather_env_state_lanes

        return gather_env_state_lanes(state, state, lane_indices)

    @staticmethod
    def where_done(state: State, reset_state: State) -> State:
        """Select reset values only for terminated or truncated lanes."""

        done = state.terminated | state.truncated

        def choose(current: jax.Array, reset: jax.Array) -> jax.Array:
            mask = done.reshape(done.shape + (1,) * (current.ndim - done.ndim))
            return jnp.where(mask, reset, current)

        metrics = Metrics(*(choose(x, y) for x, y in zip(state.metrics, reset_state.metrics, strict=True)))
        data = jax.vmap(lambda current, reset, mask: current.where(mask, reset))(
            state.data,
            reset_state.data,
            done,
        )
        return State(
            data=data,
            ctrl=choose(state.ctrl, reset_state.ctrl),
            obs=choose(state.obs, reset_state.obs),
            reward=choose(state.reward, reset_state.reward),
            terminated=choose(state.terminated, reset_state.terminated),
            truncated=choose(state.truncated, reset_state.truncated),
            step_count=choose(state.step_count, reset_state.step_count),
            rng=choose(state.rng, reset_state.rng),
            initial_object_pos=choose(state.initial_object_pos, reset_state.initial_object_pos),
            target_goal=choose(state.target_goal, reset_state.target_goal),
            metrics=metrics,
            cube_density=choose(state.cube_density, reset_state.cube_density),
            density_choice_pending=choose(state.density_choice_pending, reset_state.density_choice_pending),
        )
