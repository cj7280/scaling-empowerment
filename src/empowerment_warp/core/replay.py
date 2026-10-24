"""Functional device-resident circular replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flax import struct
import jax
import jax.numpy as jnp


GEOMETRIC_ANCHOR_REPLAY_MODE = "legacy_geometric"
POTENTIAL_REPLAY_MODES = frozenset(
    {GEOMETRIC_ANCHOR_REPLAY_MODE}
)
TWO_PHASE_SKILL_START_MODE = "two_phase"
MARGINAL_BRANCH_SKILL_START_MODE = "marginal_branch"
SKILL_START_MODES = frozenset(
    {TWO_PHASE_SKILL_START_MODE, MARGINAL_BRANCH_SKILL_START_MODE}
)


@struct.dataclass
class Transition:
    observations: jax.Array
    anchor_observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    masks: jax.Array
    next_observations: jax.Array
    next_anchor_observations: jax.Array
    dones: jax.Array
    skills: jax.Array
    next_skills: jax.Array
    loss_weights: jax.Array
    episode_ids: jax.Array
    timesteps: jax.Array


@struct.dataclass
class RepresentationTransition:
    """Compact encoder archive row; policy-only transition fields stay out."""

    observations: jax.Array
    anchor_observations: jax.Array
    next_observations: jax.Array
    skills: jax.Array
    episode_ids: jax.Array
    timesteps: jax.Array
    dones: jax.Array


@struct.dataclass
class ReplayState:
    data: Any
    cursor: jax.Array
    size: jax.Array


@dataclass(frozen=True, slots=True)
class ReplayBuffer:
    capacity: int
    sequence_stride: int = 1

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("replay capacity must be positive")
        if self.sequence_stride <= 0:
            raise ValueError("sequence_stride must be positive")

    def initialize(self, example: Any) -> ReplayState:
        batch_size = int(example.observations.shape[0])
        if batch_size <= 0:
            raise ValueError("replay example must have a leading batch dimension")

        def empty(value: jax.Array) -> jax.Array:
            return jnp.zeros((self.capacity,) + value.shape[1:], dtype=value.dtype)

        return ReplayState(
            data=jax.tree_util.tree_map(empty, example),
            cursor=jnp.asarray(0, dtype=jnp.int32),
            size=jnp.asarray(0, dtype=jnp.int32),
        )

    def insert(self, state: ReplayState, batch: Any) -> ReplayState:
        batch_size = batch.observations.shape[0]
        if batch_size > self.capacity:
            raise ValueError(
                f"insert batch size {batch_size} exceeds replay capacity {self.capacity}"
            )
        indices = (state.cursor + jnp.arange(batch_size, dtype=jnp.int32)) % self.capacity
        data = jax.tree_util.tree_map(lambda target, values: target.at[indices].set(values), state.data, batch)
        return ReplayState(
            data=data,
            cursor=(state.cursor + batch_size) % self.capacity,
            size=jnp.minimum(self.capacity, state.size + batch_size),
        )

    def insert_masked(self, state: ReplayState, batch: Any, mask: jax.Array) -> ReplayState:
        """Insert only selected rows while preserving their chronological order."""

        batch_size = batch.observations.shape[0]
        if batch_size > self.capacity:
            raise ValueError(
                f"insert batch size {batch_size} exceeds replay capacity {self.capacity}"
            )
        mask = jnp.asarray(mask, dtype=jnp.bool_).reshape((batch_size,))
        selected = jnp.nonzero(mask, size=batch_size, fill_value=0)[0]
        count = jnp.sum(mask.astype(jnp.int32))
        valid = jnp.arange(batch_size, dtype=jnp.int32) < count
        indices = (state.cursor + jnp.arange(batch_size, dtype=jnp.int32)) % self.capacity
        indices = jnp.where(valid, indices, self.capacity)

        def scatter(target: jax.Array, values: jax.Array) -> jax.Array:
            return target.at[indices].set(values[selected], mode="drop")

        data = jax.tree_util.tree_map(scatter, state.data, batch)
        return ReplayState(
            data=data,
            cursor=(state.cursor + count) % self.capacity,
            size=jnp.minimum(self.capacity, state.size + count),
        )

    def sample(
        self,
        state: ReplayState,
        key: jax.Array,
        batch_size: int,
    ) -> Any:
        if batch_size <= 0:
            raise ValueError("sample batch_size must be positive")
        indices = self.sample_indices(state, key, batch_size)
        return self.get(state, indices)

    def sample_indices(self, state: ReplayState, key: jax.Array, batch_size: int) -> jax.Array:
        if batch_size <= 0:
            raise ValueError("sample batch_size must be positive")
        return jax.random.randint(key, (batch_size,), 0, state.size)

    @staticmethod
    def get(state: ReplayState, indices: jax.Array) -> Any:
        return jax.tree_util.tree_map(lambda value: value[indices], state.data)

    def get_sequence(self, state: ReplayState, indices: jax.Array, length: int):
        """Read [time, batch] rows and valid links within one executed phase.

        Stop at the newest row, resets, phase switches, or changed skill context.
        A broken link truncates a return; it does not remove its last bootstrap.
        """
        if length < 1:
            raise ValueError("sequence length must be positive")
        oldest = jnp.where(state.size < self.capacity, 0, state.cursor)
        positions = (indices - oldest) % self.capacity
        offsets = jnp.arange(length, dtype=jnp.int32)[:, None] * self.sequence_stride
        available = positions[None, :] + offsets < state.size
        sequence_indices = jnp.where(available, (indices[None, :] + offsets) % self.capacity,
                                     indices[None, :])
        rows = self.get(state, sequence_indices)
        skill_on = jnp.linalg.norm(rows.skills, axis=-1) > 1e-6
        same_command = jnp.all(rows.skills[1:] == rows.skills[:-1], axis=-1)
        same_anchor = jnp.all(rows.anchor_observations[1:] == rows.anchor_observations[:-1], axis=-1)
        links = (available[1:] & available[:-1] & ~rows.dones[:-1]
                 & (rows.masks[:-1] > 0)
                 & (rows.episode_ids[1:] == rows.episode_ids[:-1])
                 & (rows.timesteps[1:] == rows.timesteps[:-1] + 1)
                 & same_command & (~skill_on[:-1] | same_anchor))
        return rows, links


@struct.dataclass
class EpisodeArchiveState:
    staging: Any        # (num_envs, episode_length, ...): each lane's in-flight episode by timestep
    genuine: jax.Array  # (num_envs,) the staged episode started at timestep 0 (not a staggered stub)
    episodes: Any       # (num_episodes, episode_length, ...): completed episodes, FIFO
    lengths: jax.Array  # (num_episodes,) rows stored per slot; rows at or past it are stale
    cursor: jax.Array
    size: jax.Array
    active_rows: jax.Array  # skill-on rows across stored slots


@struct.dataclass
class BranchTrajectoryArchiveState:
    anchors: jax.Array
    skills: jax.Array
    outcomes: jax.Array
    lengths: jax.Array
    cursor: jax.Array
    size: jax.Array
    active_rows: jax.Array
    fresh_group_count: jax.Array


@struct.dataclass
class BranchTrajectory:
    """One independently commanded continuation from each marginal anchor."""

    anchors: jax.Array
    skills: jax.Array
    outcomes: jax.Array
    valid: jax.Array


@dataclass(frozen=True, slots=True)
class EpisodeArchive:
    """A bag of the most recent completed episodes, whichever lane produced them.

    This is the skill buffer under vectorization: whole segments,
    futures resolved inside the stored episode, capacity counted in episodes
    rather than in rows divided by lanes.
    """

    num_envs: int
    num_episodes: int
    episode_length: int

    def __post_init__(self) -> None:
        if min(self.num_envs, self.num_episodes, self.episode_length) <= 0:
            raise ValueError("episode archive needs positive num_envs, num_episodes, episode_length")

    def initialize(self, example: Any) -> EpisodeArchiveState:
        # Validate one row per env off any leaf (SAC rows carry `observations`; the PQN
        # encoder row stores obs bit-packed as obs_map/obs_tail instead).
        leading = int(jax.tree_util.tree_leaves(example)[0].shape[0])
        if leading != self.num_envs:
            raise ValueError("episode archive example must hold one row per environment")

        def staging(value: jax.Array) -> jax.Array:
            return jnp.zeros((self.num_envs, self.episode_length) + value.shape[1:], dtype=value.dtype)

        def episodes(value: jax.Array) -> jax.Array:
            return jnp.zeros((self.num_episodes, self.episode_length) + value.shape[1:], dtype=value.dtype)

        return EpisodeArchiveState(
            staging=jax.tree_util.tree_map(staging, example),
            genuine=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
            episodes=jax.tree_util.tree_map(episodes, example),
            lengths=jnp.zeros((self.num_episodes,), dtype=jnp.int32),
            cursor=jnp.asarray(0, dtype=jnp.int32),
            size=jnp.asarray(0, dtype=jnp.int32),
            active_rows=jnp.asarray(0, dtype=jnp.int32),
        )

    @staticmethod
    def _skill_on(skills: jax.Array) -> jax.Array:
        return jnp.linalg.norm(skills, axis=-1) > 1e-6

    def insert(self, state: EpisodeArchiveState, batch: Any) -> EpisodeArchiveState:
        """Stage a step-major chunk; lanes whose episode ends are copied into the bag."""

        batch_size = int(jax.tree_util.tree_leaves(batch)[0].shape[0])
        if batch_size % self.num_envs:
            raise ValueError("episode archive inserts must contain complete vector steps")
        chunk_steps = batch_size // self.num_envs
        shaped = jax.tree_util.tree_map(
            lambda value: value.reshape((chunk_steps, self.num_envs) + value.shape[1:]),
            batch,
        )
        lanes = jnp.arange(self.num_envs, dtype=jnp.int32)
        slot_stride = jnp.arange(self.episode_length, dtype=jnp.int32)

        def one_step(carry: EpisodeArchiveState, rows: Any) -> tuple[EpisodeArchiveState, None]:
            timesteps = jnp.clip(rows.timesteps, 0, self.episode_length - 1)
            staging = jax.tree_util.tree_map(
                lambda target, value: target.at[lanes, timesteps].set(value), carry.staging, rows
            )
            # Staggered resets seed lanes mid-episode with a fresh-reset state; those
            # stub episodes never started at timestep 0 and are not archived.
            genuine = carry.genuine | (rows.timesteps == 0)
            done = jnp.asarray(rows.dones, dtype=jnp.bool_) & genuine
            rank = jnp.cumsum(done.astype(jnp.int32)) - 1
            # More finishers than slots in one step cannot all be kept; extras are dropped.
            keep = done & (rank < self.num_episodes)
            slots = jnp.where(keep, (carry.cursor + rank) % self.num_episodes, self.num_episodes)
            lengths = timesteps + 1
            new_active = jnp.sum(
                (slot_stride[None, :] < lengths[:, None]) & self._skill_on(staging.skills),
                axis=-1,
            )
            old_lengths = carry.lengths[jnp.minimum(slots, self.num_episodes - 1)]
            old_active = jnp.sum(
                (slot_stride[None, :] < old_lengths[:, None])
                & self._skill_on(carry.episodes.skills[jnp.minimum(slots, self.num_episodes - 1)]),
                axis=-1,
            )
            delta_active = jnp.sum(jnp.where(keep, new_active - old_active, 0))
            episodes = jax.tree_util.tree_map(
                lambda target, value: target.at[slots].set(value, mode="drop"),
                carry.episodes,
                staging,
            )
            new_lengths = carry.lengths.at[slots].set(lengths, mode="drop")
            count = jnp.sum(keep.astype(jnp.int32))
            return EpisodeArchiveState(
                staging=staging,
                genuine=genuine & ~jnp.asarray(rows.dones, dtype=jnp.bool_),
                episodes=episodes,
                lengths=new_lengths,
                cursor=(carry.cursor + count) % self.num_episodes,
                size=jnp.minimum(self.num_episodes, carry.size + count),
                active_rows=carry.active_rows + delta_active,
            ), None

        next_state, _ = jax.lax.scan(one_step, state, shaped)
        return next_state

    def sample_active(
        self,
        state: EpisodeArchiveState,
        key: jax.Array,
        batch_size: int,
        *,
        candidates_per_row: int = 2,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Row-uniform (slot, row) over stored skill-on rows; `valid` flags a shortfall."""

        if batch_size <= 0 or candidates_per_row <= 0:
            raise ValueError("batch_size and candidates_per_row must be positive")
        slot_key, row_key = jax.random.split(key)
        count = batch_size * candidates_per_row
        slots = jax.random.randint(slot_key, (count,), 0, jnp.maximum(state.size, 1))
        rows = jax.random.randint(row_key, (count,), 0, self.episode_length)
        active = (
            (state.size > 0)
            & (rows < state.lengths[slots])
            & self._skill_on(state.episodes.skills[slots, rows])
        )
        order = jnp.nonzero(active, size=batch_size, fill_value=0)[0]
        valid = jnp.arange(batch_size, dtype=jnp.int32) < jnp.sum(active.astype(jnp.int32))
        return slots[order], rows[order], valid

    @staticmethod
    def get(state: EpisodeArchiveState, slots: jax.Array, rows: jax.Array) -> Any:
        return jax.tree_util.tree_map(lambda value: value[slots, rows], state.episodes)


@dataclass(frozen=True, slots=True)
class BranchTrajectoryArchive:
    """Ring of marginal anchors and their fixed-skill continuation trajectories."""

    num_groups: int
    horizon: int

    def __post_init__(self) -> None:
        if self.num_groups <= 0 or self.horizon <= 0:
            raise ValueError("branch trajectory archive sizes must be positive")

    def initialize(
        self,
        anchor_example: jax.Array,
        skill_example: jax.Array,
    ) -> BranchTrajectoryArchiveState:
        anchor_shape = tuple(anchor_example.shape[1:])
        skill_shape = tuple(skill_example.shape[1:])
        return BranchTrajectoryArchiveState(
            anchors=jnp.zeros(
                (self.num_groups,) + anchor_shape,
                dtype=anchor_example.dtype,
            ),
            skills=jnp.zeros(
                (self.num_groups,) + skill_shape,
                dtype=skill_example.dtype,
            ),
            outcomes=jnp.zeros(
                (self.num_groups, self.horizon) + anchor_shape,
                dtype=anchor_example.dtype,
            ),
            lengths=jnp.zeros((self.num_groups,), dtype=jnp.int32),
            cursor=jnp.asarray(0, dtype=jnp.int32),
            size=jnp.asarray(0, dtype=jnp.int32),
            active_rows=jnp.asarray(0, dtype=jnp.int32),
            fresh_group_count=jnp.asarray(0, dtype=jnp.int32),
        )

    def insert(
        self,
        state: BranchTrajectoryArchiveState,
        batch: BranchTrajectory,
    ) -> BranchTrajectoryArchiveState:
        return self.insert_prefix(
            state,
            batch,
            jnp.asarray(batch.anchors.shape[0], dtype=jnp.int32),
        )

    def insert_prefix(
        self,
        state: BranchTrajectoryArchiveState,
        batch: BranchTrajectory,
        group_count: jax.Array,
    ) -> BranchTrajectoryArchiveState:
        """Insert a valid prefix from a fixed-width branch batch."""

        batch_group_count = int(batch.anchors.shape[0])
        if batch_group_count > self.num_groups:
            raise ValueError(
                f"branch group batch {batch_group_count} exceeds archive capacity {self.num_groups}"
            )
        if tuple(batch.outcomes.shape[:2]) != (self.horizon, batch_group_count):
            raise ValueError(
                "branch outcomes must have shape "
                f"({self.horizon}, {batch_group_count}, ...), got {batch.outcomes.shape}"
            )
        valid = jnp.asarray(batch.valid, dtype=jnp.bool_)
        if tuple(valid.shape) != (self.horizon, batch_group_count):
            raise ValueError(
                "branch valid mask must have shape "
                f"{(self.horizon, batch_group_count)}, got {valid.shape}"
            )
        insert_count = jnp.clip(
            jnp.asarray(group_count, dtype=jnp.int32),
            0,
            batch_group_count,
        )
        batch_rows = jnp.arange(batch_group_count, dtype=jnp.int32)
        insert_mask = batch_rows < insert_count
        indices = (state.cursor + batch_rows) % self.num_groups
        lengths = jnp.sum(valid.astype(jnp.int32), axis=0)
        old_lengths = state.lengths[indices]
        outcomes = jnp.swapaxes(batch.outcomes, 0, 1)

        def masked_values(old: jax.Array, new: jax.Array) -> jax.Array:
            mask = insert_mask.reshape(
                insert_mask.shape + (1,) * (new.ndim - insert_mask.ndim)
            )
            return jnp.where(mask, new, old)

        return BranchTrajectoryArchiveState(
            anchors=state.anchors.at[indices].set(
                masked_values(state.anchors[indices], batch.anchors)
            ),
            skills=state.skills.at[indices].set(
                masked_values(state.skills[indices], batch.skills)
            ),
            outcomes=state.outcomes.at[indices].set(
                masked_values(state.outcomes[indices], outcomes)
            ),
            lengths=state.lengths.at[indices].set(
                jnp.where(insert_mask, lengths, old_lengths)
            ),
            cursor=(state.cursor + insert_count) % self.num_groups,
            size=jnp.minimum(self.num_groups, state.size + insert_count),
            active_rows=state.active_rows
            + jnp.sum(jnp.where(insert_mask, lengths - old_lengths, 0)),
            fresh_group_count=jnp.where(
                insert_count > 0,
                insert_count,
                state.fresh_group_count,
            ),
        )

    def sample_active(
        self,
        state: BranchTrajectoryArchiveState,
        key: jax.Array,
        batch_size: int,
        *,
        candidates_per_row: int = 2,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample groups and a virtual single-lane row offset for horizon matching."""

        if batch_size <= 0 or candidates_per_row <= 0:
            raise ValueError("batch_size and candidates_per_row must be positive")
        group_key, row_key = jax.random.split(key)
        count = batch_size * candidates_per_row
        groups = jax.random.randint(
            group_key,
            (count,),
            0,
            jnp.maximum(state.size, 1),
        )
        rows = jax.random.randint(row_key, (count,), 0, self.horizon)
        active = (state.size > 0) & (rows < state.lengths[groups])
        order = jnp.nonzero(active, size=batch_size, fill_value=0)[0]
        valid = jnp.arange(batch_size, dtype=jnp.int32) < jnp.sum(
            active.astype(jnp.int32)
        )
        return groups[order], rows[order], valid
