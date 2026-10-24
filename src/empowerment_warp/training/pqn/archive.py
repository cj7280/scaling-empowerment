"""Dense device representation ring for live PQN or encoder-only branches.

Live mode admits every vector step immediately and selects commanded skill-on
rows with same-episode geometric futures. Branch mode admits completed sweeps,
keeps only the valid pre-terminal prefix eligible, and retains the fixed copied
physical anchor as the encoder's starting observation. Both modes use the same
geometric future sampler and time-major ring indexing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from empowerment_warp.agents.skill_classifier import SkillBatch
from empowerment_warp.core.replay_codecs import (
    CraftaxSymbolicObservationCodec,
    symbolic_codec_for_obs_dim,
)
from empowerment_warp.utils.sampling import truncated_geometric


@struct.dataclass
class EncoderArchiveRow:
    obs_map: jax.Array
    obs_tail: jax.Array
    skills: jax.Array
    episode_ids: jax.Array
    timesteps: jax.Array
    anchor_observations: jax.Array | None = None


@struct.dataclass
class PQNEncoderArchiveState:
    data: EncoderArchiveRow
    pointer: jax.Array
    size: jax.Array
    active_rows: jax.Array
    total_inserted_rows: jax.Array


class PQNEncoderArchive:
    def __init__(self, *, num_envs: int, capacity: int, episode_length: int,
                 future_gamma: float, max_future_steps: int = 0,
                 fixed_anchors: bool = False):
        if min(num_envs, capacity, episode_length) <= 0:
            raise ValueError("archive dimensions must be positive")
        if not 0.0 < future_gamma < 1.0 or max_future_steps < 0:
            raise ValueError("future_gamma must be in (0,1); max_future_steps >= 0")
        self.num_envs = int(num_envs)
        self.capacity = int(capacity)
        self.episode_length = int(episode_length)
        self.future_gamma = float(future_gamma)
        self.max_future_steps = int(max_future_steps)  # 0: reference episode/ring bounds only
        self.fixed_anchors = bool(fixed_anchors)
        self.codec: CraftaxSymbolicObservationCodec | None = None

    def initialize(self, num_envs: int, observation_dim: int, skill_dim: int):
        if num_envs != self.num_envs:
            raise ValueError("archive width must match the collector")
        self.codec = symbolic_codec_for_obs_dim(observation_dim)
        if self.codec is None:
            self.codec = CraftaxSymbolicObservationCodec(
                full_dim=int(observation_dim), tail_dim=int(observation_dim)
            )
        data = EncoderArchiveRow(
            obs_map=jnp.zeros((self.capacity, self.codec.packed_map_dim), jnp.uint8),
            obs_tail=jnp.zeros((self.capacity, self.codec.tail_dim), jnp.float32),
            skills=jnp.zeros((self.capacity, skill_dim), jnp.float32),
            episode_ids=jnp.zeros((self.capacity,), jnp.int32),
            timesteps=jnp.zeros((self.capacity,), jnp.int32),
            anchor_observations=(jnp.zeros((self.capacity, observation_dim), jnp.float32)
                                 if self.fixed_anchors else None),
        )
        zero = jnp.asarray(0, jnp.int32)
        return PQNEncoderArchiveState(data, zero, zero, zero, zero)

    @staticmethod
    def _skill_on(skills):
        return jnp.linalg.norm(skills, axis=-1) > 1e-6

    def insert(self, state, batch: dict):
        obs = jnp.asarray(batch["observations"], jnp.float32)
        count = obs.shape[0]
        if count % self.num_envs:
            raise ValueError("archive inserts must contain complete vector steps")
        # Keep the newest capacity rows if a single chunk exceeds the ring.
        skip = max(0, count - self.capacity)
        packed_map, tail = self.codec.encode_jax(obs[skip:])
        rows = EncoderArchiveRow(
            obs_map=packed_map, obs_tail=tail,
            skills=jnp.asarray(batch["skills"][skip:], jnp.float32),
            episode_ids=jnp.asarray(batch["episode_ids"][skip:], jnp.int32),
            timesteps=jnp.asarray(batch["timesteps"][skip:], jnp.int32),
            anchor_observations=(jnp.asarray(batch["anchor_observations"][skip:], jnp.float32)
                                 if self.fixed_anchors else None),
        )
        slots = (state.pointer + jnp.arange(skip, count, dtype=jnp.int32)) % self.capacity
        active_delta = jnp.sum(self._skill_on(rows.skills).astype(jnp.int32)) - jnp.sum(
            self._skill_on(state.data.skills[slots]).astype(jnp.int32)
        )
        data = jax.tree_util.tree_map(
            lambda target, value: target.at[slots].set(value), state.data, rows
        )
        return state.replace(
            data=data, pointer=(state.pointer + count) % self.capacity,
            size=jnp.minimum(self.capacity, state.size + count),
            active_rows=state.active_rows + active_delta,
            total_inserted_rows=state.total_inserted_rows + count,
        )

    def _storage_indices(self, state, indices):
        return jnp.where(state.size < self.capacity, indices,
                         (state.pointer + indices) % self.capacity)

    def _sample_skill_indices(self, state, key, batch_size):
        """Reference 4x candidates, repeated valid draws, full-ring fallback."""
        candidates = jax.random.randint(key, (4 * batch_size,), 0, jnp.maximum(state.size, 1))
        active = self._skill_on(state.data.skills[self._storage_indices(state, candidates)])
        active &= state.size > 0
        positions = jnp.nonzero(active, size=batch_size, fill_value=0)[0]
        count = jnp.sum(active.astype(jnp.int32))
        selected = candidates[positions[jnp.arange(batch_size) % jnp.maximum(count, 1)]]

        def fallback(_):
            storage = jnp.arange(self.capacity)
            valid = self._skill_on(state.data.skills) & (
                (state.size == self.capacity) | (storage < state.size)
            )
            positions = jnp.nonzero(valid, size=batch_size, fill_value=0)[0]
            count = jnp.sum(valid.astype(jnp.int32))
            selected_storage = positions[jnp.arange(batch_size) % jnp.maximum(count, 1)]
            chronological = jnp.where(state.size < self.capacity, selected_storage,
                                      (selected_storage - state.pointer) % self.capacity)
            return jnp.where(count > 0, chronological, candidates[:batch_size])

        return jax.lax.cond(count > 0, lambda _: selected, fallback, operand=None)

    def _batch_from_indices(self, state, indices, future_key):
        """Sample futures in chronological coordinates, then guard episode identity.

        The reference draws up to the configured horizon, then rejects draws that
        cross an early termination. It does not renormalize at observed death.
        """
        slots = self._storage_indices(state, indices)
        base_steps = state.data.timesteps[slots]
        horizon = jnp.minimum(
            jnp.maximum((state.size - 1 - indices) // self.num_envs, 0),
            jnp.maximum(self.episode_length - 1 - base_steps, 0),
        )
        if self.max_future_steps > 0:
            horizon = jnp.minimum(horizon, self.max_future_steps)
        offsets, has_future = truncated_geometric(future_key, horizon, self.future_gamma)
        future_indices = indices + offsets * self.num_envs
        future_slots = self._storage_indices(state, future_indices)
        valid = (
            (state.size > 0) & has_future & (future_indices < state.size)
            & (state.data.episode_ids[future_slots] == state.data.episode_ids[slots])
            & (state.data.timesteps[future_slots] == base_steps + offsets)
        )
        if self.fixed_anchors:
            valid = valid & self._skill_on(state.data.skills[future_slots])
        future_slots = jnp.where(valid, future_slots, slots)
        obs = self.codec.decode_jax(state.data.obs_map[slots], state.data.obs_tail[slots])
        future_obs = self.codec.decode_jax(
            state.data.obs_map[future_slots], state.data.obs_tail[future_slots]
        )
        skills = state.data.skills[slots]
        anchors = (state.data.anchor_observations[slots]
                   if self.fixed_anchors else obs)
        # The potential encoder reads `observations`, not `anchor_observations`,
        # as phi(s) in phi(future)-phi(s). Branch mode must put the copied physical
        # anchor there; the sampled row only selects a geometric outcome offset.
        encoder_base = anchors if self.fixed_anchors else obs
        batch = SkillBatch(
            observations=encoder_base,
            next_observations=future_obs if self.fixed_anchors else obs,
            future_observations=future_obs,
            anchor_observations=anchors, skills=skills, future_valid=valid.astype(jnp.float32),
        )
        info = {
            "archive/valid_fraction": jnp.mean(valid.astype(jnp.float32)),
            "archive/skill_on_fraction": jnp.mean(self._skill_on(skills).astype(jnp.float32)),
            "archive/mean_future_offset": jnp.sum(jnp.where(valid, offsets, 0)) / jnp.maximum(valid.sum(), 1),
            "archive/self_future_fraction": jnp.mean((~valid).astype(jnp.float32)),
        }
        return batch, info

    def sample_encoder_batch(self, state, key: jax.Array, batch_size: int):
        if batch_size <= 0:
            raise ValueError("encoder batch_size must be positive")
        sample_key, future_key = jax.random.split(key)
        indices = self._sample_skill_indices(state, sample_key, batch_size)
        return self._batch_from_indices(state, indices, future_key)
