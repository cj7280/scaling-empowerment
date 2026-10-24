"""Replay and archive samplers for the continuous SEE agent (pure functions).

`truncated_geometric` is shared with the PQN encoder archive and relabel.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from empowerment_warp.core.replay import (
    BranchTrajectoryArchiveState,
    EpisodeArchiveState,
    ReplayBuffer,
    ReplayState,
    Transition,
)
from empowerment_warp.core.rollout import potential_switch_steps
from empowerment_warp.agents.skill_classifier import SkillBatch


class ArchiveMetrics(NamedTuple):
    """Diagnostics of the encoder's temporal-pair sampling."""

    valid_fraction: jax.Array
    self_future_fraction: jax.Array
    mean_future_offset: jax.Array
    fresh_group_fraction: jax.Array


class AnchorReplayMetrics(NamedTuple):
    """Realized active-row views and unavailable-HER fallbacks."""

    true_fraction: jax.Array
    current_fraction: jax.Array
    her_fraction: jax.Array
    her_fallback_fraction: jax.Array
    her_endpoint_available_fraction: jax.Array
    segment_end_available_fraction: jax.Array


def _zero_anchor_replay_metrics() -> AnchorReplayMetrics:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return AnchorReplayMetrics(zero, zero, zero, zero, zero, zero)


def truncated_geometric(
    key: jax.Array,
    max_steps: jax.Array,
    gamma: float,
) -> tuple[jax.Array, jax.Array]:
    """k in [1, max_steps] with P(k) proportional to gamma^(k-1); invalid when max_steps == 0."""

    max_steps = jnp.asarray(max_steps, dtype=jnp.int32)
    valid = max_steps > 0
    gamma_value = jnp.asarray(min(max(gamma, 1e-6), 1.0 - 1e-6), dtype=jnp.float32)
    uniform = jax.random.uniform(key, max_steps.shape, dtype=jnp.float32)
    tail_mass = 1.0 - jnp.power(gamma_value, max_steps.astype(jnp.float32))
    target = jnp.maximum(1.0 - uniform * tail_mass, 1e-12)
    sampled = jnp.ceil(jnp.log(target) / jnp.log(gamma_value)).astype(jnp.int32)
    sampled = jnp.clip(sampled, 1, jnp.maximum(max_steps, 1))
    return jnp.where(valid, sampled, 0), valid


def backward_geometric_with_cap(
    key: jax.Array,
    max_steps: jax.Array,
    gamma: float,
) -> jax.Array:
    gamma_value = jnp.asarray(min(max(gamma, 1e-6), 1.0 - 1e-6), dtype=jnp.float32)
    uniform = jax.random.uniform(key, max_steps.shape, dtype=jnp.float32)
    sampled = jnp.floor(
        jnp.log(jnp.maximum(1.0 - uniform, 1e-12)) / jnp.log(gamma_value)
    ).astype(jnp.int32)
    return jnp.minimum(sampled, jnp.asarray(max_steps, dtype=jnp.int32))


def chronological_positions(
    replay: ReplayBuffer,
    replay_state: ReplayState,
    indices: jax.Array,
) -> jax.Array:
    oldest = jnp.where(replay_state.size < replay.capacity, 0, replay_state.cursor)
    return (indices - oldest) % replay.capacity


def episode_end_step(config, batch):
    return config.episode_length - 1


def policy_hindsight_batch(
    config,
    replay: ReplayBuffer,
    replay_state: ReplayState,
    indices: jax.Array,
    key: jax.Array,
    *,
    hindsight: bool,
) -> tuple[
    Transition,
    SkillBatch,
    jax.Array,
    jax.Array,
    AnchorReplayMetrics,
]:
    """Policy batch with hindsight (HER) anchors and relabel outcomes per replay row.

    Returns the SAC transitions, the classifier batch that scores their skill
    reward, the rows allowed to relabel, their outcome observations, and metrics.
    """

    base = replay.get(replay_state, indices)
    positions = chronological_positions(replay, replay_state, indices)

    anchor_observations = base.anchor_observations
    relabel_outcomes = base.observations
    relabel_allowed = jnp.ones(indices.shape, dtype=jnp.bool_)
    anchor_metrics = _zero_anchor_replay_metrics()
    if hindsight and config.sample_hindsight_anchors:
        buffer_back = positions // replay.sequence_stride
        skill_on = jnp.linalg.norm(base.skills, axis=-1) > 1e-6
        # Negative episode ids mark branch rows whose stored anchor is the
        # fixed physical start shared with the encoder archive.
        hindsight_eligible = skill_on & (base.episode_ids >= 0)
        segment_back = jnp.maximum(
            base.timesteps - potential_switch_steps(
                base.episode_ids,
                config.collector_potential_steps,
                config.random_potential_switch_max_steps,
                config.potential_switch_seed,
                    ),
            0,
        )
        max_back = jnp.minimum(buffer_back, segment_back)
        anchor_offsets = backward_geometric_with_cap(
            key,
            max_back,
            config.future_gamma,
        )
        anchor_indices = (indices - anchor_offsets * replay.sequence_stride) % replay.capacity
        anchor_rows = replay.get(replay_state, anchor_indices)
        hindsight_valid = (
            hindsight_eligible
            & (anchor_rows.episode_ids == base.episode_ids)
            & (anchor_rows.timesteps == base.timesteps - anchor_offsets)
        )
        anchor_mask = hindsight_valid.reshape(
            (hindsight_valid.shape[0],) + (1,) * (base.observations.ndim - 1)
        )
        anchor_observations = jnp.where(
            anchor_mask,
            anchor_rows.observations,
            base.anchor_observations,
        )
        next_anchor_observations = jnp.where(
            anchor_mask,
            anchor_rows.observations,
            base.next_anchor_observations,
        )
        base = base.replace(
            anchor_observations=anchor_observations,
            next_anchor_observations=next_anchor_observations,
        )
        relabel_allowed = hindsight_valid
        if config.achieved_skill_relabel_outcome == "discounted":
            # Discounted greedy relabel: M futures per row, offsets k ~ Geometric(1 - gamma)
            # truncated to [1, steps left in the episode]; the same future distribution the
            # encoder is trained on. The relabel aggregates their representations, so the
            # label is the skill that maximizes the segment's discounted return. Samples whose
            # row is not in the buffer / not the same episode fall back to the row's own state.
            samples = int(config.achieved_skill_relabel_samples)
            remaining = jnp.maximum(episode_end_step(config, base) - base.timesteps, 0)
            sample_keys = jax.random.split(jax.random.fold_in(key, 3), samples)

            def draw(sample_key):
                offsets, has_future = truncated_geometric(sample_key, remaining, config.future_gamma)
                f_positions = positions + offsets * replay.sequence_stride
                f_indices = (indices + offsets * replay.sequence_stride) % replay.capacity
                f_rows = replay.get(replay_state, f_indices)
                ok = (
                    hindsight_valid & has_future
                    & (f_positions < replay_state.size)
                    & (f_rows.episode_ids == base.episode_ids)
                    & (f_rows.timesteps == base.timesteps + offsets)
                )
                ok_mask = ok.reshape((ok.shape[0],) + (1,) * (base.observations.ndim - 1))
                return jnp.where(ok_mask, f_rows.next_observations, base.observations), ok

            sampled_outcomes, sampled_ok = jax.vmap(draw)(sample_keys)  # (M, rows, ...), (M, rows)
            relabel_outcomes = jnp.moveaxis(sampled_outcomes, 0, 1)      # (rows, M, ...)
            dtype = base.loss_weights.dtype
            anchor_metrics = anchor_metrics._replace(
                segment_end_available_fraction=(
                    jnp.sum(sampled_ok.astype(dtype)) / jnp.maximum(samples * jnp.sum(hindsight_valid.astype(dtype)), 1.0)
                )
            )
        elif config.achieved_skill_relabel_outcome == "segment_end":
            # Segment relabeling: the hindsight label is the skill achieved from the
            # (re-sampled) anchor to the END of the skill segment -- the episode's final
            # state, since a skill runs to the episode end -- rather than to this row's
            # own state. One consistent label per segment. Rows whose episode end is not
            # in the buffer yet keep the current-state outcome and are counted.
            end_steps = episode_end_step(config, base) - base.timesteps
            end_positions = positions + end_steps * replay.sequence_stride
            end_indices = (indices + end_steps * replay.sequence_stride) % replay.capacity
            end_rows = replay.get(replay_state, end_indices)
            end_available = (
                hindsight_valid
                & (end_steps >= 0)
                & (end_positions < replay_state.size)
                & (end_rows.episode_ids == base.episode_ids)
                & (end_rows.timesteps == episode_end_step(config, base))
            )
            end_mask = end_available.reshape(
                (end_available.shape[0],) + (1,) * (base.observations.ndim - 1)
            )
            relabel_outcomes = jnp.where(end_mask, end_rows.next_observations, relabel_outcomes)
            dtype = base.loss_weights.dtype
            anchor_metrics = anchor_metrics._replace(
                segment_end_available_fraction=(
                    jnp.sum(end_available.astype(dtype))
                    / jnp.maximum(jnp.sum(hindsight_valid.astype(dtype)), 1.0)
                )
            )

    return base, SkillBatch(
        observations=base.observations,
        next_observations=base.next_observations,
        future_observations=base.next_observations,
        anchor_observations=anchor_observations,
        skills=base.skills,
        future_valid=jnp.ones(indices.shape, dtype=jnp.bool_),
    ), relabel_allowed, relabel_outcomes, anchor_metrics


def representation_archive_batch(
    config,
    archive,
    archive_state: EpisodeArchiveState | BranchTrajectoryArchiveState,
    slots: jax.Array,
    rows: jax.Array,
    row_valid: jax.Array,
    key: jax.Array,
) -> tuple[SkillBatch, ArchiveMetrics]:
    """Temporal pairs inside one stored episode: discounted future sampling."""

    if config.uses_marginal_branches:
        return marginal_representation_archive_batch(
            config,
            archive,
            archive_state,
            slots,
            rows,
            row_valid,
            key,
        )

    base = archive.get(archive_state, slots, rows)
    horizon_cap = min(config.max_future_steps, archive.episode_length)
    end = jnp.minimum(archive_state.lengths[slots], rows + horizon_cap)
    horizon = jnp.maximum(end - 1 - rows, 0)
    # A row with no later step in its episode is its own future (offset 0).
    future_offsets, has_future = truncated_geometric(
        key,
        horizon,
        config.future_gamma,
    )
    future_rows = jnp.minimum(rows + future_offsets, archive.episode_length - 1)
    future_observations = archive_state.episodes.observations[slots, future_rows]
    future_valid = row_valid
    observation_mask = future_valid.reshape(
        (future_valid.shape[0],) + (1,) * (base.observations.ndim - 1)
    )
    future_observations = jnp.where(
        observation_mask,
        future_observations,
        base.observations,
    )
    label_future_observations = None
    if config.encoder_skill_source == "achieved_discounted":
        # M independent geometric futures per row for the encoder's HER label (the same
        # sampler the policy batch's discounted greedy label uses); the contrastive pair
        # above stays ONE future.
        label_keys = jax.random.split(jax.random.fold_in(key, 7), config.achieved_skill_relabel_samples)
        def _one(label_key):
            offsets, _ = truncated_geometric(label_key, horizon, config.future_gamma)
            gathered = archive_state.episodes.observations[
                slots, jnp.minimum(rows + offsets, archive.episode_length - 1)
            ]
            return jnp.where(observation_mask, gathered, base.observations)
        label_future_observations = jnp.moveaxis(jax.vmap(_one)(label_keys), 0, 1)  # [rows, M, obs...]
    valid_float = future_valid.astype(jnp.float32)
    moved = valid_float * has_future.astype(jnp.float32)
    metrics = ArchiveMetrics(
        valid_fraction=jnp.mean(valid_float),
        self_future_fraction=jnp.mean(valid_float * (~has_future).astype(jnp.float32)),
        mean_future_offset=jnp.sum(future_offsets.astype(jnp.float32) * moved)
        / jnp.maximum(jnp.sum(moved), 1.0),
        # Episode archive: continuous per-completion drift, no discrete sweep.
        fresh_group_fraction=jnp.asarray(0.0, dtype=jnp.float32),
    )
    return SkillBatch(
        observations=base.observations,
        next_observations=base.next_observations,
        future_observations=future_observations,
        anchor_observations=base.anchor_observations,
        skills=base.skills,
        future_valid=future_valid,
        label_future_observations=label_future_observations,
    ), metrics


def marginal_representation_archive_batch(
    config,
    archive,
    archive_state: BranchTrajectoryArchiveState,
    slots: jax.Array,
    rows: jax.Array,
    row_valid: jax.Array,
    key: jax.Array,
) -> tuple[SkillBatch, ArchiveMetrics]:
    """Pair a fixed marginal anchor with a future on its commanded branch."""

    branch_horizon = config.branch_horizon
    horizon_cap = min(config.max_future_steps, branch_horizon)
    virtual_remaining = jnp.maximum(
        jnp.minimum(
            branch_horizon - 1 - rows,
            archive_state.lengths[slots] - 1 - rows,
        ),
        0,
    )
    max_steps = jnp.minimum(
        jnp.minimum(archive_state.lengths[slots], virtual_remaining),
        horizon_cap,
    )
    future_offsets, has_future = truncated_geometric(
        key,
        max_steps,
        config.future_gamma,
    )
    last_outcome_indices = jnp.maximum(
        archive_state.lengths[slots] - 1,
        0,
    )
    future_indices = jnp.minimum(
        rows + future_offsets,
        last_outcome_indices,
    )
    anchors = archive_state.anchors[slots]
    future_observations = archive_state.outcomes[slots, future_indices]
    # The reward path reads `next_observations`, and the potential loss reads
    # `future_observations`. Pointing the first at outcome 0 would make
    # `skill/intrinsic_reward` a ONE-STEP delta while the loss beside it
    # measures the full anchor->future displacement, so the two could not be
    # read against each other. marginal_branch is potential-only, and the
    # potential loss never reads `next_observations`, so aligning them here
    # changes only what the reward metric measures.
    next_observations = future_observations
    has_outcome = archive_state.lengths[slots] > 0
    observation_mask = (row_valid & has_outcome).reshape(
        (row_valid.shape[0],) + (1,) * (anchors.ndim - 1)
    )
    outcome_mask = (row_valid & has_outcome).reshape(
        (row_valid.shape[0],) + (1,) * (anchors.ndim - 1)
    )
    future_observations = jnp.where(
        observation_mask,
        future_observations,
        anchors,
    )
    next_observations = jnp.where(
        outcome_mask,
        next_observations,
        anchors,
    )
    future_valid = row_valid
    valid_float = future_valid.astype(jnp.float32)
    moved = valid_float * has_future.astype(jnp.float32)
    metrics = ArchiveMetrics(
        valid_fraction=jnp.mean(valid_float),
        self_future_fraction=jnp.mean(
            valid_float * (~has_future).astype(jnp.float32)
        ),
        mean_future_offset=(
            jnp.sum(future_offsets.astype(jnp.float32) * moved)
            / jnp.maximum(jnp.sum(moved), 1.0)
        ),
        # Share drawn from the latest insertion. This is 2,048 groups during
        # bootstrap, then the actual 40/41 completed phase width in steady
        # state; the archive records that width rather than assuming a sweep.
        fresh_group_fraction=(
            jnp.sum(
                valid_float
                * (
                    (archive_state.cursor - 1 - slots)
                    % archive.num_groups
                    < archive_state.fresh_group_count
                ).astype(jnp.float32)
            )
            / jnp.maximum(jnp.sum(valid_float), 1.0)
        ),
    )
    return SkillBatch(
        observations=anchors,
        next_observations=next_observations,
        future_observations=future_observations,
        anchor_observations=anchors,
        skills=archive_state.skills[slots],
        future_valid=future_valid,
    ), metrics
