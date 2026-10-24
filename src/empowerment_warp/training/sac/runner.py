"""SAC collection, replay storage, checkpoints, and training loop."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any
import json
import math
import time

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.agents.skill_classifier import SkillLearner, SkillState
from empowerment_warp.checkpointing import CheckpointStore, model_checkpoint
from empowerment_warp.core.replay import (
    BranchTrajectory,
    BranchTrajectoryArchive,
    BranchTrajectoryArchiveState,
    EpisodeArchive,
    EpisodeArchiveState,
    ReplayBuffer,
    ReplayState,
    RepresentationTransition,
    Transition,
)
from empowerment_warp.core.rollout import (
    CollectorState,
    StaggeredBranchState,
    branch_anchor_layout,
    coverage_counts,
    coverage_floor_masks,
    coverage_track_specs,
    initialize_collector,
    make_branch_trajectory_rollout,
    make_rollout_chunk,
    make_staggered_branch_trajectory_rollout,
)
from empowerment_warp.envs.factory import make_environment
from empowerment_warp.tracking import WandbMirror
from empowerment_warp.agents.see_sac import (
    ContinuousSAC,
    SACState,
    SEEAgent,
    create_learners,
)
from empowerment_warp.training.sac.config import (
    ExperimentConfig,
    OffPolicyConfig,
)
from empowerment_warp.training.sac.logging import (
    BranchMetrics,
    FreshBranchHoldout,
    HoldoutCycleAccumulator,
    UpdateMetrics,
    _accumulate_exploration,
    _branch_fano_mi_lower_bound,
    _mean,
    _summarize_exploration,
    _zero_branch_metrics,
    _zero_fresh_branch_holdout,
    summarize_holdout_event,
    summarize_metrics,
)
from empowerment_warp.training.sac.progress import (
    create_progress_snapshot,
    due_progress_targets,
    log_progress_snapshot,
)
from empowerment_warp.utils.sampling import marginal_representation_archive_batch


@struct.dataclass
class OffPolicyState:
    collector: CollectorState
    replay: ReplayState
    representation_replay: EpisodeArchiveState | BranchTrajectoryArchiveState
    sac: SACState
    skill: SkillState
    rng: jax.Array
    env_steps: jax.Array
    collector_vector_steps: jax.Array
    branch_simulator_steps: jax.Array
    staggered_branch: StaggeredBranchState | None


def _branch_label_stats(skill, target_skills: jax.Array, source_skills: jax.Array,
                        weights: jax.Array | None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(target_count_min, target_count_max, source_target_match_fraction) for the branch
    metrics. Continuous skills have no classes: counts report the number of (weighted) rows,
    and "match" is the fraction of branches whose target skill lies within 0.99 cosine of its
    source skill."""
    rows = jnp.ones((target_skills.shape[0],), jnp.float32) if weights is None else weights
    counts = jnp.sum(rows)[None]
    cosine = jnp.sum(target_skills * source_skills, axis=-1) / jnp.maximum(
        jnp.linalg.norm(target_skills, axis=-1) * jnp.linalg.norm(source_skills, axis=-1), 1e-6
    )
    match = (cosine > 0.99).astype(jnp.float32)
    if weights is None:
        match_fraction = jnp.mean(match)
    else:
        match_fraction = jnp.sum(weights * match) / jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.min(counts).astype(jnp.float32), jnp.max(counts).astype(jnp.float32), match_fraction


def _holdout_branch_fields(holdout: FreshBranchHoldout, num_skills: int) -> dict[str, jax.Array]:
    """The ten held-out BranchMetrics fields shared by the bootstrap and staggered sweeps.

    Spliced into BranchMetrics(...) with ``**`` at the position of these fields, so the traced
    operations keep their order.
    """
    return dict(
        holdout_accuracy=holdout.augmented.accuracy,
        holdout_mi_lower_bound=holdout.augmented.mi_lower_bound,
        holdout_loss=holdout.augmented.loss,
        holdout_intrinsic_reward=holdout.augmented.intrinsic_reward_mean,
        holdout_mi_fano=_branch_fano_mi_lower_bound(holdout.augmented.accuracy, num_skills),
        holdout_accuracy_clean=holdout.clean.accuracy,
        holdout_mi_lower_bound_clean=holdout.clean.mi_lower_bound,
        holdout_loss_clean=holdout.clean.loss,
        holdout_mi_fano_clean=_branch_fano_mi_lower_bound(holdout.clean.accuracy, num_skills),
        holdout_sampled_rows=holdout.sampled_rows,
    )


class OffPolicyTrainer:
    """Own the compiled segment; Python only handles logging and checkpoints."""

    def __init__(
        self,
        env: Any,
        sac: ContinuousSAC,
        skill: SkillLearner,
        config: OffPolicyConfig,
    ):
        self.env = env
        self.sac = sac
        self.skill = skill
        self.config = config
        if config.reward_partition_anneal_end_steps and (
            (skill.config.continuous_reward_normalization, skill.config.encoder_objective) not in {
                ("sphere", "vmf_forward"), ("sphere", "forward_infonce"),
                ("sampled_infonce", "forward_infonce"),
            }
            or skill.config.potential_reward_reference != "anchor"
        ):
            raise ValueError("partition annealing requires matched normalized decoder anchor rewards")
        self.num_envs = env.config.num_envs if hasattr(env, "config") else env.num_envs
        self.uses_marginal_branches = config.uses_marginal_branches
        self.branch_phase_groups = 0
        self.branch_steps_per_segment = 0
        self.branch_completion_batch_size = 0
        if self.uses_marginal_branches:
            if skill.config.method != "potential":
                raise ValueError("marginal_branch currently supports potential skills only")
            if config.branch_targets_per_anchor != 1:
                raise ValueError("continuous marginal_branch supports branch_targets_per_anchor == 1 only")
            if config.representation_episodes < self.num_envs:
                raise ValueError(
                    "marginal_branch representation_episodes must fit one full-width "
                    "branch sweep"
                )
            # The LIVE collector's reset staggering is a separate concern from how
            # many phase groups the BRANCH collector partitions lanes into. They
            # were fused, which pinned branch_phase_groups to 50 and made H=1
            # unreachable. Live staggering still spreads anchors across episode
            # phases, so branch phases may be coarser without losing that spread.
            expected_stagger_groups = config.episode_length // config.rollout_steps
            if config.stagger_reset_groups != expected_stagger_groups:
                raise ValueError(
                    "marginal_branch requires one stagger group per rollout chunk: "
                    f"expected {expected_stagger_groups}, got "
                    f"{config.stagger_reset_groups}"
                )
            self.branch_phase_groups = config.branch_phase_groups
            self.branch_steps_per_segment = config.branch_steps_per_segment
            self.branch_completion_batch_size = math.ceil(
                self.num_envs / self.branch_phase_groups
            )
        chunk_size = config.rollout_steps * self.num_envs
        # Whole vector steps only: a future `k` steps later lives at index + k * num_envs,
        # which is only true when the ring wraps on a vector-step boundary.
        replay_capacity = (config.replay_capacity // self.num_envs) * self.num_envs
        if chunk_size > replay_capacity:
            raise ValueError("one rollout chunk must fit in the policy replay")
        simultaneous = config.max_simultaneous_completions(self.num_envs)
        if config.representation_episodes < simultaneous:
            raise ValueError(
                f"representation_episodes={config.representation_episodes} is smaller than the "
                f"{simultaneous} lanes that can finish an episode on one step; raise it or "
                "enable staggered resets"
            )
        self.replay = ReplayBuffer(replay_capacity, sequence_stride=self.num_envs)
        if self.uses_marginal_branches:
            self.representation_replay = BranchTrajectoryArchive(
                config.representation_episodes,
                config.branch_horizon,
            )
        else:
            # The skill buffer under vectorization: a bag of the most
            # recent completed episodes, so every stored row has its whole future.
            self.representation_replay = EpisodeArchive(
                self.num_envs,
                config.representation_episodes,
                config.episode_length,
            )

        def action_fn(
            state: SACState,
            obs: jax.Array,
            anchors: jax.Array,
            skills: jax.Array,
            keys: jax.Array,
        ) -> jax.Array:
            actions = self.sac.sample_actions(state, obs, anchors, skills, keys)
            if getattr(getattr(self.env, 'config', None), 'random_preparation', False):
                random_actions = self.env.random_actions(obs, keys)
                actions = jnp.where((jnp.linalg.norm(skills, axis=-1) <= 1.e-6)[:, None], random_actions, actions)
            return actions

        def skill_sampler(key: jax.Array, count: int) -> jax.Array:
            return self.skill.sample_skills(key, count)

        self._skill_sampler = skill_sampler
        self._rollout = make_rollout_chunk(
            env,
            action_fn,
            skill_sampler,
            chunk_steps=config.rollout_steps,
            potential_steps=config.collector_potential_steps,
            capture_source=self.uses_marginal_branches,
            potential_lane_fraction=config.potential_lane_fraction,
            random_potential_switch_max_steps=config.random_potential_switch_max_steps,
            potential_switch_seed=config.potential_switch_seed,
            skill_steps=config.branch_horizon,
        )
        self._branch_rollout = (
            make_branch_trajectory_rollout(
                env,
                action_fn,
                skill_sampler,
                self.skill.encoder_observations,
                horizon=config.branch_horizon,
                targets_per_anchor=config.branch_targets_per_anchor,
            )
            if self.uses_marginal_branches
            else None
        )
        if self.uses_marginal_branches:
            (
                self._initialize_staggered_branch,
                self._staggered_branch_rollout,
            ) = make_staggered_branch_trajectory_rollout(
                env,
                action_fn,
                self.skill.encoder_observations,
                horizon=config.branch_horizon,
                chunk_steps=self.branch_steps_per_segment,
                phase_groups=self.branch_phase_groups,
                targets_per_anchor=config.branch_targets_per_anchor,
                skill_sampler=skill_sampler,
            )
        else:
            self._initialize_staggered_branch = None
            self._staggered_branch_rollout = None
        self._bootstrap_anchor_lane = branch_anchor_layout(
            self.num_envs,
            1,
            config.branch_targets_per_anchor,
        )[0]
        self.agent = SEEAgent(sac, skill, config, self.replay, self.representation_replay)
        self._run_jit = jax.jit(self._run_impl)

    def _representation_transition(
        self,
        transition: Transition,
    ) -> RepresentationTransition:
        return RepresentationTransition(
            observations=self.skill.encoder_observations(transition.observations),
            anchor_observations=self.skill.encoder_observations(
                transition.anchor_observations
            ),
            next_observations=self.skill.encoder_observations(
                transition.next_observations
            ),
            skills=transition.skills,
            episode_ids=transition.episode_ids,
            timesteps=transition.timesteps,
            dones=transition.dones,
        )

    def initialize(self, key: jax.Array) -> OffPolicyState:
        collector_key, sac_key, skill_key, next_key = jax.random.split(key, 4)
        collector = initialize_collector(
            self.env,
            collector_key,
            self._skill_sampler,
            potential_steps=self.config.collector_potential_steps,
            episode_length=self.config.episode_length,
            stagger_groups=self.config.stagger_reset_groups,
            potential_lane_fraction=self.config.potential_lane_fraction,
            random_potential_switch_max_steps=self.config.random_potential_switch_max_steps,
            potential_switch_seed=self.config.potential_switch_seed,
        )
        sac_state = self.sac.initialize(sac_key)
        skill_state = self.skill.initialize(skill_key)
        observation = collector.env_state.obs
        actions = jnp.zeros((self.num_envs, self.sac.config.action_dim), dtype=jnp.float32)
        example = Transition(
            observations=observation,
            anchor_observations=collector.anchor_observations,
            actions=actions,
            rewards=jnp.zeros((self.num_envs,), dtype=jnp.float32),
            masks=jnp.ones((self.num_envs,), dtype=jnp.float32),
            next_observations=observation,
            next_anchor_observations=collector.anchor_observations,
            dones=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
            skills=collector.skills,
            next_skills=collector.skills,
            loss_weights=jnp.ones((self.num_envs,), dtype=jnp.float32),
            episode_ids=collector.episode_ids,
            timesteps=collector.env_state.step_count,
        )
        representation_example = self._representation_transition(example)
        if self.uses_marginal_branches:
            encoder_observations = self.skill.encoder_observations(observation)
            representation_state = self.representation_replay.initialize(
                encoder_observations,
                collector.skills,
            )
            staggered_branch = StaggeredBranchState(
                env_state=collector.env_state,
                policy_anchors=observation,
                encoder_anchors=encoder_observations,
                source_skills=collector.skills,
                skills=collector.skills,
                outcomes=jnp.zeros(
                    (self.config.branch_horizon,) + encoder_observations.shape,
                    dtype=encoder_observations.dtype,
                ),
                valid=jnp.zeros(
                    (self.config.branch_horizon, self.num_envs),
                    dtype=jnp.bool_,
                ),
                progress=jnp.zeros((self.num_envs,), dtype=jnp.int32),
                genuine=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
                alive=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
                active=jnp.asarray(False),
                skill_shift=jnp.asarray(0, dtype=jnp.int32),
                anchor_xy=jnp.zeros((self.num_envs, 2), dtype=jnp.float32),
                policy_outcomes=jnp.zeros(
                    (self.config.branch_horizon,) + observation.shape,
                    dtype=observation.dtype,
                ),
                actions=jnp.zeros(
                    (self.config.branch_horizon, self.num_envs, self.sac.config.action_dim),
                    dtype=jnp.float32,
                ),
                terminated=jnp.zeros(
                    (self.config.branch_horizon, self.num_envs), dtype=jnp.bool_
                ),
                truncated=jnp.zeros(
                    (self.config.branch_horizon, self.num_envs), dtype=jnp.bool_
                ),
                alive_rows=jnp.zeros(
                    (self.config.branch_horizon, self.num_envs), dtype=jnp.bool_
                ),
                emit_index=jnp.asarray(0, dtype=jnp.int32),
            )
        else:
            representation_state = self.representation_replay.initialize(
                representation_example
            )
            staggered_branch = None
        return OffPolicyState(
            collector=collector,
            replay=self.replay.initialize(example),
            representation_replay=representation_state,
            sac=sac_state,
            skill=skill_state,
            rng=next_key,
            env_steps=jnp.asarray(0, dtype=jnp.int32),
            collector_vector_steps=jnp.asarray(0, dtype=jnp.int32),
            branch_simulator_steps=jnp.asarray(0, dtype=jnp.int32),
            staggered_branch=staggered_branch,
        )


    def _fresh_branch_holdout(
        self,
        skill_state: SkillState,
        branch: BranchTrajectory,
        group_count: jax.Array,
        key: jax.Array,
    ) -> FreshBranchHoldout:
        """Score newly completed branches before the encoder trains on them.

        `skill/accuracy` is measured on an archive the encoder revisits for a
        whole refresh cycle. This uses the same sampler and loss on the valid
        prefix of the just-completed batch, which is held out by construction.
        """

        sample_key, batch_key, loss_key = jax.random.split(key, 3)
        lengths = jnp.sum(branch.valid.astype(jnp.int32), axis=0)
        group_count = jnp.clip(group_count, 0, branch.anchors.shape[0])
        group_mask = jnp.arange(branch.anchors.shape[0]) < group_count
        lengths = jnp.where(group_mask, lengths, 0)
        fresh = BranchTrajectoryArchiveState(
            anchors=branch.anchors,
            skills=branch.skills,
            outcomes=jnp.swapaxes(branch.outcomes, 0, 1),
            lengths=lengths,
            cursor=jnp.asarray(0, dtype=jnp.int32),
            size=group_count,
            active_rows=jnp.sum(lengths),
            fresh_group_count=group_count,
        )
        slots, rows, row_valid = self.representation_replay.sample_active(
            fresh,
            sample_key,
            self.config.batch_size,
        )
        batch, _ = marginal_representation_archive_batch(
            self.agent.config,
            self.agent.representation_replay,
            fresh,
            slots,
            rows,
            row_valid,
            batch_key,
        )
        # `marginal_representation_archive_batch` already aligns
        # `next_observations` with the future, so reward and loss share a
        # geometry. Archive rows are the encoder view, as `update_preprocessed`
        # expects.
        augmented = self.skill.evaluate_preprocessed(skill_state, batch, loss_key)
        clean = self.skill.evaluate_preprocessed(
            skill_state,
            batch,
            loss_key,
            apply_phi_noise=False,
        )
        return FreshBranchHoldout(
            augmented=augmented,
            clean=clean,
            sampled_rows=jnp.sum(row_valid.astype(jnp.float32)),
        )

    def _run_impl(self, state: OffPolicyState) -> tuple[OffPolicyState, UpdateMetrics]:
        if self.uses_marginal_branches:
            (
                rng,
                capture_key,
                bootstrap_key,
                staggered_key,
                bootstrap_holdout_key,
                staggered_holdout_key,
                update_root,
            ) = jax.random.split(state.rng, 7)
            capture_step = jax.random.randint(
                capture_key,
                (),
                0,
                self.config.rollout_steps,
            )
            collector, transitions, exploration_metrics, source = self._rollout(
                state.sac,
                state.collector,
                capture_step,
            )
        else:
            collector, transitions, exploration_metrics = self._rollout(
                state.sac,
                state.collector,
            )
            rng, update_root = jax.random.split(state.rng)
        replay_state = self.replay.insert(state.replay, transitions)
        physical_steps = self.config.rollout_steps * self.num_envs
        env_steps = state.env_steps + physical_steps
        collector_vector_steps = (
            state.collector_vector_steps + self.config.rollout_steps
        )
        if self.uses_marginal_branches:
            staggered_branch = state.staggered_branch
            if staggered_branch is None:
                raise ValueError("marginal_branch state is missing its branch collector")
            branch_due = (
                collector_vector_steps % self.config.episode_length
            ) == 0
            bootstrap_due = branch_due & (
                state.representation_replay.size
                < self.config.representation_episodes
            )

            def bootstrap_archive(
                carry: tuple[
                    BranchTrajectoryArchiveState,
                    StaggeredBranchState,
                ],
            ) -> tuple[
                BranchTrajectoryArchiveState,
                StaggeredBranchState,
                BranchMetrics,
            ]:
                representation_state, branch_state = carry
                branch = self._branch_rollout(state.sac, source, bootstrap_key)
                # Before any insert: the sweep is held out by construction here.
                holdout = self._fresh_branch_holdout(
                    state.skill,
                    branch,
                    jnp.asarray(self.num_envs, dtype=jnp.int32),
                    bootstrap_holdout_key,
                )
                representation_state = self.representation_replay.insert(
                    representation_state,
                    branch,
                )
                branch_state = jax.lax.cond(
                    representation_state.size >= self.config.representation_episodes,
                    lambda current: self._initialize_staggered_branch(
                        current,
                        source,
                        branch.skills,
                    ),
                    lambda current: current,
                    branch_state,
                )
                count_min, count_max, match_fraction = _branch_label_stats(
                    self.skill, branch.skills, source.skills[self._bootstrap_anchor_lane], None
                )
                inserted_groups = jnp.asarray(self.num_envs, dtype=jnp.float32)
                return representation_state, branch_state, BranchMetrics(
                    applied=jnp.asarray(1.0, dtype=jnp.float32),
                    simulator_steps=jnp.asarray(
                        self.num_envs * self.config.branch_horizon,
                        dtype=jnp.float32,
                    ),
                    valid_fraction=jnp.mean(branch.valid.astype(jnp.float32)),
                    target_count_min=count_min,
                    target_count_max=count_max,
                    source_target_match_fraction=match_fraction,
                    inserted_groups=inserted_groups,
                    discarded_startup_groups=jnp.asarray(0.0, dtype=jnp.float32),
                    collector_active=branch_state.active.astype(jnp.float32),
                    archive_refresh_fraction=(
                        inserted_groups / self.config.representation_episodes
                    ),
                    **_holdout_branch_fields(holdout, self.skill.config.num_skills),
                    holdout_trajectories=inserted_groups,
                    holdout_physical_anchors=jnp.asarray(
                        math.ceil(
                            self.num_envs
                            / self.config.branch_targets_per_anchor
                        ),
                        dtype=jnp.float32,
                    ),
                    holdout_cycle_eligible=jnp.asarray(0.0, dtype=jnp.float32),
                    # One-shot bootstrap stays encoder-only; the steady state
                    # feeds the policy from the staggered collector.
                    policy_rows_alive_fraction=jnp.asarray(0.0, dtype=jnp.float32),
                    skill_disp_mean=jnp.asarray(0.0, dtype=jnp.float32),
                    skill_endpoint_spread=jnp.asarray(0.0, dtype=jnp.float32),
                )

            (
                representation_state,
                staggered_branch,
                bootstrap_metrics,
            ) = jax.lax.cond(
                bootstrap_due,
                bootstrap_archive,
                lambda carry: (
                    carry[0],
                    carry[1],
                    _zero_branch_metrics(),
                ),
                (state.representation_replay, staggered_branch),
            )

            def advance_staggered(
                values: tuple[
                    BranchTrajectoryArchiveState,
                    StaggeredBranchState,
                    ReplayState,
                ],
            ) -> tuple[
                BranchTrajectoryArchiveState,
                StaggeredBranchState,
                ReplayState,
                BranchMetrics,
            ]:
                archive, branch_state, replay_in = values
                (
                    branch_state,
                    branch,
                    completed_source_skills,
                    completed_count,
                    discarded_count,
                    branch_transitions,
                    branch_alive_fraction,
                    skill_disp_mean,
                    skill_endpoint_spread,
                ) = self._staggered_branch_rollout(
                    state.sac,
                    branch_state,
                    source,
                    staggered_key,
                )
                archive = self.representation_replay.insert_prefix(
                    archive,
                    branch,
                    completed_count,
                )
                holdout = jax.lax.cond(
                    completed_count > 0,
                    lambda _: self._fresh_branch_holdout(
                        state.skill,
                        branch,
                        completed_count,
                        staggered_holdout_key,
                    ),
                    lambda _: _zero_fresh_branch_holdout(),
                    operand=None,
                )
                completed_mask = (
                    jnp.arange(self.branch_completion_batch_size) < completed_count
                )
                count_min, count_max, match_fraction = _branch_label_stats(
                    self.skill, branch.skills, completed_source_skills, completed_mask.astype(jnp.float32)
                )
                valid_rows = branch.valid & completed_mask[None, :]
                completed_float = completed_count.astype(jnp.float32)
                if self.config.branch_transitions_to_policy_replay:
                    # A segment with no completion would otherwise insert a
                    # block of pure padding: inert, but it still consumes
                    # replay slots and sampling weight.
                    replay_out = jax.lax.cond(
                        completed_count > 0,
                        lambda rows: self.replay.insert(rows, branch_transitions),
                        lambda rows: rows,
                        replay_in,
                    )
                else:
                    replay_out = replay_in
                return archive, branch_state, replay_out, BranchMetrics(
                    applied=(completed_count > 0).astype(jnp.float32),
                    simulator_steps=jnp.asarray(
                        self.num_envs * self.branch_steps_per_segment,
                        dtype=jnp.float32,
                    ),
                    valid_fraction=(
                        jnp.sum(valid_rows.astype(jnp.float32))
                        / jnp.maximum(
                            completed_float * self.config.branch_horizon,
                            1.0,
                        )
                    ),
                    target_count_min=count_min,
                    target_count_max=count_max,
                    source_target_match_fraction=match_fraction,
                    inserted_groups=completed_float,
                    discarded_startup_groups=discarded_count.astype(jnp.float32),
                    collector_active=jnp.asarray(1.0, dtype=jnp.float32),
                    archive_refresh_fraction=(
                        completed_float / self.config.representation_episodes
                    ),
                    **_holdout_branch_fields(holdout, self.skill.config.num_skills),
                    holdout_trajectories=completed_float,
                    holdout_physical_anchors=jnp.ceil(
                        completed_float
                        / self.config.branch_targets_per_anchor
                    ),
                    holdout_cycle_eligible=(completed_count > 0).astype(
                        jnp.float32
                    ),
                    policy_rows_alive_fraction=branch_alive_fraction,
                    skill_disp_mean=skill_disp_mean,
                    skill_endpoint_spread=skill_endpoint_spread,
                )

            # Use the collector activity at segment entry. The second bootstrap
            # initializes it, and the following segment performs its first six
            # branch steps; this keeps the first full trajectory exactly 300 steps.
            advance_due = state.staggered_branch.active
            (
                representation_state,
                staggered_branch,
                replay_state,
                staggered_metrics,
            ) = jax.lax.cond(
                advance_due,
                advance_staggered,
                lambda values: (
                    values[0],
                    values[1],
                    values[2],
                    _zero_branch_metrics(),
                ),
                (representation_state, staggered_branch, replay_state),
            )
            branch_metrics = BranchMetrics(
                *(
                    getattr(bootstrap_metrics, field)
                    + getattr(staggered_metrics, field)
                    for field in BranchMetrics._fields
                )
            )
        else:
            representation_state = self.representation_replay.insert(
                state.representation_replay,
                self._representation_transition(transitions),
            )
            staggered_branch = state.staggered_branch
            branch_metrics = _zero_branch_metrics()
        branch_simulator_steps = state.branch_simulator_steps + (
            branch_metrics.simulator_steps.astype(jnp.int32)
        )
        # Wait for a full skill buffer: train once the episode bag is full.
        ready = (replay_state.size >= self.config.batch_size) & (
            representation_state.size >= self.config.representation_episodes
        )
        metric_shape = (self.config.updates_per_chunk,)

        (sac_state, skill_state), (
            sac_metrics, skill_metrics, relabel_metrics, archive_metrics, anchor_replay_metrics,
        ) = self.agent.update(state.sac, state.skill, replay_state, representation_state,
                              env_steps, update_root, ready)
        metrics = UpdateMetrics(
            sac=sac_metrics,
            skill=skill_metrics,
            relabel=relabel_metrics,
            archive=archive_metrics,
            anchor_replay=anchor_replay_metrics,
            branch=branch_metrics,
            exploration=exploration_metrics,
            replay_size=jnp.broadcast_to(replay_state.size, metric_shape),
            representation_size=jnp.broadcast_to(representation_state.size, metric_shape),
            representation_active_rows=jnp.broadcast_to(
                representation_state.active_rows, metric_shape
            ),
            env_steps=jnp.broadcast_to(env_steps, metric_shape),
            branch_simulator_steps_total=jnp.broadcast_to(
                branch_simulator_steps,
                metric_shape,
            ),
            update_applied=jnp.broadcast_to(ready.astype(jnp.float32), metric_shape),
        )
        return OffPolicyState(
            collector=collector,
            replay=replay_state,
            representation_replay=representation_state,
            sac=sac_state,
            skill=skill_state,
            rng=rng,
            env_steps=env_steps,
            collector_vector_steps=collector_vector_steps,
            branch_simulator_steps=branch_simulator_steps,
            staggered_branch=staggered_branch,
        ), metrics

    def run(self, state: OffPolicyState) -> tuple[OffPolicyState, UpdateMetrics]:
        return self._run_jit(state)


@dataclass(frozen=True, slots=True)
class Experiment:
    env: Any
    sac: ContinuousSAC
    skill: SkillLearner
    trainer: OffPolicyTrainer

def make_experiment(config: ExperimentConfig) -> Experiment:
    env = make_environment(config)
    sac, skill, config = create_learners(config, env)
    trainer = OffPolicyTrainer(env, sac, skill, config)
    return Experiment(env=env, sac=sac, skill=skill, trainer=trainer)

def require_runtime(config: ExperimentConfig) -> None:
    platform = jax.devices()[0].platform
    if platform != "gpu" and not config.allow_cpu:
        raise RuntimeError(
            f"training requires a GPU, but JAX selected {platform!r}; pass --allow-cpu only for tests"
        )


def train(config: ExperimentConfig, *, resume: bool = False, stop_after_segments: int | None = None) -> Path:
    if stop_after_segments is not None and stop_after_segments < 1:
        raise ValueError("stop_after_segments must be positive")
    require_runtime(config)
    run_dir = config.resolved_run_dir
    if run_dir.exists() and any(p.name not in {"recipe.json", "slurm_metadata.txt", "source_snapshot.json"} for p in run_dir.iterdir()) and not resume:
        raise FileExistsError(f"run directory is not empty: {run_dir}; pass --resume to continue")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    reference_path = run_dir / "reference_manifest.json"
    if resume:
        if not config_path.exists():
            raise FileNotFoundError(f"resume requested but {config_path} is absent")
        frozen = ExperimentConfig.from_dict(json.loads(config_path.read_text())).to_dict()
        requested = config.to_dict()
        if frozen != requested:
            # The one permitted difference is a LARGER budget (a continuation from the latest
            # checkpoint). Everything else must match the frozen run config.
            frozen_other = {k: v for k, v in frozen.items() if k != "total_env_steps"}
            requested_other = {k: v for k, v in requested.items() if k != "total_env_steps"}
            if frozen_other != requested_other or requested["total_env_steps"] <= frozen["total_env_steps"]:
                raise ValueError("resume config differs from the frozen run config")
            resume_log = run_dir / "resume_log.json"
            entries = json.loads(resume_log.read_text()) if resume_log.exists() else []
            entries.append({
                "previous_total_env_steps": frozen["total_env_steps"],
                "new_total_env_steps": requested["total_env_steps"],
                "note": "continuation from the latest MODEL checkpoint: networks + env_steps restored; "
                        "replay, episode archive and collector state start EMPTY at the resume point",
            })
            resume_log.write_text(json.dumps(entries, indent=2) + "\n")
            config_path.write_text(json.dumps(requested, indent=2, sort_keys=True) + "\n")

    else:
        config_path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")
        reference_path.write_text(
            json.dumps(config.reference_manifest(), indent=2, sort_keys=True) + "\n"
        )

    experiment = make_experiment(config)
    state = experiment.trainer.initialize(jax.random.PRNGKey(config.seed))
    coverage_specs = coverage_track_specs(experiment.env, state.collector.env_state)
    coverage_masks = coverage_floor_masks(experiment.env, coverage_specs)
    checkpoints = CheckpointStore(run_dir / "checkpoints", keep=config.keep_checkpoints)
    if resume:
        manifest = checkpoints.latest_manifest()
        if manifest is not None and manifest.get("payload_kind") == "model":
            restored = checkpoints.restore(model_checkpoint(state))
            state = state.replace(
                sac=restored.sac,
                skill=restored.skill,
                env_steps=restored.env_steps,
            )
        else:
            state = checkpoints.restore(state)

    transitions_per_segment = config.rollout_steps * config.num_envs
    start_segment = int(np.asarray(jax.device_get(state.env_steps))) // transitions_per_segment
    start_env_steps = int(np.asarray(jax.device_get(state.env_steps)))
    start_branch_steps = int(
        np.asarray(jax.device_get(state.branch_simulator_steps))
    )
    metrics_path = run_dir / "metrics.jsonl"
    wandb_mirror = WandbMirror.start(run_dir, config.to_dict(), resume=resume)
    exploration_totals: dict[str, float] = {}
    holdout_cycle = HoldoutCycleAccumulator(
        required_events=experiment.trainer.branch_phase_groups,
        num_skills=config.num_skills,
    )
    completed_progress_targets: set[int] = set()
    if config.progress_eval_every_env_steps is not None:
        for path in (run_dir / "progress").glob("target_*/progress_snapshot.json"):
            manifest = json.loads(path.read_text())
            if manifest.get("state") == "complete":
                completed_progress_targets.add(int(manifest["target_env_steps"]))
    training_started = time.perf_counter()
    stop_reason = "total_env_steps"
    completed_segments = start_segment
    try:
        if (start_env_steps == 0 and config.progress_eval_every_env_steps is not None
                and config.environment in {"ant", "ant_soccer", "builderbench_corner3"}
                and 0 not in completed_progress_targets):
            checkpoints.save(0, model_checkpoint(state), config.to_dict(), payload_kind="model")
            initial = create_progress_snapshot(run_dir, config, target_env_steps=0, checkpoint_step=0)
            log_progress_snapshot(run_dir, wandb_mirror, initial)
            completed_progress_targets.add(0)
        training_started = time.perf_counter()
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            for segment in count(start_segment):
                previous_env_steps = int(np.asarray(jax.device_get(state.env_steps)))
                if previous_env_steps >= config.total_env_steps:
                    break
                start = time.perf_counter()
                state, metrics = experiment.trainer.run(state)
                jax.block_until_ready(state.env_steps)
                elapsed = time.perf_counter() - start
                train_seconds = time.perf_counter() - training_started
                completed_segments = segment + 1
                actual_env_steps = int(np.asarray(jax.device_get(state.env_steps)))
                reached_budget = actual_env_steps >= config.total_env_steps
                reached_smoke_limit = (stop_after_segments is not None
                                       and completed_segments - start_segment >= stop_after_segments)
                reached_time_limit = (
                    config.max_train_seconds is not None
                    and train_seconds >= config.max_train_seconds
                )
                _accumulate_exploration(exploration_totals, metrics, config.environment)
                holdout_cycle_summary = holdout_cycle.add(metrics)
                holdout_event_summary = summarize_holdout_event(metrics)
                should_log = (
                    segment == start_segment
                    or stop_after_segments is not None
                    or (segment + 1) % config.log_every_segments == 0
                    or reached_time_limit
                    or reached_budget
                    or reached_smoke_limit
                    or holdout_cycle_summary is not None
                )
                if should_log:
                    current_env_steps = int(np.asarray(jax.device_get(state.env_steps)))
                    live_steps_collected = current_env_steps - start_env_steps
                    current_branch_steps = int(
                        np.asarray(jax.device_get(state.branch_simulator_steps))
                    )
                    branch_steps_collected = current_branch_steps - start_branch_steps
                    timing_metrics = {
                        "train/wall_seconds": train_seconds,
                        "train/cumulative_sps": live_steps_collected / train_seconds,
                        "train/cumulative_live_sps": (
                            live_steps_collected / train_seconds
                        ),
                        "train/cumulative_total_simulator_sps": (
                            (live_steps_collected + branch_steps_collected)
                            / train_seconds
                        ),
                    }
                    row = {
                        "segment": segment + 1,
                        **summarize_metrics(metrics, elapsed, actual_env_steps - previous_env_steps),
                        **holdout_event_summary,
                        **(holdout_cycle_summary or {}),
                        **timing_metrics,
                        **_summarize_exploration(exploration_totals, config.environment),
                        # Cumulative training-rollout state coverage, split by the phase the lane
                        # was in (potential z=0 / skill / union), read off the collector carry.
                        **coverage_counts(
                            state.collector.coverage_grids,
                            coverage_specs,
                            coverage_masks,
                        ),
                    }
                    exploration_totals = {}
                    if config.reward_partition_anneal_end_steps:
                        row["sac/reward_partition_weight"] = _mean(
                            experiment.trainer.config.reward_partition_weight(state.env_steps)
                        )
                    metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                    metrics_file.flush()
                    print(json.dumps(row, sort_keys=True), flush=True)
                    wandb_mirror.log(row)
                if (segment + 1) % config.checkpoint_every_segments == 0:
                    checkpoints.save(
                        int(np.asarray(jax.device_get(state.env_steps))),
                        model_checkpoint(state),
                        config.to_dict(),
                        payload_kind="model",
                    )
                if config.progress_eval_every_env_steps is not None:
                    current_env_steps = int(np.asarray(jax.device_get(state.env_steps)))
                    due_targets = due_progress_targets(
                        current_env_steps,
                        config.total_env_steps,
                        config.progress_eval_every_env_steps,
                    )
                    for target in due_targets:
                        if target in completed_progress_targets:
                            continue
                        checkpoints.save(
                            current_env_steps,
                            model_checkpoint(state),
                            config.to_dict(),
                            payload_kind="model",
                        )
                        progress = create_progress_snapshot(
                            run_dir,
                            config,
                            target_env_steps=target,
                            checkpoint_step=current_env_steps,
                        )
                        log_progress_snapshot(run_dir, wandb_mirror, progress)
                        completed_progress_targets.add(target)
                if reached_time_limit or reached_smoke_limit:
                    stop_reason = "smoke_segments" if reached_smoke_limit else "max_train_seconds"
                    break
        train_seconds = time.perf_counter() - training_started
        final_step = int(np.asarray(jax.device_get(state.env_steps)))
        final_branch_steps = int(
            np.asarray(jax.device_get(state.branch_simulator_steps))
        )
        checkpoints.save(
            final_step,
            model_checkpoint(state),
            config.to_dict(),
            payload_kind="model",
        )
        if (config.progress_eval_every_env_steps is not None
                and final_step not in completed_progress_targets
                and final_step < config.total_env_steps):
            # Bounded checks must also exercise the final evaluation and media path.
            progress = create_progress_snapshot(run_dir, config,
                target_env_steps=final_step, checkpoint_step=final_step)
            log_progress_snapshot(run_dir, wandb_mirror, progress)
        live_steps_collected = final_step - start_env_steps
        branch_steps_collected = final_branch_steps - start_branch_steps
        (run_dir / "train_summary.json").write_text(
            json.dumps(
                {
                    "completed_segments": completed_segments - start_segment,
                    "env_steps_collected": live_steps_collected,
                    "branch_simulator_steps_collected": branch_steps_collected,
                    "total_simulator_steps_collected": (
                        live_steps_collected + branch_steps_collected
                    ),
                    "cumulative_live_sps": live_steps_collected / train_seconds,
                    "cumulative_total_simulator_sps": (
                        (live_steps_collected + branch_steps_collected) / train_seconds
                    ),
                    "final_env_steps": final_step,
                    "max_train_seconds": config.max_train_seconds,
                    "start_env_steps": start_env_steps,
                    "stop_reason": stop_reason,
                    "train_wall_seconds": train_seconds,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return run_dir
    finally:
        wandb_mirror.finish()
