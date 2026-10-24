"""SAC metric tuples, held-out cycle aggregation, and metric-row summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.agents.see_sac import (
    AnchorReplayMetrics,
    ArchiveMetrics,
    SACMetrics,
    _zero_skill_metrics,
)
from empowerment_warp.agents.skill_classifier import RelabelMetrics, SkillMetrics
from empowerment_warp.core.rollout import ExplorationMetrics


def _branch_fano_mi_lower_bound(accuracy: jax.Array, num_skills: int) -> jax.Array:
    """Calibration-free MI lower bound from accuracy alone (Fano's inequality).

    `log K - CE` is a VARIATIONAL bound: it reports a large negative number when
    the critic is merely miscalibrated, which is not the same as "no
    information". At tau=0.1 with phi capped at norm 5 the logits span ~+/-100,
    so a single confident miss costs ~100 nats and the bound goes vacuous long
    before the classifier is actually uninformative.

    Fano depends only on the error rate, so it cannot be moved by
    overconfidence: I(z; outcome | anchor) >= log K - H(pe) - pe * log(K - 1).
    It is exactly 0 at chance and rises with real accuracy. Report it beside the
    CE bound, never instead of it -- Fano is loose when the critic IS calibrated.
    """

    if num_skills < 2:
        # Continuous skills: no class count, Fano is undefined; the CE bound beside it stands.
        return jnp.zeros_like(jnp.asarray(accuracy, jnp.float32))
    acc = jnp.clip(accuracy, 1e-6, 1.0 - 1e-6)
    err = 1.0 - acc
    binary_entropy = -(acc * jnp.log(acc) + err * jnp.log(err))
    bound = (
        jnp.log(jnp.asarray(num_skills, jnp.float32))
        - binary_entropy
        - err * jnp.log(jnp.asarray(num_skills - 1, jnp.float32))
    )
    return jnp.maximum(bound, 0.0)

class BranchMetrics(NamedTuple):
    """Collection and archive-ingestion diagnostics for marginal branches."""

    applied: jax.Array
    simulator_steps: jax.Array
    valid_fraction: jax.Array
    target_count_min: jax.Array
    target_count_max: jax.Array
    source_target_match_fraction: jax.Array
    inserted_groups: jax.Array
    discarded_startup_groups: jax.Array
    collector_active: jax.Array
    archive_refresh_fraction: jax.Array
    holdout_accuracy: jax.Array
    holdout_mi_lower_bound: jax.Array
    holdout_loss: jax.Array
    holdout_intrinsic_reward: jax.Array
    holdout_mi_fano: jax.Array
    holdout_accuracy_clean: jax.Array
    holdout_mi_lower_bound_clean: jax.Array
    holdout_loss_clean: jax.Array
    holdout_mi_fano_clean: jax.Array
    holdout_sampled_rows: jax.Array
    holdout_trajectories: jax.Array
    holdout_physical_anchors: jax.Array
    holdout_cycle_eligible: jax.Array
    policy_rows_alive_fraction: jax.Array
    # How far a finished branch travelled and the total endpoint dispersion
    # within one shared-anchor block. The latter includes action noise.
    skill_disp_mean: jax.Array
    skill_endpoint_spread: jax.Array


class FreshBranchHoldout(NamedTuple):
    augmented: SkillMetrics
    clean: SkillMetrics
    sampled_rows: jax.Array


class UpdateMetrics(NamedTuple):
    sac: SACMetrics
    skill: SkillMetrics
    relabel: RelabelMetrics
    archive: ArchiveMetrics
    anchor_replay: AnchorReplayMetrics
    branch: BranchMetrics
    exploration: ExplorationMetrics
    replay_size: jax.Array
    representation_size: jax.Array
    representation_active_rows: jax.Array
    env_steps: jax.Array
    branch_simulator_steps_total: jax.Array
    update_applied: jax.Array


def _zero_branch_metrics() -> BranchMetrics:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return BranchMetrics(*(zero for _ in BranchMetrics._fields))


def _zero_fresh_branch_holdout() -> FreshBranchHoldout:
    return FreshBranchHoldout(
        augmented=_zero_skill_metrics(),
        clean=_zero_skill_metrics(),
        sampled_rows=jnp.asarray(0.0, dtype=jnp.float32),
    )


_EXPLORATION_FIELDS = {
    "builderbench_corner3": {
        "obj_moved_sum": "exploration/obj_moved",
        "cube_disp_xyz_sum": "exploration/episode_cube_disp_xyz",
        "cube_disp_max_xyz_sum": "exploration/episode_cube_disp_max_xyz",
        "cube_disp_xy_sum": "exploration/episode_cube_disp_xy",
        "cube_disp_max_xy_sum": "exploration/episode_cube_disp_max_xy",
        "switch_gripper_cube_dist_sum": "exploration/potential_end_gripper_cube_distance",
    },
    "ant": {
        "agent_disp_xy_sum": "exploration/episode_agent_disp_xy",
        "ball_disp_xy_sum": "exploration/episode_ball_disp_xy",
        "switch_center_dist_sum": "exploration/potential_end_center_distance",
        "switch_agent_ball_dist_sum": "exploration/potential_end_agent_ball_distance",
    },
    "ant_soccer": {
        "agent_disp_xy_sum": "exploration/episode_agent_disp_xy",
        "ball_disp_xy_sum": "exploration/episode_ball_disp_xy",
        "switch_center_dist_sum": "exploration/potential_end_center_distance",
        "switch_agent_ball_dist_sum": "exploration/potential_end_agent_ball_distance",
    },
}
# `switch_*_sum` fields are means over SWITCH events (the potential policy's state when a lane
# started its skill), everything else over completed episodes.


def _mean(value: Any) -> float:
    return float(np.asarray(jax.device_get(value), dtype=np.float64).mean())


@dataclass(slots=True)
class HoldoutCycleAccumulator:
    """Aggregate every fresh stagger phase before publishing held-out metrics."""

    required_events: int
    num_skills: int
    events: int = 0
    sampled_rows: float = 0.0
    trajectories: float = 0.0
    physical_anchors: float = 0.0
    correct: float = 0.0
    loss_sum: float = 0.0
    intrinsic_reward_sum: float = 0.0
    clean_correct: float = 0.0
    clean_loss_sum: float = 0.0
    mi_sum: float = 0.0
    clean_mi_sum: float = 0.0

    def add(self, metrics: Any) -> dict[str, float] | None:
        branch = jax.device_get(metrics.branch)
        if self.required_events <= 0 or _mean(branch.holdout_cycle_eligible) < 0.5:
            return None
        rows = _mean(branch.holdout_sampled_rows)

        self.events += 1
        self.trajectories += _mean(branch.holdout_trajectories)
        self.physical_anchors += _mean(branch.holdout_physical_anchors)
        if rows > 0.0:
            self.sampled_rows += rows
            self.correct += rows * _mean(branch.holdout_accuracy)
            self.loss_sum += rows * _mean(branch.holdout_loss)
            self.intrinsic_reward_sum += rows * _mean(
                branch.holdout_intrinsic_reward
            )
            self.clean_correct += rows * _mean(branch.holdout_accuracy_clean)
            self.clean_loss_sum += rows * _mean(branch.holdout_loss_clean)
            self.mi_sum += rows * _mean(branch.holdout_mi_lower_bound)
            self.clean_mi_sum += rows * _mean(branch.holdout_mi_lower_bound_clean)
        if self.events < self.required_events:
            return None
        if self.events != self.required_events:
            raise ValueError("held-out cycle accumulated too many phase events")
        if self.sampled_rows <= 0.0:
            raise ValueError("held-out cycle contained no valid sampled rows")

        accuracy = self.correct / self.sampled_rows
        loss = self.loss_sum / self.sampled_rows
        clean_accuracy = self.clean_correct / self.sampled_rows
        clean_loss = self.clean_loss_sum / self.sampled_rows
        summary = {
            "skill/accuracy_heldout": accuracy,
            "skill/loss_heldout": loss,
            "skill/mi_lower_bound_heldout": self.mi_sum / self.sampled_rows,
            "skill/intrinsic_reward_heldout": (
                self.intrinsic_reward_sum / self.sampled_rows
            ),
            "skill/accuracy_heldout_clean": clean_accuracy,
            "skill/loss_heldout_clean": clean_loss,
            "skill/mi_lower_bound_heldout_clean": (
                self.clean_mi_sum / self.sampled_rows
            ),
            "skill/heldout_sampled_rows": self.sampled_rows,
            "skill/heldout_trajectories": self.trajectories,
            "skill/heldout_physical_anchors": self.physical_anchors,
            "skill/heldout_phase_events": float(self.events),
        }
        self.events = 0
        self.sampled_rows = 0.0
        self.trajectories = 0.0
        self.physical_anchors = 0.0
        self.correct = 0.0
        self.loss_sum = 0.0
        self.intrinsic_reward_sum = 0.0
        self.clean_correct = 0.0
        self.clean_loss_sum = 0.0
        self.mi_sum = 0.0
        self.clean_mi_sum = 0.0
        return summary


def summarize_holdout_event(metrics: Any) -> dict[str, float]:
    """Return the current phase as a noisy diagnostic, never as the primary curve."""

    branch = metrics.branch
    if _mean(branch.holdout_cycle_eligible) < 0.5:
        return {}
    summary = {
        "skill/accuracy_heldout_event": _mean(branch.holdout_accuracy),
        "skill/loss_heldout_event": _mean(branch.holdout_loss),
        "skill/mi_lower_bound_heldout_event": _mean(
            branch.holdout_mi_lower_bound
        ),
        "skill/intrinsic_reward_heldout_event": _mean(
            branch.holdout_intrinsic_reward
        ),
        "skill/accuracy_heldout_clean_event": _mean(
            branch.holdout_accuracy_clean
        ),
        "skill/loss_heldout_clean_event": _mean(branch.holdout_loss_clean),
        "skill/mi_lower_bound_heldout_clean_event": _mean(
            branch.holdout_mi_lower_bound_clean
        ),
        "skill/heldout_sampled_rows_event": _mean(branch.holdout_sampled_rows),
        "skill/heldout_trajectories_event": _mean(branch.holdout_trajectories),
        "skill/heldout_physical_anchors_event": _mean(
            branch.holdout_physical_anchors
        ),
    }
    return summary


def summarize_potential_metrics(sac: Any) -> dict[str, float]:
    """Potential-row diagnostics, omitted when no such replay rows were trained."""
    weight = float(np.asarray(sac.potential_weight).sum())
    if weight <= 0:
        return {}
    total = lambda values: float(np.asarray(values).sum())
    result = {
        "potential/replay_fraction": weight / total(sac.replay_weight),
        "potential/reward_mean": total(sac.potential_reward_total) / weight,
        "potential/critic_q_mean": total(sac.potential_q_total) / weight,
        "potential/critic_target_mean": total(sac.potential_target_total) / weight,
        "potential/critic_td_rmse": float(np.sqrt(total(sac.potential_td_squared_total) / weight)),
    }
    value_weight = total(sac.potential_skill_value_weight)
    if value_weight > 0:
        result["potential/skill_value_mean"] = total(sac.potential_skill_value_total) / value_weight
    return result


def summarize_metrics(metrics: Any, seconds: float, transitions: int) -> dict[str, float]:
    branch_steps = _mean(metrics.branch.simulator_steps)
    total_simulator_steps = transitions + branch_steps
    return {
        "env_steps": _mean(metrics.env_steps[-1]),
        "live_env_steps": _mean(metrics.env_steps[-1]),
        "branch_simulator_steps": branch_steps,
        "cumulative_branch_simulator_steps": _mean(
            metrics.branch_simulator_steps_total[-1]
        ),
        "total_simulator_steps": total_simulator_steps,
        "replay_size": _mean(metrics.replay_size[-1]),
        "representation_size": _mean(metrics.representation_size[-1]),
        "representation_active_rows": _mean(metrics.representation_active_rows[-1]),
        "updates_applied": _mean(metrics.update_applied),
        "sps": transitions / seconds,
        "live_sps": transitions / seconds,
        "total_simulator_sps": total_simulator_steps / seconds,
        "branch/applied": _mean(metrics.branch.applied),
        "branch/valid_fraction": _mean(metrics.branch.valid_fraction),
        "branch/target_count_min": _mean(metrics.branch.target_count_min),
        "branch/target_count_max": _mean(metrics.branch.target_count_max),
        "branch/source_target_match_fraction": _mean(
            metrics.branch.source_target_match_fraction
        ),
        "branch/inserted_groups": _mean(metrics.branch.inserted_groups),
        "branch/discarded_startup_groups": _mean(
            metrics.branch.discarded_startup_groups
        ),
        "branch/collector_active": _mean(metrics.branch.collector_active),
        "branch/archive_refresh_fraction": _mean(
            metrics.branch.archive_refresh_fraction
        ),
        "branch/policy_rows_alive_fraction": _mean(
            metrics.branch.policy_rows_alive_fraction
        ),
        "branch/skill_disp_mean": _mean(metrics.branch.skill_disp_mean),
        # Total endpoint dispersion within shared-anchor blocks; this includes
        # stochastic action noise and is not a skill-only effect.
        "branch/skill_endpoint_spread": _mean(metrics.branch.skill_endpoint_spread),
        "sac/critic_loss": _mean(metrics.sac.critic_loss),
        "sac/actor_loss": _mean(metrics.sac.actor_loss),
        "sac/alpha": _mean(metrics.sac.alpha),
        "sac/entropy": _mean(metrics.sac.entropy),
        # The reward the POLICY receives on skill-on rows -- distinct from
        # "skill/intrinsic_reward", which is scored on the encoder batch.
        "sac/policy_reward_skill_on": _mean(metrics.sac.policy_reward_skill_on),
        **summarize_potential_metrics(metrics.sac),
        "skill/loss": _mean(metrics.skill.loss),
        "skill/intrinsic_reward": _mean(metrics.skill.intrinsic_reward_mean),
        "skill/accuracy": _mean(metrics.skill.accuracy),
        "skill/mi_lower_bound": _mean(metrics.skill.mi_lower_bound),
        "skill/alignment": _mean(metrics.skill.alignment),
        "skill/log_partition": _mean(metrics.skill.log_partition),
        # Lipschitz constraint diagnostics. A violation fraction pinned at 0 from
        # the start means the budget is mis-scaled and the constraint is inert --
        # the arm would then be testing nothing.
        "archive/future_valid_fraction": _mean(metrics.archive.valid_fraction),
        "archive/self_future_fraction": _mean(metrics.archive.self_future_fraction),
        "archive/mean_future_offset": _mean(metrics.archive.mean_future_offset),
        "archive/fresh_group_fraction": _mean(metrics.archive.fresh_group_fraction),
        "relabel/allowed_fraction": _mean(metrics.relabel.allowed_fraction),
        "relabel/relabeled_fraction": _mean(metrics.relabel.relabeled_fraction),
        "relabel/changed_fraction": _mean(metrics.relabel.changed_fraction),
        "relabel/confidence_mean": _mean(metrics.relabel.confidence_mean),
        "relabel/weight_mean": _mean(metrics.relabel.weight_mean),
        "relabel/weight_max": _mean(metrics.relabel.weight_max),
        "anchor_replay/true_fraction": _mean(metrics.anchor_replay.true_fraction),
        "anchor_replay/current_fraction": _mean(metrics.anchor_replay.current_fraction),
        "anchor_replay/her_fraction": _mean(metrics.anchor_replay.her_fraction),
        "anchor_replay/her_fallback_fraction": _mean(
            metrics.anchor_replay.her_fallback_fraction
        ),
        "anchor_replay/her_endpoint_available_fraction": _mean(
            metrics.anchor_replay.her_endpoint_available_fraction
        ),
        "anchor_replay/segment_end_available_fraction": _mean(
            metrics.anchor_replay.segment_end_available_fraction
        ),
    }


def _accumulate_exploration(
    totals: dict[str, float],
    metrics: Any,
    environment: str,
) -> None:
    fields = _EXPLORATION_FIELDS.get(environment, {})
    exploration = jax.device_get(metrics.exploration)
    totals["completed_episodes"] = totals.get("completed_episodes", 0.0) + float(
        np.asarray(exploration.completed_episodes)
    )
    totals["switch_count"] = totals.get("switch_count", 0.0) + float(np.asarray(exploration.switch_count))
    for field in fields:
        totals[field] = totals.get(field, 0.0) + float(np.asarray(getattr(exploration, field)))


def _summarize_exploration(totals: dict[str, float], environment: str) -> dict[str, float]:
    count = totals.get("completed_episodes", 0.0)
    summary = {"exploration/completed_episodes": count}
    if count <= 0.0:
        return summary
    switches = totals.get("switch_count", 0.0)
    for field, metric_name in _EXPLORATION_FIELDS.get(environment, {}).items():
        if field.startswith("switch_"):
            if switches > 0.0:
                summary[metric_name] = totals.get(field, 0.0) / switches
            continue
        summary[metric_name] = totals.get(field, 0.0) / count
    return summary
