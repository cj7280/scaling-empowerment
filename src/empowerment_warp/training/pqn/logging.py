"""Reference metric namespace for potential-PQN rows and the Craftax achievement keys."""

from __future__ import annotations

import jax
import jax.numpy as jnp


# Metric namespace applied at emit time: W&B and metrics.jsonl use three groups
# (training/, exploration/, evaluation/). Internal row keys are unchanged.
_REFERENCE_KEY_MAP = {
    "pqn/epsilon": "training/pqn/epsilon", "pqn/q_mean": "training/pqn/q_mean",
    "pqn/target_mean": "training/pqn/target_mean", "pqn/td_loss": "training/pqn/td_loss",
    "pqn/env_reward_mean": "training/pqn/env_reward_mean", "pqn/reward_mean": "training/pqn/env_reward_mean",
    "pqn/done_frac": "training/pqn/done_frac", "pqn/skill_on_frac": "training/reward/skill_on_frac",
    "reward/partition_weight": "training/reward/partition_weight",
    "reward/empowerment_mean": "training/reward/empowerment_mean",
    "reward/potential_value_mean": "training/reward/potential_value_mean",
    "reward/potential_to_skill_ratio": "training/reward/potential_to_skill_ratio",
    "reward/pqn_reward_mean": "training/reward/pqn_reward_mean",
    "reward/skill_on_frac": "training/reward/skill_on_frac",
    "relabel/changed_fraction": "training/relabel/achieved_skill_relabel_changed_frac",
    "relabel/confidence_mean": "training/relabel/achieved_skill_relabel_conf_mean",
    "relabel/relabeled_fraction": "training/relabel/achieved_skill_relabel_frac",
    "skill/accuracy": "training/representation_archive/encoder/accuracy",
    "skill/mi_lower_bound": "training/representation_archive/encoder/mi_lb",
    "skill/loss": "training/representation_archive/encoder/encoder_loss",
    "skill/alignment": "training/representation_archive/encoder/alignment",
    "skill/log_partition": "training/representation_archive/encoder/log_partition",
    "archive/valid_fraction": "training/representation_archive/encoder/valid_frac",
    "archive/skill_on_fraction": "training/representation_archive/encoder/skill_on_frac",
    "archive/mean_future_offset": "training/representation_archive/encoder/temporal_k_mean",
    "archive/self_future_fraction": "training/representation_archive/encoder/self_future_frac",
    "sps": "time/sps", "train/wall_seconds": "time/total_time",
    "segment": "training/pqn/collected_chunks",
    "eval/completed_episodes": "evaluation/completed_episodes",
    "eval/mean_completed_return": "evaluation/episode_return",
    "eval/episode_return_per_lane": "evaluation/episode_return_per_lane",
}


def _reference_keys(metrics: dict) -> dict:
    """Rename emitted keys into the reference namespace (env_steps kept as the x-axis)."""
    out = {}
    for key, value in metrics.items():
        if key in _REFERENCE_KEY_MAP:
            new = _REFERENCE_KEY_MAP[key]
        elif key.startswith("representation_archive/"):
            new = "training/" + key
        else:
            new = key
        out[new] = value
    return out


def _craftax_achievement_keys(env, num_envs: int) -> list[str]:
    """The env's per-step Achievements/* info keys (sorted, static); [] if none."""
    keys = jax.eval_shape(
        lambda k: env.step(env.reset(jax.random.split(k, num_envs)),
                           jnp.zeros((num_envs,), jnp.int32)).info,
        jax.random.PRNGKey(0),
    )
    return sorted(k for k in keys if str(k).startswith("Achievements/"))
