"""Typed, serializable experiment recipes with explicit reference-recipe parity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
import json
import math

import jax
import jax.numpy as jnp

from empowerment_warp.core.replay import (
    GEOMETRIC_ANCHOR_REPLAY_MODE,
    MARGINAL_BRANCH_SKILL_START_MODE,
    POTENTIAL_REPLAY_MODES,
    SKILL_START_MODES,
    TWO_PHASE_SKILL_START_MODE,
)


SkillMethod = Literal["potential"]

SAC_ENVIRONMENTS = frozenset({"ant", "ant_soccer", "builderbench_corner3"})


_REFERENCE_PROVENANCE = json.loads(Path(__file__).with_name("reference_provenance.json").read_text())

BUILDERBENCH_INCUMBENT_SAC = "empowerment:bb_base_corner3_block_only"
BUILDERBENCH_INCUMBENT_SOURCES = tuple(_REFERENCE_PROVENANCE["BUILDERBENCH_INCUMBENT_SOURCES"])
BUILDERBENCH_INCUMBENT_SOURCE_SHA256 = _REFERENCE_PROVENANCE["BUILDERBENCH_INCUMBENT_SOURCE_SHA256"]
BUILDERBENCH_REFERENCE_COMMIT = "971b6d77928c93b68b0633d80e6d3aeb46f614f3"
STANDING_POLICY_REPLAY_CAPACITY = 10_000_000
PRODUCTION_NUM_ENVS = 2_048
PRODUCTION_POLICY_UPDATE_SAMPLE_RATIO = 4
# Each 400-step episode has a 100-step zero-skill prefix, leaving 300 rows
# eligible for representation sampling. This is the smallest whole-episode bag
# with at least one million eligible rows: 3,334 * 300 = 1,000,200.
STANDING_REPRESENTATION_EPISODES = 3_334
BUILDERBENCH_INTENTIONAL_DEVIATIONS = _REFERENCE_PROVENANCE["BUILDERBENCH_INTENTIONAL_DEVIATIONS"]

# These fields define the learner rather than the GPU execution shape. A config
# carrying a reference recipe id is rejected if any drift unless the caller
# explicitly marks the run as a deviation.
BUILDERBENCH_INCUMBENT_SCIENTIFIC_FIELDS: dict[str, Any] = {
    "total_env_steps": 10_000_000,
    "episode_length": 400,
    "potential_steps": 100,
    "skill_dim": 9,
    "num_skills": 30,
    "discrete_skills": True,
    "discrete_skill_library": "random",
    "hidden_dims": (512, 512),
    "actor_layer_norm": False,
    "critic_layer_norm": True,
    "encoder_layer_norm": True,
    "encoder_obs_mode": "object_pos",
    "actor_final_scale": 0.01,
    "critic_final_scale": 1.0,
    # Deliberate deviation from the reference's 0.0 (see BUILDERBENCH_INTENTIONAL_DEVIATIONS).
    "skill_mean_init_std": 0.2,
    "continuous_reward_temperature": True,
    "achieved_skill_relabel_outcome": "discounted",
    "state_dependent_std": False,
    "target_entropy": -12.5,
    "discount": 0.99,
    "critic_tau": 0.02,
    "encoder_target_tau": 0.99,
    "encoder_temperature": 0.1,
    "phi_norm_cap": 0.0,
    "phi_noise_std": 0.005,
    "reward_encoder": "slow",
    "use_anchor_in_policy": True,
    "use_anchor_in_critic": True,
    "sample_hindsight_anchors": True,
    "potential_replay_mode": GEOMETRIC_ANCHOR_REPLAY_MODE,
    "achieved_skill_relabeling": True,
    "achieved_skill_relabel_min_prob": 2.0 / 30.0,
    "achieved_skill_relabel_weight_mode": "effective_number",
    "replace_skill_zero_reward_with_random_value": True,
    "separate_skill0_actor": True,
    "separate_skill0_critic": True,
    "discrete_reward_softmax": True,
    "clip_discrete_reward": True,
    "future_gamma": 0.99,
    "max_future_steps": 400,
    "batch_size": 256,
    "replay_capacity": STANDING_POLICY_REPLAY_CAPACITY,
    "representation_episodes": STANDING_REPRESENTATION_EPISODES,
    "staggered_resets": True,
}

ANT_STANDARD_42_STATE_SAC = "empowerment:ant_potential_base_antsoccer_standard_42_state"
ANTSOCCER_INCUMBENT_SAC = "empowerment:antsoccer_potential_base_closer_ball"
ANT_STANDARD_42_STATE_SOURCES = (
    "base_scripts/antsoccer/potential/ab_tests/phi_geometry/antsoccer_ppo_phigeom_ab.sh",
    "base_scripts/runners/antsoccer_rev_sac.sh",
    "src/empowerment/agents/potential_empowerment.py",
    "src/empowerment/utils/datasets.py",
    "src/empowerment/envs/ogbench/presets.py",
)
ANTSOCCER_INCUMBENT_SOURCES = tuple(_REFERENCE_PROVENANCE["ANTSOCCER_INCUMBENT_SOURCES"])
ANT_STANDARD_42_STATE_SOURCE_SHA256 = {
    "base_scripts/antsoccer/potential/ab_tests/phi_geometry/antsoccer_ppo_phigeom_ab.sh": (
        "117bc53d06a3dd9e8f8ac7c2102b0960028fdca576bf328af5c682a7bb21e4d4"
    ),
    "base_scripts/runners/antsoccer_rev_sac.sh": (
        "50a872ec8d3b3f340b90e772aec9d10ef268f8846f60a5b31166c85809cc495e"
    ),
    "src/empowerment/agents/potential_empowerment.py": (
        "6cc5823b4412d89d87fc0323fc221ec86dc27947896ef36898ddbacc0a0e8e27"
    ),
    "src/empowerment/utils/datasets.py": (
        "25de588aaabd2005b8c508341eaab804f97e7817ea3b804ceaf2373eccf06605"
    ),
    "src/empowerment/envs/ogbench/presets.py": (
        "6b67b7031f8ae91a1936a9d128cfc1d2b3b56b8853b653f9dddf8a2f24a2af38"
    ),
}
ANTSOCCER_INCUMBENT_SOURCE_SHA256 = _REFERENCE_PROVENANCE["ANTSOCCER_INCUMBENT_SOURCE_SHA256"]
ANTSOCCER_REFERENCE_COMMIT = "62fbb57813892fd7dd46e9d65d169487c90fc66d"
ANTSOCCER_INTENTIONAL_DEVIATIONS = _REFERENCE_PROVENANCE["ANTSOCCER_INTENTIONAL_DEVIATIONS"]
ANT_STANDARD_42_STATE_INTENTIONAL_DEVIATIONS = {
    name: details
    for name, details in ANTSOCCER_INTENTIONAL_DEVIATIONS.items()
    if name != "encoder_training_noise"
}
ANTSOCCER_INCUMBENT_SCIENTIFIC_FIELDS: dict[str, Any] = {
    "total_env_steps": 10_000_000,
    "episode_length": 400,
    "potential_steps": 100,
    "skill_dim": 2,
    "num_skills": 30,
    "discrete_skills": True,
    "discrete_skill_library": "auto",
    "hidden_dims": (512, 512),
    "actor_layer_norm": False,
    "critic_layer_norm": True,
    "encoder_layer_norm": True,
    "encoder_obs_mode": "ball_xy",
    "antsoccer_preset": "closer_ball",
    "actor_final_scale": 0.0,
    "critic_final_scale": 1.0,
    "skill_mean_init_std": 0.2,
    "continuous_reward_temperature": True,
    "achieved_skill_relabel_outcome": "discounted",
    "state_dependent_std": True,
    "target_entropy": 0.0,
    "discount": 0.99,
    "critic_tau": 0.02,
    "encoder_target_tau": 0.99,
    "encoder_temperature": 0.1,
    "phi_norm_cap": 5.0,
    "phi_noise_std": 0.1,
    "reward_encoder": "slow",
    "use_anchor_in_policy": True,
    "use_anchor_in_critic": True,
    "sample_hindsight_anchors": True,
    "potential_replay_mode": GEOMETRIC_ANCHOR_REPLAY_MODE,
    "achieved_skill_relabeling": True,
    "achieved_skill_relabel_min_prob": 2.0 / 30.0,
    "achieved_skill_relabel_weight_mode": "effective_number",
    "replace_skill_zero_reward_with_random_value": True,
    "separate_skill0_actor": True,
    "separate_skill0_critic": True,
    "discrete_reward_softmax": True,
    "clip_discrete_reward": True,
    "future_gamma": 0.99,
    "max_future_steps": 400,
    "batch_size": 1024,
    "replay_capacity": STANDING_POLICY_REPLAY_CAPACITY,
    "representation_episodes": STANDING_REPRESENTATION_EPISODES,
    "staggered_resets": True,
    "skill_start_mode": TWO_PHASE_SKILL_START_MODE,
}

# The standing "Ant" comparison keeps AntSoccer dynamics and the ball. It uses
# the standard reset, and both the policy and empowerment encoder receive all
# 42 state values: 22 qpos values followed by 20 qvel values.
ANT_STANDARD_42_STATE_SCIENTIFIC_FIELDS: dict[str, Any] = {
    **ANTSOCCER_INCUMBENT_SCIENTIFIC_FIELDS,
    "encoder_obs_mode": "full",
    "antsoccer_preset": "standard",
    "phi_noise_std": 0.05,
}


# Knobs that only discrete skills would read, pinned to the one value every recipe and saved
# SAC config carries. Their YAML keys stay valid.
_INERT_DISCRETE_KNOBS: dict[str, Any] = {
    "achieved_skill_relabel_min_prob": 2.0 / 30.0,
    "achieved_skill_relabel_weight_mode": "effective_number",
    "achieved_skill_relabel_effective_beta": 0.99,
    "achieved_skill_relabel_weight_max": 3.0,
    "discrete_reward_softmax": True,
    "clip_discrete_reward": True,
}


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    environment: str
    algorithm: SkillMethod = "potential"
    recipe_id: str = "custom"
    allow_reference_deviation: bool = False
    seed: int = 0
    num_envs: int = 256
    total_env_steps: int = 100_000_000
    max_train_seconds: float | None = None
    episode_length: int = 200
    potential_steps: int = 0
    random_potential_switch_max_steps: int = 0  # 0 keeps the fixed prefix
    skill_start_mode: str = TWO_PHASE_SKILL_START_MODE
    skill_dim: int = 8
    num_skills: int = 16
    discrete_skills: bool = False
    discrete_skill_library: str = "random"
    hidden_dims: tuple[int, ...] = (512, 512)
    actor_layer_norm: bool = True
    critic_layer_norm: bool = True
    encoder_layer_norm: bool = True
    encoder_obs_mode: str = "full"
    antsoccer_preset: str = "standard"
    actor_final_scale: float = 1.0
    critic_final_scale: float = 1.0
    skill_mean_init_std: float = 0.0
    state_dependent_std: bool = True
    target_entropy: float | None = None
    discount: float = 0.99
    critic_tau: float = 0.02
    encoder_target_tau: float = 0.005
    encoder_temperature: float = 0.1
    encoder_objective: str = "auto"
    phi_norm_cap: float = 5.0
    phi_noise_std: float = 0.0
    # 'current' makes the potential reward a per-step delta. It telescopes to the
    # same return as 'anchor' but is dense, so credit assignment is local. Only
    # telescopes for the RAW inner product -- pair with discrete_reward_softmax=False.
    potential_reward_reference: str = "anchor"
    # Multiply the potential reward to match the anchor form's effective magnitude
    # (~1/(1-gamma)) so a delta run's SAC dynamics are comparable. 1.0 = unchanged.
    potential_reward_scale: float = 1.0
    continuous_reward_temperature: bool = False
    continuous_reward_normalization: str = "none"
    reward_partition_anneal_start_steps: int = 0
    reward_partition_anneal_end_steps: int = 0  # 0 disables the schedule
    reward_encoder: str = "live"  # encoder copy: "live" (trained params) or "slow" (their EMA); "online"/"target" also accepted
    use_anchor_in_policy: bool = False
    use_anchor_in_critic: bool = False
    sample_hindsight_anchors: bool = False
    potential_replay_mode: str = GEOMETRIC_ANCHOR_REPLAY_MODE
    achieved_skill_relabeling: bool = False
    achieved_skill_relabel_min_prob: float = 0.0
    # Continuous relabel gate on ||G|| (representation units); 0 = reference parity (no gate).
    achieved_skill_relabel_min_norm: float = 0.0
    # Encoder InfoNCE labels: "commanded" (reference) or "achieved_discounted" (the actor/critic
    # HER label, unit(G) over M discounted futures, stop-grad; continuous skills only).
    encoder_skill_source: str = "commanded"
    # Pair form: "difference" = bilinear <phi(s')-phi(s), z> (the method); "pair_mlp" = <f(s,s'), z>
    # with f an MLP over the concatenated pair (arbitrary decoder q(z|s,s'); potential method only).
    pair_form: str = "difference"
    # "No policy switching" ablation: fraction of lanes running z=0 for whole episodes, the rest one
    # skill from reset (requires potential_steps=0). 0 = the two-phase method.
    potential_lane_fraction: float = 0.0
    achieved_skill_relabel_weight_mode: str = "none"
    # Per-row Bernoulli share of relabel-eligible rows that actually relabel; the
    # complement keeps its commanded skill. 1.0 relabels every confident row. The knob exists because the confidence gate has no bite for
    # continuous skills, so composition is the only relabeled/real control there.
    achieved_skill_relabel_fraction: float = 1.0
    # Hindsight label target: "current" relabels a replay row to the skill achieved from its
    # (geometric) anchor to the row's OWN state, so the label drifts along the segment;
    # "segment_end" relabels to the skill achieved from that anchor to the END of the skill
    # segment (the episode's final state), one consistent label per segment -- the PQN/Craftax
    # port's semantics. Base = "current" (what every SAC standing run has used).
    achieved_skill_relabel_outcome: str = "current"
    # "discounted": M futures per row drawn from Geometric(1 - future_gamma) capped at the segment
    # end; the relabel aggregates them in representation space (the discounted greedy relabel).
    achieved_skill_relabel_samples: int = 8
    achieved_skill_relabel_effective_beta: float = 0.99
    achieved_skill_relabel_weight_max: float = 3.0
    # Paper: z = 0 rows earn the potential reward E_z[V(s, z)] and train a separate
    # potential actor/critic; "skill0"/"random_value" are the serialized YAML names.
    replace_skill_zero_reward_with_random_value: bool = False
    separate_skill0_actor: bool = False
    separate_skill0_critic: bool = False
    discrete_reward_softmax: bool = False
    clip_discrete_reward: bool = False
    future_gamma: float = 0.99
    max_future_steps: int = 200
    rollout_steps: int = 16
    updates_per_chunk: int = 16
    encoder_updates_per_chunk: int = 0
    branch_transitions_to_policy_replay: bool = False
    # Lanes sharing one anchor, each with a distinct skill. 1 lets the
    # encoder memorise anchors instead of reading the continuation.
    branch_targets_per_anchor: int = 1
    # 0 = derive from `episode_length // rollout_steps`. 1 completes every lane's
    # branch each segment, which is what directed information (H=1) needs.
    branch_phase_groups: int = 0
    skill0_value_num_skills: int = 8  # Monte-Carlo z draws for the potential value
    # Live lanes run the potential policy (z=0) for their whole episode, so
    # branch anchors come from the potential policy's marginal and skill-on
    # rows are exactly the branch rows.
    live_potential_only: bool = False
    # Encoder-only branches with the standard live potential -> skill schedule.
    branch_live_two_phase: bool = False
    batch_size: int = 1024
    replay_capacity: int = 1_000_000
    representation_episodes: int = 64
    staggered_resets: bool = True
    log_every_segments: int = 10
    checkpoint_every_segments: int = 1_000
    progress_eval_every_env_steps: int | None = None
    keep_checkpoints: int = 0
    run_dir: str = "artifacts/runs/default"
    allow_cpu: bool = False
    allow_unpromoted_physics: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if self.encoder_objective == "auto":
            objective = ("forward_infonce" if self.algorithm == "potential" and not self.discrete_skills
                         else "symmetric_infonce")
            object.__setattr__(self, "encoder_objective", objective)
        if self.environment not in SAC_ENVIRONMENTS:
            raise ValueError(f"unknown environment: {self.environment}")
        if self.algorithm != "potential":
            raise ValueError(f"unknown algorithm: {self.algorithm}")
        positive = (
            self.num_envs,
            self.total_env_steps,
            self.episode_length,
            self.skill_dim,
            self.num_skills,
            self.rollout_steps,
            self.updates_per_chunk,
            self.batch_size,
            self.replay_capacity,
            self.representation_episodes,
            self.log_every_segments,
            self.checkpoint_every_segments,
            self.max_future_steps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("sizes, cadences, and budgets must be positive")
        if self.max_train_seconds is not None and self.max_train_seconds <= 0.0:
            raise ValueError("max_train_seconds must be positive when set")
        if (
            self.progress_eval_every_env_steps is not None
            and self.progress_eval_every_env_steps <= 0
        ):
            raise ValueError("progress_eval_every_env_steps must be positive when set")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.discrete_skill_library not in {"auto", "random"}:
            raise ValueError("discrete_skill_library must be 'auto' or 'random'")
        # Only discrete skills would read these (SkillLearner's K-way reward and relabel),
        # which are not implemented; every recipe uses these values.
        for name, value in _INERT_DISCRETE_KNOBS.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} only affects discrete skills, which are not implemented; it must be {value!r}")
        if min(self.actor_final_scale, self.critic_final_scale, self.skill_mean_init_std) < 0.0:
            raise ValueError("network initialization scales must be nonnegative")
        if not 0.0 <= self.discount <= 1.0 or not 0.0 < self.critic_tau <= 1.0:
            raise ValueError("discount must be in [0,1] and critic_tau in (0,1]")
        if not 0.0 < self.encoder_target_tau <= 1.0 or self.encoder_temperature <= 0.0:
            raise ValueError("encoder target tau and temperature must be positive")
        if self.encoder_objective not in {"symmetric_infonce", "forward_infonce", "vmf_forward"}:
            raise ValueError("unsupported encoder_objective")
        if self.encoder_objective == "symmetric_infonce" and self.algorithm == "potential" and not self.discrete_skills:
            raise ValueError("symmetric continuous-skill CSF is not supported; select forward_infonce or vmf_forward")
        if self.encoder_objective == "forward_infonce" and (
            self.algorithm != "potential" or self.discrete_skills
            or self.encoder_skill_source not in {"commanded", "achieved_discounted"}
        ):
            raise ValueError("forward_infonce requires potential, continuous skills")
        if self.encoder_objective == "vmf_forward" and (
            self.algorithm != "potential" or self.discrete_skills or self.skill_dim not in (2, 4, 9, 16)
            or self.encoder_skill_source != "commanded"
        ):
            raise ValueError("vmf_forward requires potential, continuous 2/4/9/16-D commanded skills")
        if self.potential_reward_reference not in {"anchor", "current"}:
            raise ValueError("potential_reward_reference must be 'anchor' or 'current'")
        if self.random_potential_switch_max_steps < 0:
            raise ValueError("random potential switch maximum must be nonnegative")
        if self.random_potential_switch_max_steps:
            if (
                self.environment != "ant" or self.algorithm != "potential"
                or self.skill_start_mode != TWO_PHASE_SKILL_START_MODE
                or self.live_potential_only or self.potential_lane_fraction != 0
                or self.random_potential_switch_max_steps >= self.episode_length
                or 2 * self.potential_steps != self.random_potential_switch_max_steps
            ):
                raise ValueError(
                    "random potential switching requires Ant two_phase, no separate potential "
                    "lanes/density/live-only mode, max < episode_length, and potential_steps=max/2"
                )
        if self.continuous_reward_normalization not in {"none", "sphere", "sampled_infonce"}:
            raise ValueError("unsupported continuous_reward_normalization")
        if self.reward_partition_anneal_end_steps == 0:
            if self.reward_partition_anneal_start_steps != 0:
                raise ValueError("partition annealing requires an end step")
        elif not 0 <= self.reward_partition_anneal_start_steps < self.reward_partition_anneal_end_steps:
            raise ValueError("partition annealing requires 0 <= start < end")
        elif (
            (self.continuous_reward_normalization, self.encoder_objective) not in {
                ("sphere", "vmf_forward"), ("sphere", "forward_infonce"),
                ("sampled_infonce", "forward_infonce"),
            }
            or self.potential_reward_reference != "anchor"
        ):
            raise ValueError("partition annealing requires matched normalized decoder anchor rewards")
        if (self.pair_form == "pair_mlp" and self.continuous_reward_normalization == "sphere"
                and self.potential_reward_reference != "anchor"):
            raise ValueError("normalized pair_mlp requires anchor rewards")
        if self.continuous_reward_normalization == "sphere" and (
            self.algorithm != "potential" or self.discrete_skills or self.skill_dim not in (2, 4, 9, 16)
            or not self.continuous_reward_temperature
        ):
            raise ValueError("sphere reward requires continuous 2/4/9/16-D potential scores with temperature")
        if self.phi_noise_std < 0.0:
            raise ValueError("phi_noise_std must be nonnegative")
        if self.reward_encoder not in {"live", "slow"}:
            raise ValueError("reward_encoder must be 'live' or 'slow'")
        if not self.encoder_obs_mode:
            raise ValueError("encoder_obs_mode must be nonempty")
        if self.antsoccer_preset not in {
            "standard",
            "closer_ball",
            "center_ball",
            "past_center_ball",
        }:
            raise ValueError(
                "antsoccer_preset must be standard, closer_ball, center_ball, or "
                "past_center_ball"
            )
        if (
            self.environment in {"ant", "ant_soccer"}
            and self.antsoccer_preset == "standard"
            and self.encoder_obs_mode == "ball_xy"
        ):
            raise ValueError(
                "standard AntSoccer with a ball_xy-only encoder is a known dead task: "
                "the initial policy cannot reach the distant ball. Use the "
                "past_center_ball preset (ball just past the arena center on the "
                "ant->ball line) or the closer_ball reset."
            )
        if not 0.0 <= self.future_gamma < 1.0:
            raise ValueError("future_gamma must be in [0,1)")
        if (
            self.branch_transitions_to_policy_replay
            and self.skill_start_mode != MARGINAL_BRANCH_SKILL_START_MODE
        ):
            raise ValueError(
                "branch_transitions_to_policy_replay requires "
                "skill_start_mode=marginal_branch: no other mode collects "
                "branch rollouts"
            )
        if self.skill0_value_num_skills < 1:
            raise ValueError("skill0_value_num_skills must be at least 1")
        if self.encoder_updates_per_chunk < 0:
            raise ValueError("encoder_updates_per_chunk must be nonnegative")
        if self.encoder_updates_per_chunk:
            if self.encoder_updates_per_chunk > self.updates_per_chunk:
                raise ValueError(
                    "encoder_updates_per_chunk cannot exceed updates_per_chunk"
                )
            if self.updates_per_chunk % self.encoder_updates_per_chunk:
                raise ValueError(
                    "encoder_updates_per_chunk must divide updates_per_chunk"
                )
        if self.potential_replay_mode not in POTENTIAL_REPLAY_MODES:
            raise ValueError(
                "potential_replay_mode must be 'legacy_geometric'"
            )
        if self.skill_start_mode not in SKILL_START_MODES:
            raise ValueError(
                "skill_start_mode must be 'two_phase' or 'marginal_branch'"
            )
        if self.branch_live_two_phase and (
            self.skill_start_mode != MARGINAL_BRANCH_SKILL_START_MODE
            or self.live_potential_only
        ):
            raise ValueError("branch_live_two_phase requires ordinary encoder-only marginal branches")
        if self.skill_start_mode == MARGINAL_BRANCH_SKILL_START_MODE:
            marginal_requirements = {
                "environment in {'ant', 'ant_soccer', 'builderbench_corner3'}": self.environment
                in {"ant", "ant_soccer", "builderbench_corner3"},
                "algorithm='potential'": self.algorithm == "potential",
                "branch_targets_per_anchor==1 for continuous skills": (
                    self.discrete_skills or self.branch_targets_per_anchor == 1
                ),
                "staggered_resets=True": self.staggered_resets,
                "potential_steps>0": self.potential_steps > 0,
                "episode_length divisible by rollout_steps": (
                    self.episode_length % self.rollout_steps == 0
                ),
                "representation_episodes>=num_envs": (
                    self.representation_episodes >= self.num_envs
                ),
            }
            missing = [
                name
                for name, satisfied in marginal_requirements.items()
                if not satisfied
            ]
            if missing:
                raise ValueError("marginal_branch requires " + ", ".join(missing))
        if self.achieved_skill_relabel_min_norm < 0.0:
            raise ValueError("achieved_skill_relabel_min_norm must be nonnegative")
        if self.encoder_skill_source not in {"commanded", "achieved_discounted"}:
            raise ValueError("encoder_skill_source must be 'commanded' or 'achieved_discounted'")
        if self.pair_form not in {"difference", "pair_mlp"}:
            raise ValueError("pair_form must be 'difference' or 'pair_mlp'")
        if not 0.0 <= self.potential_lane_fraction < 1.0:
            raise ValueError("potential_lane_fraction must be in [0, 1)")
        if self.potential_lane_fraction > 0.0 and self.potential_steps != 0:
            raise ValueError("potential_lane_fraction > 0 (separate lanes) requires potential_steps == 0")
        if self.achieved_skill_relabel_outcome not in {"current", "segment_end", "discounted"}:
            raise ValueError("achieved_skill_relabel_outcome must be 'current', 'segment_end' or 'discounted'")
        if self.achieved_skill_relabel_samples < 1:
            raise ValueError("achieved_skill_relabel_samples must be at least 1")
        if not 0.0 < self.achieved_skill_relabel_fraction <= 1.0:
            raise ValueError(
                "achieved relabel fraction must be in (0,1]; use "
                "achieved_skill_relabeling=False to disable relabeling entirely"
            )
        if self.keep_checkpoints < 0:
            raise ValueError("keep_checkpoints must be nonnegative; zero keeps every checkpoint")
        chunk_size = self.rollout_steps * self.num_envs
        if chunk_size > self.replay_capacity:
            raise ValueError("one rollout chunk must fit in the policy replay")
        simultaneous = (
            math.ceil(self.num_envs / max(1, min(self.episode_length // self.rollout_steps, self.num_envs)))
            if self.staggered_resets
            else self.num_envs
        )
        if self.representation_episodes < simultaneous:
            raise ValueError(
                f"representation_episodes={self.representation_episodes} must be at least the "
                f"{simultaneous} lanes that can finish an episode on the same step"
            )
        for recipe_id, environment, label in (
            (BUILDERBENCH_INCUMBENT_SAC, "builderbench_corner3", "reference BuilderBench SAC"),
            (ANTSOCCER_INCUMBENT_SAC, "ant_soccer", "reference AntSoccer SAC"),
            (ANT_STANDARD_42_STATE_SAC, "ant", "standard-reset 42-state Ant SAC"),
        ):
            if self.recipe_id != recipe_id:
                continue
            if self.environment != environment or self.algorithm != "potential":
                raise ValueError(f"{label} recipe only applies to {environment} Potential/SAC")
            mismatches = self.reference_mismatches()
            if mismatches and not self.allow_reference_deviation:
                details = ", ".join(
                    f"{name}={actual!r} (expected {expected!r})"
                    for name, (actual, expected) in mismatches.items()
                )
                raise ValueError(
                    f"{label} drift detected: {details}; set allow_reference_deviation=True "
                    "only for an explicit ablation or smoke"
                )

    @property
    def resolved_run_dir(self) -> Path:
        return Path(self.run_dir).expanduser().resolve()

    @property
    def uses_soccer_dynamics(self) -> bool:
        """Whether this run includes the AntSoccer ball, arena, and goal."""

        # True for every "ant" recipe: configs/ant/see.yaml (and default_recipe("ant"),
        # used by the Ant baselines) carry this recipe_id, so Ant runs with the AntSoccer
        # ball and arena (42-state observation).
        return self.environment == "ant_soccer" or self.recipe_id == ANT_STANDARD_42_STATE_SAC

    @property
    def locomotion_observation_contract(self) -> str:
        if self.uses_soccer_dynamics and self.encoder_obs_mode == "full":
            return (
                "AntSoccer dynamics; policy and encoder each receive all 42 state values: "
                "22 configuration values followed by 20 velocity values"
            )
        if self.uses_soccer_dynamics and self.encoder_obs_mode == "ball_xy":
            return (
                "AntSoccer dynamics; policy receives all 42 state values and encoder "
                "receives only ball x and y"
            )
        return "no-ball Ant dynamics; policy and encoder each receive all 29 state values"

    @property
    def policy_update_sample_ratio(self) -> float:
        return self.updates_per_chunk * self.batch_size / (self.rollout_steps * self.num_envs)

    @property
    def resolved_live_potential_steps(self) -> int:
        return (
            0
            if self.skill_start_mode == MARGINAL_BRANCH_SKILL_START_MODE
            and not self.branch_live_two_phase
            else self.potential_steps
        )

    @property
    def resolved_branch_horizon(self) -> int:
        if self.skill_start_mode != MARGINAL_BRANCH_SKILL_START_MODE:
            return 0
        return self.episode_length - self.potential_steps

    @property
    def branch_simulator_to_live_ratio(self) -> float:
        if self.skill_start_mode != MARGINAL_BRANCH_SKILL_START_MODE:
            return 0.0
        return self.resolved_branch_horizon / self.episode_length

    @property
    def resolved_branch_phase_groups(self) -> int:
        if self.skill_start_mode != MARGINAL_BRANCH_SKILL_START_MODE:
            return 0
        if self.branch_phase_groups:
            return self.branch_phase_groups
        return self.episode_length // self.rollout_steps

    @property
    def resolved_branch_steps_per_segment(self) -> int:
        if self.skill_start_mode != MARGINAL_BRANCH_SKILL_START_MODE:
            return 0
        return self.resolved_branch_horizon // self.resolved_branch_phase_groups

    @property
    def resolved_branch_completions_per_segment(self) -> int:
        if self.skill_start_mode != MARGINAL_BRANCH_SKILL_START_MODE:
            return 0
        return math.ceil(
            self.num_envs / self.resolved_branch_phase_groups
        )

    def reference_mismatches(self) -> dict[str, tuple[Any, Any]]:
        if self.recipe_id == BUILDERBENCH_INCUMBENT_SAC:
            expected_fields = BUILDERBENCH_INCUMBENT_SCIENTIFIC_FIELDS
            expected_ratio = float(PRODUCTION_POLICY_UPDATE_SAMPLE_RATIO)
        elif self.recipe_id == ANTSOCCER_INCUMBENT_SAC:
            expected_fields = ANTSOCCER_INCUMBENT_SCIENTIFIC_FIELDS
            expected_ratio = float(PRODUCTION_POLICY_UPDATE_SAMPLE_RATIO)
        elif self.recipe_id == ANT_STANDARD_42_STATE_SAC:
            expected_fields = ANT_STANDARD_42_STATE_SCIENTIFIC_FIELDS
            expected_ratio = float(PRODUCTION_POLICY_UPDATE_SAMPLE_RATIO)
        else:
            return {}
        mismatches: dict[str, tuple[Any, Any]] = {}
        if self.random_potential_switch_max_steps:
            mismatches["random_potential_switch_max_steps"] = (
                self.random_potential_switch_max_steps, 0
            )
        for name, expected in expected_fields.items():
            actual = getattr(self, name)
            if actual != expected:
                mismatches[name] = (actual, expected)
        if self.policy_update_sample_ratio != expected_ratio:
            mismatches["policy_update_sample_ratio"] = (
                self.policy_update_sample_ratio,
                expected_ratio,
            )
        return mismatches

    def reference_manifest(self) -> dict[str, Any]:
        mismatches = self.reference_mismatches()
        if self.recipe_id == BUILDERBENCH_INCUMBENT_SAC:
            semantics = "immutable_stable_builderbench_training_contract"
            commit = BUILDERBENCH_REFERENCE_COMMIT
            sources = BUILDERBENCH_INCUMBENT_SOURCES
            hashes = BUILDERBENCH_INCUMBENT_SOURCE_SHA256
            deviations = BUILDERBENCH_INTENTIONAL_DEVIATIONS
        elif self.recipe_id == ANTSOCCER_INCUMBENT_SAC:
            semantics = "immutable_stable_antsoccer_closer_ball_training_contract"
            commit = ANTSOCCER_REFERENCE_COMMIT
            sources = ANTSOCCER_INCUMBENT_SOURCES
            hashes = ANTSOCCER_INCUMBENT_SOURCE_SHA256
            deviations = ANTSOCCER_INTENTIONAL_DEVIATIONS
        elif self.recipe_id == ANT_STANDARD_42_STATE_SAC:
            semantics = "ant_comparison_uses_antsoccer_dynamics_standard_reset_policy_42_values_encoder_same_42_values"
            commit = ANTSOCCER_REFERENCE_COMMIT
            sources = ANT_STANDARD_42_STATE_SOURCES
            hashes = ANT_STANDARD_42_STATE_SOURCE_SHA256
            deviations = ANT_STANDARD_42_STATE_INTENTIONAL_DEVIATIONS
        else:
            semantics = "custom"
            commit = None
            sources = ()
            hashes = {}
            deviations = {}
        return {
            "recipe_id": self.recipe_id,
            "reference_semantics": semantics,
            "reference_commit": commit,
            "reference_sources": [
                {"path": path, "sha256": hashes[path]}
                for path in sources
            ],
            "contract_fields_match": not mismatches,
            "exact_reference_parity": not mismatches
            and not deviations,
            "parity_with_declared_deviations": not mismatches,
            "intentional_deviations": deviations,
            "deviation_allowed": self.allow_reference_deviation,
            "mismatches": {
                name: {"actual": actual, "expected": expected}
                for name, (actual, expected) in mismatches.items()
            },
            "policy_update_sample_ratio": self.policy_update_sample_ratio,
            "potential_replay_mode": self.potential_replay_mode,
            "skill_start_mode": self.skill_start_mode,
            "resolved_live_potential_steps": self.resolved_live_potential_steps,
            "potential_critic_objective": "potential_reward_return",
            "resolved_branch_horizon": self.resolved_branch_horizon,
            "branch_interval_live_vector_steps": (
                self.episode_length
                if self.skill_start_mode == MARGINAL_BRANCH_SKILL_START_MODE
                else 0
            ),
            "branch_simulator_to_live_ratio": self.branch_simulator_to_live_ratio,
            "branch_transitions_to_policy_replay": (
                self.branch_transitions_to_policy_replay
            ),
            "encoder_updates_per_chunk": (
                self.encoder_updates_per_chunk or self.updates_per_chunk
            ),
            "encoder_follows_sac_replay_ratio": (
                not self.encoder_updates_per_chunk
            ),
            "branch_phase_groups": (
                self.resolved_branch_phase_groups
            ),
            "branch_steps_per_segment": (
                self.resolved_branch_steps_per_segment
            ),
            "branch_completions_per_segment": (
                self.resolved_branch_completions_per_segment
            ),
            "representation_archive_unit": (
                "marginal_anchor_branch"
                if self.skill_start_mode == MARGINAL_BRANCH_SKILL_START_MODE
                else "completed_episode"
            ),
            "representation_groups": self.representation_episodes,
            "representation_active_rows_per_group": (
                None if self.random_potential_switch_max_steps else self.episode_length - self.potential_steps
            ),
            "representation_active_row_capacity": (
                None if self.random_potential_switch_max_steps else
                self.representation_episodes * (self.episode_length - self.potential_steps)
            ),
            **({
                "potential_switch_distribution": "uniform_inclusive",
                "potential_switch_min_steps": 0,
                "potential_switch_max_steps": self.random_potential_switch_max_steps,
                "expected_potential_steps": self.random_potential_switch_max_steps / 2,
                "representation_active_rows_per_group_expected": (
                    self.episode_length - self.random_potential_switch_max_steps / 2
                ),
                "representation_active_row_capacity_expected": self.representation_episodes * (
                    self.episode_length - self.random_potential_switch_max_steps / 2
                ),
                "representation_active_row_capacity_bounds": [
                    self.representation_episodes * (self.episode_length - self.random_potential_switch_max_steps),
                    self.representation_episodes * self.episode_length,
                ],
            } if self.random_potential_switch_max_steps else {}),
            "branch_phase_groups_override": self.branch_phase_groups,
            "branch_targets_per_anchor": (
                self.branch_targets_per_anchor
                if self.skill_start_mode == MARGINAL_BRANCH_SKILL_START_MODE
                else 0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_dims"] = list(self.hidden_dims)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        values = dict(payload)
        retired_defaults = {
            "achieved_skill_relabel_old_replay_only": False,
            "achieved_skill_relabel_recent_window": 0,
            "geometric_potential_switch": False,
            "closed_vmf": False,
            "closed_vmf_critic_version": 2,
            "powderworld_random_preparation": False,
            "builderbench_density_choice": False,
            "builderbench_centered_start": False,
            "builderbench_density_min": 124.0,
            "builderbench_density_max": 12400.0,
            "builderbench_density_target_entropy": 0.0,
            "potential_lipschitz": False,
            "metra_distance_budget": 1.0,
            "dense_potential_target": False,
            "nonnegative_kl_bootstrap": False,
            "information_trace_lambda": 0.0,
            "information_trace_steps": 8,
            "freeze_skill_learning": False,
            "information_reward": "posterior_kl",
            "information_critic_updates_per_step": 1,
            "skill_exploration_weight": 1.0,
        }
        for name, default in retired_defaults.items():
            if name in values and values.pop(name) != default:
                raise ValueError(f"retired feature {name} is active in this checkpoint")
        values["hidden_dims"] = tuple(values["hidden_dims"])
        if "potential_replay_mode" not in values:
            values["potential_replay_mode"] = GEOMETRIC_ANCHOR_REPLAY_MODE
            if values.get("recipe_id") in {
                BUILDERBENCH_INCUMBENT_SAC,
                ANTSOCCER_INCUMBENT_SAC,
                ANT_STANDARD_42_STATE_SAC,
            }:
                # Configs without this field used the geometric replay path; loading them must
                # keep that path instead of silently resuming different math.
                values["allow_reference_deviation"] = True
        return cls(**values)


def _updates_for_sample_ratio(
    *, num_envs: int, rollout_steps: int, batch_size: int
) -> int:
    sampled_rows = (
        PRODUCTION_POLICY_UPDATE_SAMPLE_RATIO * rollout_steps * num_envs
    )
    if sampled_rows % batch_size:
        raise ValueError(
            "production rollout_steps * num_envs * replay ratio must be divisible "
            "by batch_size"
        )
    return sampled_rows // batch_size


def _incumbent_builderbench_values(num_envs: int) -> dict[str, Any]:
    rollout_steps = 8
    updates_per_chunk = _updates_for_sample_ratio(
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        batch_size=int(BUILDERBENCH_INCUMBENT_SCIENTIFIC_FIELDS["batch_size"]),
    )
    return {
        "recipe_id": BUILDERBENCH_INCUMBENT_SAC,
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "updates_per_chunk": updates_per_chunk,
        **BUILDERBENCH_INCUMBENT_SCIENTIFIC_FIELDS,
    }


def _incumbent_antsoccer_values(num_envs: int) -> dict[str, Any]:
    rollout_steps = 8
    updates_per_chunk = _updates_for_sample_ratio(
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        batch_size=int(ANTSOCCER_INCUMBENT_SCIENTIFIC_FIELDS["batch_size"]),
    )
    return {
        "recipe_id": ANTSOCCER_INCUMBENT_SAC,
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "updates_per_chunk": updates_per_chunk,
        **ANTSOCCER_INCUMBENT_SCIENTIFIC_FIELDS,
    }


def _standard_42_state_ant_values(num_envs: int) -> dict[str, Any]:
    rollout_steps = 8
    updates_per_chunk = _updates_for_sample_ratio(
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        batch_size=int(ANT_STANDARD_42_STATE_SCIENTIFIC_FIELDS["batch_size"]),
    )
    return {
        "recipe_id": ANT_STANDARD_42_STATE_SAC,
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "updates_per_chunk": updates_per_chunk,
        **ANT_STANDARD_42_STATE_SCIENTIFIC_FIELDS,
    }


def default_recipe(environment: str, algorithm: SkillMethod = "potential", **overrides: Any) -> ExperimentConfig:
    """Return a resolved recipe; BuilderBench Potential defaults to reference parity."""

    if environment == "builderbench_corner3" and algorithm == "potential":
        requested_num_envs = int(overrides.get("num_envs", PRODUCTION_NUM_ENVS))
        values = {"environment": environment, "algorithm": algorithm}
        values.update(_incumbent_builderbench_values(requested_num_envs))
    elif environment == "ant_soccer" and algorithm == "potential":
        requested_num_envs = int(overrides.get("num_envs", PRODUCTION_NUM_ENVS))
        values = {"environment": environment, "algorithm": algorithm}
        values.update(_incumbent_antsoccer_values(requested_num_envs))
    elif environment == "ant" and algorithm == "potential":
        requested_num_envs = int(overrides.get("num_envs", PRODUCTION_NUM_ENVS))
        values = {"environment": environment, "algorithm": algorithm}
        values.update(_standard_42_state_ant_values(requested_num_envs))
    else:
        num_envs = 256
        values = {
            "environment": environment,
            "algorithm": algorithm,
            "num_envs": num_envs,
            "discrete_skills": False,
            "discrete_skill_library": "random",
        }
    values.update(overrides)
    if values.get("recipe_id") in {
        BUILDERBENCH_INCUMBENT_SAC,
        ANTSOCCER_INCUMBENT_SAC,
        ANT_STANDARD_42_STATE_SAC,
    } and "updates_per_chunk" not in overrides:
        values["updates_per_chunk"] = _updates_for_sample_ratio(
            num_envs=int(values["num_envs"]),
            rollout_steps=int(values["rollout_steps"]),
            batch_size=int(values["batch_size"]),
        )
    return ExperimentConfig(**values)


@dataclass(frozen=True, slots=True)
class OffPolicyConfig:
    rollout_steps: int = 16
    updates_per_chunk: int = 16
    batch_size: int = 1024
    replay_capacity: int = 1_000_000
    episode_length: int = 200
    representation_episodes: int = 64
    future_gamma: float = 0.99
    max_future_steps: int = 200
    sample_hindsight_anchors: bool = False
    potential_replay_mode: str = GEOMETRIC_ANCHOR_REPLAY_MODE
    # Bernoulli share of relabel-eligible rows that actually relabel; 1.0 relabels all
    # of them and draws no extra RNG.
    achieved_skill_relabel_fraction: float = 1.0
    achieved_skill_relabel_outcome: str = "current"
    achieved_skill_relabel_samples: int = 8
    encoder_skill_source: str = "commanded"   # or "achieved_discounted": HER label for the encoder rows
    # "No policy switching" ablation: this fraction of lanes runs the potential policy (z=0)
    # for whole episodes from reset, the rest run one skill from reset.
    # Requires potential_steps == 0 and two_phase mode. 0.0 = the two-phase method.
    potential_lane_fraction: float = 0.0
    potential_steps: int = 0
    random_potential_switch_max_steps: int = 0
    potential_switch_seed: int = 0
    stagger_reset_groups: int = 0
    skill_start_mode: str = TWO_PHASE_SKILL_START_MODE
    encoder_updates_per_chunk: int = 0
    branch_transitions_to_policy_replay: bool = False
    # Lanes sharing one anchor, each with a DISTINCT skill. 1 makes
    # anchor -> label a function, which the encoder memorises instead of
    # reading the continuation.
    branch_targets_per_anchor: int = 1
    # 0 = derive from `episode_length // rollout_steps`.
    # Set to 1 for directed information: every lane completes its branch each
    # segment, so all `num_envs` anchors are fresh instead of `num_envs/50`.
    branch_phase_groups_override: int = 0
    # Run the LIVE lanes on the potential policy (z=0) for their whole
    # episode, so branch anchors are drawn from the potential policy's own
    # marginal and no lane ever switches from one commanded skill to
    # another. Skill-on rows are then EXACTLY the branch rows, whose anchor
    # geometry is the one the encoder trains on, so the empowerment reward
    # is never evaluated off-distribution.
    live_potential_only: bool = False
    branch_live_two_phase: bool = False
    reward_partition_anneal_start_steps: int = 0
    reward_partition_anneal_end_steps: int = 0

    def __post_init__(self) -> None:
        start, end = self.reward_partition_anneal_start_steps, self.reward_partition_anneal_end_steps
        if (end == 0 and start != 0) or (end != 0 and not 0 <= start < end):
            raise ValueError("partition annealing requires 0 <= start < end, or both zero")
        if min(
            self.rollout_steps,
            self.updates_per_chunk,
            self.batch_size,
            self.replay_capacity,
            self.episode_length,
            self.representation_episodes,
        ) <= 0:
            raise ValueError("rollout/update/batch/replay/episode/archive sizes must be positive")
        if self.potential_steps >= self.episode_length:
            raise ValueError("potential_steps must leave at least one skill-on step per episode")
        if not 0 <= self.random_potential_switch_max_steps < self.episode_length:
            raise ValueError("random_potential_switch_max_steps must be in [0, episode_length)")
        if self.random_potential_switch_max_steps and (
            self.skill_start_mode != TWO_PHASE_SKILL_START_MODE
            or self.potential_lane_fraction != 0.0
            or self.live_potential_only
        ):
            raise ValueError("random potential switching requires two_phase without separate potential lanes")
        if not 0.0 <= self.potential_lane_fraction < 1.0:
            raise ValueError("potential_lane_fraction must be in [0, 1)")
        if self.potential_lane_fraction > 0.0 and (self.potential_steps != 0 or self.skill_start_mode != TWO_PHASE_SKILL_START_MODE):
            raise ValueError("potential_lane_fraction > 0 (separate lanes) requires potential_steps == 0 and two_phase mode")
        if min(
            self.potential_steps,
            self.stagger_reset_groups,
        ) < 0:
            raise ValueError("relabel windows and stagger groups must be nonnegative")
        if not 0.0 < self.achieved_skill_relabel_fraction <= 1.0:
            raise ValueError("achieved_skill_relabel_fraction must be in (0,1]")
        if not 0.0 <= self.future_gamma < 1.0:
            raise ValueError("future_gamma must be in [0, 1)")
        if self.max_future_steps <= 0:
            raise ValueError("max_future_steps must be positive")
        if self.encoder_updates_per_chunk < 0:
            raise ValueError("encoder_updates_per_chunk must be nonnegative")
        if self.encoder_updates_per_chunk:
            if self.encoder_updates_per_chunk > self.updates_per_chunk:
                raise ValueError(
                    "encoder_updates_per_chunk cannot exceed updates_per_chunk"
                )
            if self.updates_per_chunk % self.encoder_updates_per_chunk:
                raise ValueError(
                    "encoder_updates_per_chunk must divide updates_per_chunk so the "
                    "encoder stride is uniform across the segment"
                )
        if self.potential_replay_mode not in POTENTIAL_REPLAY_MODES:
            raise ValueError(
                "potential_replay_mode must be 'legacy_geometric'"
            )
        if self.skill_start_mode not in SKILL_START_MODES:
            raise ValueError(
                "skill_start_mode must be 'two_phase' or 'marginal_branch'"
            )
        if self.skill_start_mode == MARGINAL_BRANCH_SKILL_START_MODE:
            if self.potential_steps <= 0:
                raise ValueError(
                    "marginal_branch requires positive potential_steps to define "
                    "the branch horizon"
                )
            if self.episode_length % self.rollout_steps:
                raise ValueError(
                    "marginal_branch requires episode_length divisible by rollout_steps"
                )
            if self.stagger_reset_groups <= 0:
                raise ValueError("marginal_branch requires staggered resets")
            if self.branch_targets_per_anchor < 1:
                raise ValueError("branch_targets_per_anchor must be at least 1")
        if self.branch_live_two_phase and (
            not self.uses_marginal_branches or self.live_potential_only
        ):
            raise ValueError("branch_live_two_phase requires ordinary encoder-only marginal branches")
        if self.live_potential_only and not self.uses_marginal_branches:
            raise ValueError(
                "live_potential_only requires marginal_branch: without branches "
                "no skill-on rows would exist at all"
            )
        if (
            self.branch_transitions_to_policy_replay
            and self.skill_start_mode != MARGINAL_BRANCH_SKILL_START_MODE
        ):
            raise ValueError(
                "branch_transitions_to_policy_replay requires marginal_branch: "
                "no other mode collects branch rollouts"
            )
        if self.uses_marginal_branches:
            if self.branch_phase_groups < 1:
                raise ValueError("branch_phase_groups must be at least 1")
            if self.branch_horizon % self.branch_phase_groups:
                raise ValueError(
                    "marginal_branch requires branch_horizon divisible by its "
                    "episode phase groups"
                )

    @property
    def resolved_encoder_updates_per_chunk(self) -> int:
        """Encoder gradient steps per rollout chunk; 0 means follow SAC's ratio."""

        return self.encoder_updates_per_chunk or self.updates_per_chunk

    def reward_partition_weight(self, env_steps: jax.Array) -> jax.Array:
        """Current replay-reward coefficient; global collected transitions are the clock."""
        if self.reward_partition_anneal_end_steps == 0:
            return jnp.asarray(1.0, dtype=jnp.float32)
        start = self.reward_partition_anneal_start_steps
        duration = self.reward_partition_anneal_end_steps - start
        return jnp.clip((env_steps - start) / duration, 0.0, 1.0)

    @property
    def encoder_update_stride(self) -> int:
        return self.updates_per_chunk // self.resolved_encoder_updates_per_chunk

    @property
    def uses_marginal_branches(self) -> bool:
        return self.skill_start_mode == MARGINAL_BRANCH_SKILL_START_MODE

    @property
    def collector_potential_steps(self) -> int:
        if self.live_potential_only:
            # Past any reachable step_count, so a live lane never activates a
            # skill. `branch_horizon` stays tied to `potential_steps`, which is
            # why this cannot simply reuse that value.
            return self.episode_length + 1
        return (0 if self.uses_marginal_branches
                and not self.branch_live_two_phase else self.potential_steps)

    @property
    def branch_horizon(self) -> int:
        return self.episode_length - self.potential_steps

    @property
    def branch_phase_groups(self) -> int:
        if self.branch_phase_groups_override:
            return self.branch_phase_groups_override
        return self.episode_length // self.rollout_steps

    @property
    def branch_steps_per_segment(self) -> int:
        return self.branch_horizon // self.branch_phase_groups

    def max_simultaneous_completions(self, num_envs: int) -> int:
        """Lanes that can finish on the same step, excluding early terminations."""
        if self.stagger_reset_groups <= 0:
            return num_envs
        groups = min(self.stagger_reset_groups, num_envs)
        return math.ceil(num_envs / groups)
