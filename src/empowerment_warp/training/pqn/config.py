"""Frozen run configuration for potential-PQN Craftax training."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import jax.numpy as jnp

from empowerment_warp.agents.see_pqn import (
    PotentialPQNAgent,
    get_config as potential_pqn_config,
)
from empowerment_warp.agents.skill_classifier import SkillConfig
from empowerment_warp.envs.craftax_view import _craftax_encoder_view


# Fields with a single supported value; every Craftax recipe carries it (the dataclass
# defaults below agree).
_REMOVED_BRANCH_PINS = {
    "separate_skill0_q": True,
    "sample_hindsight_anchors": False,
    "relabel_encoder": "live",
    "relabel_outcome": "discounted",
    "relabel_samples": 8,
    "achieved_skill_relabel_min_norm": 0.0,
    "discrete_skill_library": "axis",
    "discrete_reward_softmax": True,
    "achieved_skill_relabel_min_prob": 0.0,
    "achieved_skill_relabel_weight_mode": "effective_number",
    "achieved_skill_relabel_effective_beta": 0.99,
    "achieved_skill_relabel_weight_max": 3.0,
}


@dataclass(frozen=True, slots=True)
class PotentialPQNRunConfig:
    """Frozen recipe for one potential-PQN Craftax run (reference: potential_pqn.sh)."""

    @classmethod
    def from_dict(cls, payload: dict) -> "PotentialPQNRunConfig":
        """Read saved PQN configs, dropping inert PPO fields that older configs carry."""
        values = dict(payload)
        retired_defaults = {
            "ppo_adam_eps": 1e-5,
            "ppo_clip_coef": 0.2,
            "ppo_clip_vloss": True,
            "ppo_ent_coef": 0.01,
            "ppo_gae_lambda": 0.8,
            "ppo_max_grad_norm": 1.0,
            "ppo_norm_adv": True,
            "ppo_vf_coef": 0.5,
        }
        for name, default in retired_defaults.items():
            if name in values and values.pop(name) != default:
                raise ValueError(f"retired PPO field {name} is active in this checkpoint")
        return cls(**values)

    environment: str
    run_dir: str
    rl_algorithm: str = "potential-pqn"
    seed: int = 0
    num_envs: int = 1024
    episode_length: int = 1000
    optimistic_reset_ratio: int = 16
    total_env_steps: int = 500_000_000
    # Agent SCHEDULE denominator for the epsilon anneal + LR linear decay (agent
    # `pqn_total_env_steps`), DECOUPLED from the loop budget. As in the reference PQN
    # recipe, a 500M run follows a 1B schedule truncated at 500M (epsilon anneals over
    # 5% of 1B = 50M; LR still at 50% when training ends). Tying this to total_env_steps
    # anneals 2x faster and drives the LR to ~0 by the end.
    pqn_schedule_env_steps: int = 1_000_000_000
    rollout_steps: int = 128
    discount: float = 0.95
    lr: float = 3e-4
    pqn_hidden_size: int = 512
    pqn_num_layers: int = 1  # reference get_config default; 1B/500M baselines both 1
    pqn_num_epochs: int = 4
    pqn_num_minibatches: int = 4
    pqn_lambda: float = 0.5
    pqn_epsilon_start: float = 1.0
    pqn_epsilon_end: float = 0.005
    pqn_epsilon_fraction: float = 0.05
    pqn_norm_input: bool = True
    pqn_lr_linear_decay: bool = True
    pqn_max_grad_norm: float = 0.5
    # Context enters the Q recurrence pre-LSTM (False, reference default; the 1B
    # eps50m baseline predates the flag) vs post-LSTM context-free carry (True, the
    # 500M 'contextfree' variant). Potential path only.
    pqn_context_free_recurrence: bool = False
    # Recipes set this (see.yaml: true). The default stays False: from_dict reads saved runs whose
    # config.json predates the key and which reset memory, and True would conflict with the
    # achieved_skill_relabeling=True default in __post_init__.
    pqn_preserve_memory_at_switch: bool = False
    # Empowerment / skill representation (maps to warp SkillConfig).
    skill_dim: int = 16
    num_skills: int = 16
    discrete_skill_library: str = "axis"
    potential_steps: int = 100
    potential_lane_fraction: float = 0.0
    skill_logit_init_std: float = 0.0
    separate_skill0_q: bool = True  # Reproduction first: reference two-Q controller.
    potential_reward_reference: str = "anchor"  # use_phi_Q=fixed_anchor
    reward_encoder: str = "live"
    relabel_encoder: str = "live"  # the trained copy, as on every engine; "slow" = EMA copy
    # Which outcome the achieved-skill label reads: "segment_end" (the segment's last observed
    # row) or "discounted" (M futures per row from
    # Geometric(1 - discount) within the segment, aggregated in phi-space before the classifier;
    # the same rule as SAC's achieved_skill_relabel_outcome).
    relabel_outcome: str = "discounted"
    relabel_samples: int = 8
    # Full-state checkpoint cadence in chunks (0 = only at the end). The full state carries the
    # encoder archive (~3 GB at 10M rows), so it is written less often than params_*.pkl.
    full_state_every_segments: int = 0
    full_state_include_archive: bool = True
    # Multiplies the skill (intrinsic) reward. The discrete softmax reward carries the encoder
    # temperature inside its logits (<dphi, z>/tau, tau = 0.1); the continuous raw reward
    # <dphi, z> does not, so scale = 1/tau = 10 puts sym-CSF on the discrete path's reward scale.
    potential_reward_scale: float = 1.0
    continuous_reward_temperature: bool = True
    encoder_temperature: float = 0.1  # tau_nce
    encoder_objective: str = "auto"
    continuous_reward_normalization: str = "none"
    reward_partition_anneal_start_steps: int = 0
    reward_partition_anneal_end_steps: int = 0
    phi_norm_cap: float = 5.0
    encoder_hidden_dims: tuple[int, ...] = (512, 512)
    encoder_learning_rate: float = 1e-3  # crl_lr
    encoder_max_grad_norm: float = 0.0  # Reference encoder Adam has no gradient clipping.
    encoder_target_tau: float = 1.0
    encoder_batch_size: int = 4096  # reference BATCH_SIZE; encoder batch sampled from the archive
    discrete_reward_softmax: bool = True  # Reference log(K * p(skill | displacement)).
    phi_noise_std: float = 0.0  # A8 axis; craftax reference runs no encoder-input noise
    # A1 axis: False = continuous unit-sphere skills
    # z ~ unit sphere. Then the library / num_skills are inert for training, the reward is the
    # raw <phi(s')-phi(ref), z>, achieved-skill relabel = unit displacement direction (weights
    # == 1), and the z=0 value needs potential_skill_samples > 0 (leaf: 8).
    discrete_skills: bool = True
    # Intrinsic encoder/reward obs view (reference ENCODER_OBS_MODE). "inventory" =
    # the 12 craftax-classic inventory tail columns (the reference default for this
    # family); "inventory_status" = the full 22-value tail; "full"/"auto" = whole obs.
    encoder_obs_mode: str = "inventory"
    representation_archive_size: int = 10_000_000
    keep_checkpoints: int = 0  # 0 preserves all params checkpoints; 1 keeps the latest.
    # "branches": encoder-only, fixed-anchor sweeps from copied live physical states.
    # Live PQN collection, reward and optimizer schedule are unchanged.
    encoder_data_source: str = "live"
    encoder_branch_steps: int = 0  # 0 uses episode_length - potential_steps.
    # Encoder steps per Q optimizer step, not per collected rollout.
    representation_archive_update_ratio: float = 1.0
    future_gamma: float = 0.95  # reference FUTURE_GAMMA
    max_future_steps: int = 0  # Reference: episode/ring bounds only, no extra cap.
    # Total retained transitions; 0 uses encoder_batch_size, as in the reference.
    representation_train_start_rows: int = 10000
    potential_skill_samples: int = 0
    # Achieved-skill relabeling (reference base: ON, segment unit, effective_number).
    # Relabels the commanded skill to the achieved segment direction on the POLICY/reward
    # batch only (never the encoder) -- decision 11. Weights are computed but inert for
    # PQN's unweighted Q(lambda) MSE (they only matter to the SAC critic path).
    achieved_skill_relabeling: bool = True
    sample_hindsight_anchors: bool = False
    relabel_unit: str = "row"
    achieved_skill_relabel_min_prob: float = 0.0
    achieved_skill_relabel_min_norm: float = 0.0   # continuous relabel gate on ||G||; 0 = no gate
    pair_form: str = "difference"   # ablation: "pair_mlp" = <f(s,s'), z> decoder instead of the difference form
    achieved_skill_relabel_weight_mode: str = "effective_number"
    achieved_skill_relabel_effective_beta: float = 0.99
    achieved_skill_relabel_weight_max: float = 3.0
    # Host-loop cadence.
    log_every_segments: int = 10
    checkpoint_every_segments: int = 1_000
    eval_every_segments: int = 0
    eval_episode_length: int = 0
    max_train_seconds: float | None = None
    max_chunks: int = 0  # Smoke stop only; preserves the full epsilon/LR budget.
    eval_at_start: bool = True
    eval_contexts: int = 128
    eval_skills_per_context: int = 1
    eval_media_interval: int = 50_000_000  # Collected transitions, delta-gated at eval events.
    eval_video_episodes: int = 1  # One shared-prefix media context; 0 disables media.
    eval_media_skill_count: int = 4
    eval_video_frame_skip: int = 4
    eval_fixed_reset_seed: int = 0  # Scalar media only.
    eval_render_block_pixel_size: str = "agent"
    allow_cpu: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def agent_class(self):
        return PotentialPQNAgent

    def agent_config(self) -> dict:
        cfg = dict(potential_pqn_config())
        cfg.update(
            lr=self.lr,
            discount=self.discount,
            pqn_hidden_size=self.pqn_hidden_size,
            pqn_num_layers=self.pqn_num_layers,
            pqn_norm_input=self.pqn_norm_input,
            pqn_lr_linear_decay=self.pqn_lr_linear_decay,
            pqn_rollout_steps=self.rollout_steps,
            pqn_expected_num_envs=self.num_envs,
            pqn_num_epochs=self.pqn_num_epochs,
            pqn_num_minibatches=self.pqn_num_minibatches,
            pqn_lambda=self.pqn_lambda,
            pqn_max_grad_norm=self.pqn_max_grad_norm,
            pqn_epsilon_start=self.pqn_epsilon_start,
            pqn_epsilon_end=self.pqn_epsilon_end,
            pqn_epsilon_fraction=self.pqn_epsilon_fraction,
            pqn_total_env_steps=self.pqn_schedule_env_steps,
            pqn_context_free_recurrence=self.pqn_context_free_recurrence,
            pqn_preserve_memory_at_switch=self.pqn_preserve_memory_at_switch,
            # Mirror PotentialPQNAgent.create(): the two-Q potential controller carries two LSTMs (4h+1).
            pqn_recurrent_state_size=4 * self.pqn_hidden_size + 1,
            skill_dim=self.skill_dim,
            num_skills=self.num_skills,
            potential_skill_samples=self.potential_skill_samples,
            potential_steps=self.potential_steps,
            separate_skill0_q=self.separate_skill0_q,
            skill_logit_init_std=self.skill_logit_init_std,
        )
        return cfg

    def skill_config(self, observation_dim: int) -> SkillConfig:
        # Encoder obs VIEW (reference parity, empowerment/envs/craftax/encoder_views.py):
        # the intrinsic encoder/reward see only a SLICE of the symbolic obs (the actor/
        # critic keep the full obs). Default "inventory" (12 inventory tail columns) —
        # the ego-centric map scrolls into noise over a geometric future, so the full
        # obs collapses the encoder; the inventory view changes coherently with the
        # skill. See [[project_pqn_encoder_archive_churn]]. None => full obs.
        view_indices, view_dim = _craftax_encoder_view(observation_dim, self.encoder_obs_mode)
        return SkillConfig(
            method="potential",
            observation_dim=view_dim,
            observation_indices=view_indices,
            skill_dim=self.skill_dim,
            discrete_skills=self.discrete_skills,
            # SkillConfig demands num_skills=0 for continuous draws; the runner keeps
            # num_skills (16) only as the eval probe-set size (see eval_skill_library).
            num_skills=0,
            discrete_skill_library=self.discrete_skill_library,
            skill_library_seed=self.seed,
            hidden_dims=tuple(self.encoder_hidden_dims),
            learning_rate=self.encoder_learning_rate,
            encoder_max_grad_norm=self.encoder_max_grad_norm,
            target_tau=self.encoder_target_tau,
            temperature=self.encoder_temperature,
            encoder_objective=self.encoder_objective,
            continuous_reward_normalization=self.continuous_reward_normalization,
            phi_noise_std=self.phi_noise_std,
            phi_norm_cap=self.phi_norm_cap,
            reward_encoder=self.reward_encoder,
            relabel_encoder=self.relabel_encoder,
            potential_reward_scale=self.potential_reward_scale,
            continuous_reward_temperature=self.continuous_reward_temperature,
            discrete_reward_softmax=self.discrete_reward_softmax,
            potential_reward_reference=self.potential_reward_reference,
            achieved_skill_relabeling=self.achieved_skill_relabeling,
            achieved_skill_relabel_min_prob=self.achieved_skill_relabel_min_prob,
            achieved_skill_relabel_min_norm=self.achieved_skill_relabel_min_norm,
            pair_form=self.pair_form,
            achieved_skill_relabel_weight_mode=self.achieved_skill_relabel_weight_mode,
            achieved_skill_relabel_effective_beta=self.achieved_skill_relabel_effective_beta,
            achieved_skill_relabel_weight_max=self.achieved_skill_relabel_weight_max,
        )

    def __post_init__(self):
        if self.pqn_preserve_memory_at_switch and (self.achieved_skill_relabeling or self.sample_hindsight_anchors):
            raise ValueError("preserving switch memory requires commanded skills and anchors without hindsight relabeling")
        # Removed branches: plain/shared-Q and context-free controllers, Craftax hindsight
        # relabeling, and the discrete-skill library, reward and relabel. Their fields stay
        # (YAMLs and frozen run configs carry them) but must hold the value every run used.
        # discrete_skills is enforced by SkillConfig; achieved_skill_relabeling and relabel_unit,
        # whose dataclass defaults differ from the recipes, are enforced by the runner.
        for name, value in _REMOVED_BRANCH_PINS.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} selects an unsupported branch; it must be {value!r}")
        if self.encoder_objective == "auto":
            # A discrete config (the dataclass default) still resolves as before; SkillConfig
            # then rejects it when the learner is built.
            object.__setattr__(self, "encoder_objective",
                               "symmetric_infonce" if self.discrete_skills else "forward_infonce")
        if not self.discrete_skills and self.encoder_objective == "symmetric_infonce":
            raise ValueError("symmetric continuous-skill CSF is not supported; select forward_infonce or vmf_forward")
        start, end = self.reward_partition_anneal_start_steps, self.reward_partition_anneal_end_steps
        if start < 0 or end < 0 or (end == 0 and start != 0) or (end > 0 and end <= start):
            raise ValueError("partition anneal requires 0 <= start < end, or both zero to disable")
        if end and self.continuous_reward_normalization not in {"sphere", "sampled_infonce"}:
            raise ValueError("partition anneal requires a normalized reward")
        if self.rl_algorithm != "potential-pqn":
            raise ValueError("rl_algorithm must be 'potential-pqn'")
        if self.rollout_steps < 2:
            raise ValueError("PQN training rollouts require at least two steps.")
        if min(self.num_envs, self.rollout_steps, self.episode_length, self.optimistic_reset_ratio,
               self.pqn_num_epochs, self.pqn_num_minibatches, self.encoder_batch_size,
               self.representation_archive_size) <= 0:
            raise ValueError("PQN rollout, optimizer and archive sizes must be positive")
        if self.num_envs % self.pqn_num_minibatches:
            raise ValueError("pqn_num_minibatches must divide num_envs exactly")
        if not math.isfinite(self.representation_archive_update_ratio) or self.representation_archive_update_ratio < 0:
            raise ValueError("representation_archive_update_ratio must be finite and nonnegative")
        if self.representation_train_start_rows < 0:
            raise ValueError("representation_train_start_rows must be nonnegative")
        if self.encoder_train_start_rows > self.representation_archive_size:
            raise ValueError("encoder readiness threshold exceeds archive row capacity")
        if self.encoder_data_source not in {"live", "branches"}:
            raise ValueError("encoder_data_source must be 'live' or 'branches'")
        if self.encoder_branch_steps < 0:
            raise ValueError("encoder_branch_steps must be nonnegative")
        if self.encoder_data_source == "branches" and not 2 <= self.branch_horizon_steps <= self.episode_length:
            raise ValueError("encoder branch length must be between 2 and episode_length")
        if not 0 < self.future_gamma < 1 or self.max_future_steps < 0:
            raise ValueError("future_gamma must be in (0,1); max_future_steps >= 0")
        if self.skill_logit_init_std < 0.0:
            raise ValueError("skill_logit_init_std must be >= 0")
        if self.potential_lane_fraction < 0.0 or self.potential_lane_fraction >= 1.0:
            raise ValueError("potential_lane_fraction must be in [0, 1)")
        if self.potential_lane_fraction > 0.0 and self.potential_steps != 0:
            raise ValueError(
                "separate-lanes mode (potential_lane_fraction > 0) requires potential_steps == 0: "
                "skill lanes start at reset, potential lanes never hand off"
            )
        if not self.discrete_skills and self.potential_skill_samples <= 0:
            raise ValueError(
                "continuous skills need potential_skill_samples > 0 for the z=0 value "
                "(reference leaf: 8)"
            )
        if self.pqn_schedule_env_steps <= 0:
            raise ValueError("pqn_schedule_env_steps must be positive")
        if self.max_chunks < 0:
            raise ValueError("max_chunks must be nonnegative")
        if self.keep_checkpoints < 0:
            raise ValueError("keep_checkpoints must be nonnegative")
        if min(self.eval_contexts, self.eval_skills_per_context,
               self.eval_media_skill_count, self.eval_video_frame_skip) <= 0:
            raise ValueError("Evaluation context/skill counts and frame skip must be positive")
        if self.eval_video_episodes not in (0, 1):
            raise ValueError("Reference branch media uses one context; eval_video_episodes must be 0 or 1")
        if self.eval_media_interval < 0 or self.eval_render_block_pixel_size not in ("agent", "img", "human"):
            raise ValueError("Invalid evaluation media interval or render block size")

    @property
    def branch_horizon_steps(self) -> int:
        return self.encoder_branch_steps or self.episode_length - self.potential_steps

    @property
    def encoder_train_start_rows(self) -> int:
        return self.representation_train_start_rows or self.encoder_batch_size

    def reward_partition_weight(self, env_steps):
        """Collected-transition clock; evaluated at the end of each rollout."""
        if self.reward_partition_anneal_end_steps == 0:
            return jnp.asarray(1.0, jnp.float32)
        start = self.reward_partition_anneal_start_steps
        return jnp.clip((jnp.asarray(env_steps, jnp.float32) - start)
                        / (self.reward_partition_anneal_end_steps - start), 0.0, 1.0)

    @property
    def encoder_updates_per_chunk(self) -> int:
        return math.ceil(self.representation_archive_update_ratio
                         * self.pqn_num_epochs * self.pqn_num_minibatches)

    def effective_recipe(self) -> dict:
        """Derived from the same config consumed by the live training loop."""
        collected = self.num_envs * self.rollout_steps
        return {
            "collected_transitions_per_chunk": collected,
            "q_optimizer_steps_per_chunk": self.pqn_num_epochs * self.pqn_num_minibatches,
            "encoder_updates_per_chunk_after_ready": self.encoder_updates_per_chunk,
            "encoder_sampled_rows_per_chunk_after_ready": self.encoder_updates_per_chunk * self.encoder_batch_size,
            "encoder_samples_per_collected_transition": self.encoder_updates_per_chunk * self.encoder_batch_size / collected,
            "encoder_objective": self.encoder_objective,
            "continuous_reward_normalization": self.continuous_reward_normalization,
            "reward_partition_anneal_start_steps": self.reward_partition_anneal_start_steps,
            "reward_partition_anneal_end_steps": self.reward_partition_anneal_end_steps,
            "archive_capacity_transitions": self.representation_archive_size,
            "encoder_train_start_total_rows": self.encoder_train_start_rows,
            "first_encoder_eligible_env_step": (
                math.ceil(math.ceil(self.encoder_train_start_rows / (self.num_envs * self.branch_horizon_steps))
                          * self.episode_length / self.rollout_steps) * collected
                if self.encoder_data_source == "branches" else
                math.ceil(self.encoder_train_start_rows / collected) * collected),
            "encoder_readiness_condition": ("valid branch rows reach threshold; first step is a lower bound"
                                            if self.encoder_data_source == "branches" else "stored live rows reach threshold"),
            "archive_admission": ("completed branch sweeps; post-terminal padding ineligible"
                                  if self.encoder_data_source == "branches" else
                                  "all transitions, immediately, time-major"),
            "archive_sampling": ("fixed copied-state anchors and commanded branch outcomes; "
                                 "valid same-branch geometric futures"
                                 if self.encoder_data_source == "branches" else
                                 "commanded skill-on rows; same-episode geometric futures"),
            "encoder_data_source": self.encoder_data_source,
            "branch_horizon_steps": (self.branch_horizon_steps
                                     if self.encoder_data_source == "branches" else 0),
            "branch_sweep_interval_live_vector_steps": (self.episode_length
                                                         if self.encoder_data_source == "branches" else 0),
            "branch_physical_steps_per_live_transition": (
                self.branch_horizon_steps / self.episode_length
                if self.encoder_data_source == "branches" else 0.0),
            "controller": self.rl_algorithm,
            "policy_optimizer_steps_per_chunk": self.pqn_num_epochs * self.pqn_num_minibatches,
            "exploration": "epsilon-greedy",
            "potential_skill_reduction": "mean",
            "potential_action_reduction": "max (greedy PQN action)",
            "potential_value_carry": ("observed pre-step history" if self.pqn_preserve_memory_at_switch
                                      else "fresh (zero) carry"),
            "potential_skill_count": self.potential_skill_samples or self.num_skills,
            "potential_skill_draws": "fresh Monte Carlo unit-sphere draws per row per call",
            "skill_start_mode": "separate_lanes" if self.potential_lane_fraction > 0 else "two_phase",
            "potential_lane_fraction": self.potential_lane_fraction,
            "potential_prefix_steps": self.potential_steps,
            "skill_logit_init_std": self.skill_logit_init_std,
            "relabel_encoder": self.relabel_encoder,
            "relabel_outcome": self.relabel_outcome,
            "relabel_samples": self.relabel_samples,
            "sample_hindsight_anchors": self.sample_hindsight_anchors,
            "relabel_unit": self.relabel_unit,
            "potential_reward_scale": self.potential_reward_scale,
            "continuous_reward_temperature": self.continuous_reward_temperature,
            "continuous_reward_effective_scale": (
                self.potential_reward_scale / self.encoder_temperature
                if self.continuous_reward_temperature else self.potential_reward_scale
            ),
            "evaluation_protocol": "reference shared-prefix skill branches",
            "evaluation_deterministic": False,
            "evaluation_seed": self.seed,
            "evaluation_interval_transitions": self.eval_every_segments * collected,
            "evaluation_media_interval_transitions": self.eval_media_interval,
            "schedule_denominator_env_steps": self.pqn_schedule_env_steps,
            "epsilon_anneal_env_steps": int(self.pqn_epsilon_fraction * self.pqn_schedule_env_steps),
            "lr_fraction_remaining_at_end_of_run": (
                max(0.0, 1.0 - self.total_env_steps / self.pqn_schedule_env_steps)
                if self.pqn_lr_linear_decay else 1.0
            ),
        }
