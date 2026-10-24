"""SEE with continuous SAC.

Start at ``SEEAgent.update`` for the learning sequence and ``create_learners``
for construction. ``ContinuousSAC`` names its update pieces ``potential_reward``,
``critic_loss``, ``actor_loss`` and ``alpha_loss``. Networks live in
``utils/networks.py``; the skill classifier lives in ``agents/skill_classifier.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple
import math

from flax import struct
import jax
import jax.numpy as jnp
import optax

from empowerment_warp.agents.skill_classifier import (
    RelabelMetrics,
    SkillConfig,
    SkillLearner,
    SkillMetrics,
    SkillState,
)
from empowerment_warp.core.replay import (
    GEOMETRIC_ANCHOR_REPLAY_MODE,
    POTENTIAL_REPLAY_MODES,
    Transition,
)
from empowerment_warp.core.rollout import potential_switch_steps
from empowerment_warp.utils.networks import (
    GaussianActor,
    PotentialActor,
    TwinContinuousCritic,
    TwinPotentialCritic,
)
from empowerment_warp.utils.sampling import (
    AnchorReplayMetrics,
    ArchiveMetrics,
    _zero_anchor_replay_metrics,
    policy_hindsight_batch,
    representation_archive_batch,
)

if TYPE_CHECKING:
    from empowerment_warp.training.sac.config import ExperimentConfig, OffPolicyConfig


class SACMetrics(NamedTuple):
    critic_loss: jax.Array
    actor_loss: jax.Array
    alpha_loss: jax.Array
    alpha: jax.Array
    entropy: jax.Array
    q_mean: jax.Array
    target_q_mean: jax.Array
    # Sufficient statistics over potential-phase replay rows. Divide only after
    # aggregating updates, so empty batches do not dilute the reported means.
    replay_weight: jax.Array
    potential_weight: jax.Array
    potential_skill_value_weight: jax.Array
    potential_skill_value_total: jax.Array
    potential_reward_total: jax.Array
    potential_q_total: jax.Array
    potential_target_total: jax.Array
    potential_td_squared_total: jax.Array
    # The intrinsic reward the POLICY actually receives, restricted to skill-on
    # rows. `skill/intrinsic_reward` is scored on the ENCODER's archive batch,
    # which is a different batch and a different displacement, so it cannot
    # answer "is the policy being rewarded for using its skill".
    policy_reward_skill_on: jax.Array


@dataclass(frozen=True, slots=True)
class SACConfig:
    observation_dim: int
    action_dim: int
    skill_dim: int
    hidden_dims: tuple[int, ...] = (512, 512)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.02
    initial_alpha: float = 1.0
    target_entropy: float | None = None
    actor_layer_norm: bool = False
    critic_layer_norm: bool = True
    actor_final_scale: float = 0.0
    critic_final_scale: float = 1.0
    skill_mean_init_std: float = 0.0
    state_dependent_std: bool = False
    use_anchor_in_policy: bool = False
    use_anchor_in_critic: bool = False
    # True = z = 0 rows earn the potential reward E_z[V(s, z)] (serialized name).
    replace_skill_zero_reward_with_random_value: bool = False
    # Monte-Carlo draws of z used to estimate the skill-0 row's potential,
    # E_z[V(s,z)]. One draw is unbiased in the batch MEAN but SAC consumes
    # PER-ROW rewards, so at n=1 each transition carries a single sample of
    # an expectation over the whole library -- unbiased and very noisy.
    skill0_value_num_skills: int = 1
    separate_skill0_actor: bool = False
    separate_skill0_critic: bool = False
    potential_steps: int = 0
    potential_replay_mode: str = GEOMETRIC_ANCHOR_REPLAY_MODE
    random_potential_switch_max_steps: int = 0
    potential_switch_seed: int = 0
    episode_length: int = 0

    def __post_init__(self) -> None:
        if self.observation_dim <= 0 or self.action_dim <= 0 or self.skill_dim < 0:
            raise ValueError("observation/action dimensions must be positive and skill_dim nonnegative")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if min(self.actor_lr, self.critic_lr, self.alpha_lr) <= 0:
            raise ValueError("learning rates must be positive")
        if not 0.0 <= self.discount <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("discount must be in [0,1] and tau in (0,1]")
        if self.initial_alpha <= 0:
            raise ValueError("initial_alpha must be positive")
        if min(self.actor_final_scale, self.critic_final_scale, self.skill_mean_init_std) < 0.0:
            raise ValueError("network initialization scales must be nonnegative")
        if self.potential_steps < 0:
            raise ValueError("potential_steps must be nonnegative")
        if self.random_potential_switch_max_steps < 0:
            raise ValueError("random potential switch maximum must be nonnegative")
        if self.potential_replay_mode not in POTENTIAL_REPLAY_MODES:
            raise ValueError(
                "potential_replay_mode must be 'legacy_geometric'"
            )
        if self.skill0_value_num_skills < 1:
            raise ValueError("skill0_value_num_skills must be at least 1")
        if (self.separate_skill0_actor or self.separate_skill0_critic) and not (
            self.replace_skill_zero_reward_with_random_value
        ):
            raise ValueError("separate skill-zero networks require reward replacement")



@struct.dataclass
class SACState:
    # skill0_* = the potential actor/critic (z = 0 rows); the names are checkpointed.
    actor_params: Any
    critic_params: Any
    target_critic_params: Any
    skill0_actor_params: Any
    skill0_critic_params: Any
    target_skill0_critic_params: Any
    log_alpha: jax.Array  # Density mode: [density, gripper]; otherwise a scalar.
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    skill0_actor_opt_state: Any
    skill0_critic_opt_state: Any
    alpha_opt_state: optax.OptState
    step: jax.Array


def _polyak(target: Any, source: Any, tau: float) -> Any:
    return jax.tree_util.tree_map(lambda old, new: (1.0 - tau) * old + tau * new, target, source)


def _weighted_mean(values: jax.Array, weights: jax.Array) -> jax.Array:
    weights = jnp.asarray(weights, dtype=values.dtype).reshape((-1,))
    rows = values.reshape((values.shape[0], -1)).mean(axis=-1)
    return jnp.sum(rows * weights) / jnp.maximum(jnp.sum(weights), 1.0)


def _phase_balanced_mean(
    values: jax.Array,
    weights: jax.Array,
    potential_mask: jax.Array,
) -> jax.Array:
    rows = values.reshape((values.shape[0], -1)).mean(axis=-1)
    weights = jnp.asarray(weights, dtype=rows.dtype).reshape((-1,))
    potential_weights = weights * potential_mask.astype(rows.dtype)
    skill_on_weights = weights * (~potential_mask).astype(rows.dtype)
    potential_count = jnp.sum(potential_weights)
    skill_on_count = jnp.sum(skill_on_weights)
    potential_mean = jnp.sum(rows * potential_weights) / jnp.maximum(potential_count, 1.0)
    skill_on_mean = jnp.sum(rows * skill_on_weights) / jnp.maximum(skill_on_count, 1.0)
    fallback = _weighted_mean(rows, weights)
    return jnp.where(
        potential_count > 0,
        jnp.where(skill_on_count > 0, 0.5 * (potential_mean + skill_on_mean), potential_mean),
        jnp.where(skill_on_count > 0, skill_on_mean, fallback),
    )


class ContinuousSAC:
    """Tanh-Gaussian SAC for continuous environments."""

    def __init__(self, config: SACConfig):
        entropy_dims = config.action_dim
        target_entropy = 0.5 * entropy_dims if config.target_entropy is None else config.target_entropy
        entropy_ceiling = entropy_dims * math.log(2.0)
        if target_entropy > entropy_ceiling:
            raise ValueError(
                f"continuous target_entropy={target_entropy} exceeds the maximum "
                f"{entropy_ceiling} for {entropy_dims} actions in [-1, 1]"
            )
        if not (config.separate_skill0_actor and config.separate_skill0_critic):
            # Only separate skill-0 actor and critic networks are implemented.
            raise ValueError("separate_skill0_actor and separate_skill0_critic must be True")
        self.config = config
        if config.replace_skill_zero_reward_with_random_value and config.skill_dim <= 0:
            raise ValueError("skill-zero reward replacement requires skills")
        self.actor = GaussianActor(
            config.hidden_dims,
            config.action_dim,
            config.actor_layer_norm,
            config.actor_final_scale,
            config.skill_dim,
            config.skill_mean_init_std,
            config.state_dependent_std,
            config.use_anchor_in_policy,
        )
        self.critic = TwinContinuousCritic(
            config.hidden_dims,
            config.critic_layer_norm,
            config.critic_final_scale,
            config.use_anchor_in_critic,
        )
        self.potential_actor = PotentialActor(
            config.hidden_dims,
            config.action_dim,
            config.actor_layer_norm,
            config.actor_final_scale,
            config.state_dependent_std,
        )
        self.potential_critic = TwinPotentialCritic(
            config.hidden_dims,
            config.critic_layer_norm,
            config.critic_final_scale,
        )
        self.actor_optimizer = optax.adam(config.actor_lr)
        self.critic_optimizer = optax.adam(config.critic_lr)
        self.alpha_optimizer = optax.adam(config.alpha_lr)
        self._update_jit = jax.jit(self._update_impl)

    @property
    def target_entropy(self) -> float:
        default = 0.5 * self.config.action_dim
        return float(default if self.config.target_entropy is None else self.config.target_entropy)

    def _actor_entropy_loss(self, log_prob, weights, potential_mask, observations, log_alpha):
        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))
        return alpha * _weighted_mean(log_prob, weights)

    def initialize(self, key: jax.Array) -> SACState:
        actor_key, critic_key, potential_actor_key, potential_critic_key = jax.random.split(key, 4)
        observations = jnp.zeros((1, self.config.observation_dim), dtype=jnp.float32)
        skills = jnp.zeros((1, self.config.skill_dim), dtype=jnp.float32)
        actions = jnp.zeros((1, self.config.action_dim), dtype=jnp.float32)
        actor_params = self.actor.init(actor_key, observations, observations, skills)["params"]
        critic_params = self.critic.init(
            critic_key, observations, observations, skills, actions
        )["params"]
        skill0_actor_params = self.potential_actor.init(potential_actor_key, observations)["params"]
        skill0_actor_opt_state = self.actor_optimizer.init(skill0_actor_params)
        skill0_critic_params = self.potential_critic.init(potential_critic_key, observations, actions)["params"]
        skill0_critic_opt_state = self.critic_optimizer.init(skill0_critic_params)
        log_alpha = jnp.log(jnp.asarray(self.config.initial_alpha, dtype=jnp.float32))
        alpha_opt_state = self.alpha_optimizer.init(log_alpha)
        return SACState(
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=critic_params,
            skill0_actor_params=skill0_actor_params,
            skill0_critic_params=skill0_critic_params,
            target_skill0_critic_params=skill0_critic_params,
            log_alpha=log_alpha,
            actor_opt_state=self.actor_optimizer.init(actor_params),
            critic_opt_state=self.critic_optimizer.init(critic_params),
            skill0_actor_opt_state=skill0_actor_opt_state,
            skill0_critic_opt_state=skill0_critic_opt_state,
            alpha_opt_state=alpha_opt_state,
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    @staticmethod
    def _tanh_sample(
        mean: jax.Array,
        log_std: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        noise = jax.random.normal(key, mean.shape)
        raw = mean + jnp.exp(log_std) * noise
        actions = jnp.tanh(raw)
        normal_log_prob = -0.5 * (
            jnp.square((raw - mean) / jnp.exp(log_std))
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        log_prob_rows = normal_log_prob - jnp.log(1.0 - jnp.square(actions) + 1e-6)
        log_prob = jnp.sum(log_prob_rows, axis=-1)
        return actions, log_prob

    def _sample(
        self,
        actor_params: Any,
        observations: jax.Array,
        anchor_observations: jax.Array,
        skills: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        mean, log_std = self.actor.apply(
            {"params": actor_params}, observations, anchor_observations, skills
        )
        return self._tanh_sample(mean, log_std, key)

    def _sample_potential_action(
        self,
        actor_params: Any,
        observations: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        mean, log_std = self.potential_actor.apply({"params": actor_params}, observations)
        return self._tanh_sample(mean, log_std, key)

    def _sample_policy(
        self,
        state: SACState,
        observations: jax.Array,
        anchor_observations: jax.Array,
        skills: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        skill_key, potential_key = jax.random.split(key)
        actions, log_prob = self._sample(
            state.actor_params, observations, anchor_observations, skills, skill_key
        )
        potential_mask = jnp.linalg.norm(skills, axis=-1) <= 1e-6
        potential_actions, potential_log_prob = self._sample_potential_action(
            state.skill0_actor_params, observations, potential_key
        )
        actions = jnp.where(potential_mask[:, None], potential_actions, actions)
        log_prob = jnp.where(potential_mask, potential_log_prob, log_prob)
        return actions, log_prob, potential_mask

    def sample_actions(
        self,
        state: SACState,
        observations: jax.Array,
        anchor_observations: jax.Array,
        skills: jax.Array,
        keys: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        if deterministic:
            mean, _ = self.actor.apply(
                {"params": state.actor_params}, observations, anchor_observations, skills
            )
            potential_mean, _ = self.potential_actor.apply(
                {"params": state.skill0_actor_params}, observations
            )
            mask = (jnp.linalg.norm(skills, axis=-1) <= 1e-6)[:, None]
            mean = jnp.where(mask, potential_mean, mean)
            actions = jnp.tanh(mean)
            return actions
        return jax.vmap(
            lambda obs, anchor, skill, sample_key: self._sample_policy(
                state,
                obs[None],
                anchor[None],
                skill[None],
                sample_key,
            )[0][0]
        )(observations, anchor_observations, skills, keys)

    def value(
        self,
        state: SACState,
        obs: jax.Array,
        anchors: jax.Array,
        skills: jax.Array,
        *,
        deterministic: bool,
        target: bool = True,
        potential: bool = False,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Critic readout V(s, anchor, z) = min_i Q_i(s, anchor, z, a) with a from the actor.

        Deterministic: a = tanh(actor mean); otherwise a tanh-Gaussian sample drawn with ``key``.
        ``target`` reads the target critic params. ``potential=True`` marks ``skills`` as the zero
        skill: the separate skill-0 actor and critic replace the skill-conditioned ones (the skill-0
        networks see ``obs`` only).
        """
        if not deterministic and key is None:
            raise ValueError("a stochastic value readout needs a key")
        if potential:
            mean, log_std = self.potential_actor.apply({"params": state.skill0_actor_params}, obs)
        else:
            mean, log_std = self.actor.apply({"params": state.actor_params}, obs, anchors, skills)
        actions = jnp.tanh(mean) if deterministic else self._tanh_sample(mean, log_std, key)[0]
        if potential:
            params = state.target_skill0_critic_params if target else state.skill0_critic_params
            q = self.potential_critic.apply({"params": params}, obs, actions)
        else:
            params = state.target_critic_params if target else state.critic_params
            q = self.critic.apply({"params": params}, obs, anchors, skills, actions)
        return jnp.min(q, axis=-1)

    def _sample_random_skills(
        self,
        key: jax.Array,
        count: int,
        dtype: jnp.dtype,
    ) -> jax.Array:
        skills = jax.random.normal(key, (count, self.config.skill_dim), dtype=dtype)
        return skills / jnp.maximum(jnp.linalg.norm(skills, axis=-1, keepdims=True), 1e-8)

    def _next_value(self, state, batch, next_skills, potential_mask, next_action_key):
        next_actions, _, _ = self._sample_policy(
            state,
            batch.next_observations,
            batch.next_anchor_observations,
            next_skills,
            next_action_key,
        )
        next_skill_q = self.critic.apply(
            {"params": state.target_critic_params},
            batch.next_observations,
            batch.next_anchor_observations,
            next_skills,
            jax.lax.stop_gradient(next_actions),
        )
        next_q = jnp.min(next_skill_q, axis=-1)
        next_potential_q = self.potential_critic.apply(
            {"params": state.target_skill0_critic_params},
            batch.next_observations,
            jax.lax.stop_gradient(next_actions),
        )
        return jnp.where(potential_mask, jnp.min(next_potential_q, axis=-1), next_q)

    def _entropy_weights(self, observations, weights):
        return weights

    def alpha_loss(self, log_alpha, entropy):
        """Temperature loss: exp(log_alpha) * stop_gradient(entropy - target entropy)."""
        return jnp.exp(log_alpha) * jax.lax.stop_gradient(entropy - self.target_entropy)

    def _update_alpha(self, state, entropy, observations, weights):
        def alpha_loss(log_alpha):
            return self.alpha_loss(log_alpha, entropy)
        loss, grad = jax.value_and_grad(alpha_loss)(state.log_alpha)
        updates, opt_state = self.alpha_optimizer.update(grad, state.alpha_opt_state, state.log_alpha)
        return optax.apply_updates(state.log_alpha, updates), opt_state, loss

    def _potential_context(self, observations, anchors):
        return observations, anchors

    def _skill_value(self, state, obs, anchors, skills, key):
        actions, _ = self._sample(state.actor_params, obs, anchors, skills, key)
        return jnp.min(self.critic.apply({"params": state.target_critic_params}, obs, anchors, skills, jax.lax.stop_gradient(actions)), axis=-1)






    def _actor_values(self, state, params, potential_params, batch, actor_key):
        potential_mask = jnp.linalg.norm(batch.skills, axis=-1) <= 1e-6
        skill_key, potential_key = jax.random.split(actor_key)
        actions, log_prob = self._sample(
            params,
            batch.observations,
            batch.anchor_observations,
            batch.skills,
            skill_key,
        )
        potential_actions, potential_log_prob = self._sample_potential_action(
            potential_params, batch.observations, potential_key
        )
        actions = jnp.where(potential_mask[:, None], potential_actions, actions)
        log_prob = jnp.where(potential_mask, potential_log_prob, log_prob)
        q_values = self.critic.apply(
            {"params": state.critic_params},
            batch.observations,
            batch.anchor_observations,
            batch.skills,
            actions,
        )
        minimum_q = jnp.min(q_values, axis=-1)
        potential_q = self.potential_critic.apply(
            {"params": state.skill0_critic_params},
            batch.observations,
            actions,
        )
        minimum_q = jnp.where(potential_mask, jnp.min(potential_q, axis=-1), minimum_q)
        return log_prob, minimum_q


    def potential_reward(self, state, batch, potential_mask, random_skill_key):
        """Rewards with z=0 rows replaced by the discounted potential.

        A z=0 row gets gamma^max(switch - t, 1) * max(E_z[min twin target Q], 0).
        Returns (rewards, potential_value); potential_value is zero when the
        replacement is off.
        """
        rewards = batch.rewards
        potential_value = jnp.zeros_like(rewards)
        if self.config.replace_skill_zero_reward_with_random_value:
            value_observations = batch.observations
            value_anchors = batch.anchor_observations
            value_observations, value_anchors = self._potential_context(value_observations, value_anchors)
            def _potential_value_draw(draw_key: jax.Array) -> jax.Array:
                skill_key, action_key = jax.random.split(draw_key)
                draw_skills = self._sample_random_skills(
                    skill_key, batch.skills.shape[0], batch.skills.dtype
                )
                # min over the TWIN CRITICS (standard clipped double-Q), never
                # over skills: the skill axis is an expectation, averaged below.
                return self._skill_value(state, value_observations, value_anchors, draw_skills, action_key)

            draws = self.config.skill0_value_num_skills
            if draws == 1:
                potential_value = _potential_value_draw(
                    jax.random.fold_in(random_skill_key, 0)
                )
            else:
                potential_value = jnp.mean(
                    jax.vmap(_potential_value_draw)(
                        jax.random.split(random_skill_key, draws)
                    ),
                    axis=0,
                )
            # Floor the state estimate AFTER averaging skills, before discounting.
            potential_value = jnp.maximum(potential_value, 0.0)
            delay = jnp.maximum(
                potential_switch_steps(
                    batch.episode_ids, self.config.potential_steps,
                    self.config.random_potential_switch_max_steps, self.config.potential_switch_seed,
                ).astype(batch.timesteps.dtype)
                - batch.timesteps,
                1,
            )
            scale = jnp.power(
                jnp.asarray(self.config.discount, dtype=rewards.dtype),
                delay.astype(rewards.dtype),
            )
            if self.config.replace_skill_zero_reward_with_random_value:
                replacement_reward = scale * potential_value
                rewards = jnp.where(potential_mask, replacement_reward, rewards)

        return rewards, potential_value

    def critic_loss(self, params, potential_params, batch, target_q, weights, potential_mask):
        """Phase-balanced squared TD error of the twin critics; returns (loss, predicted)."""
        predicted = self.critic.apply(
            {"params": params},
            batch.observations,
            batch.anchor_observations,
            batch.skills,
            batch.actions,
        )
        potential_predicted = self.potential_critic.apply(
            {"params": potential_params}, batch.observations, batch.actions
        )
        predicted = jnp.where(potential_mask[:, None], potential_predicted, predicted)
        squared_error = jnp.square(predicted - target_q[:, None])
        return _phase_balanced_mean(squared_error, weights, potential_mask), predicted

    def actor_loss(self, params, potential_params, state, batch, weights, potential_mask, actor_key):
        """Entropy term minus phase-balanced min-Q; returns (loss, (log_prob, minimum_q))."""
        log_prob, minimum_q = self._actor_values(
            state, params, potential_params, batch, actor_key
        )
        q_balanced = _phase_balanced_mean(minimum_q, weights, potential_mask)
        loss = self._actor_entropy_loss(log_prob, weights, potential_mask,
                                        batch.observations, state.log_alpha) - q_balanced
        return loss, (log_prob, minimum_q)

    def _update_impl(
        self,
        state: SACState,
        batch: Transition,
        key: jax.Array,
    ) -> tuple[SACState, SACMetrics]:
        (
            next_action_key,
            random_skill_key,
            random_action_key,
            actor_key,
        ) = jax.random.split(key, 4)
        weights = batch.loss_weights
        potential_mask = jnp.linalg.norm(batch.skills, axis=-1) <= 1e-6
        rewards, potential_value = self.potential_reward(state, batch, potential_mask, random_skill_key)

        next_skills = batch.next_skills
        if self.config.replace_skill_zero_reward_with_random_value:
            next_skills = jnp.where(potential_mask[:, None], jnp.zeros_like(next_skills), next_skills)
        next_q = self._next_value(state, batch, next_skills, potential_mask, next_action_key)
        target_q = (
            (1.0 - self.config.discount) * rewards
            + self.config.discount * batch.masks * next_q
        )
        target_q = jax.lax.stop_gradient(target_q)

        def critic_loss(
            params: Any,
            potential_params: Any,
        ) -> tuple[jax.Array, jax.Array]:
            return self.critic_loss(params, potential_params, batch, target_q, weights, potential_mask)

        (critic_aux, critic_grads) = jax.value_and_grad(
            critic_loss, argnums=(0, 1), has_aux=True
        )(state.critic_params, state.skill0_critic_params)
        (critic_loss_value, predicted_q) = critic_aux
        critic_grad, potential_critic_grad = critic_grads
        potential_critic_updates, skill0_critic_opt_state = self.critic_optimizer.update(
            potential_critic_grad,
            state.skill0_critic_opt_state,
            state.skill0_critic_params,
        )
        skill0_critic_params = optax.apply_updates(
            state.skill0_critic_params, potential_critic_updates
        )
        critic_updates, critic_opt_state = self.critic_optimizer.update(
            critic_grad,
            state.critic_opt_state,
            state.critic_params,
        )
        critic_params = optax.apply_updates(state.critic_params, critic_updates)

        def actor_loss(
            params: Any,
            potential_params: Any,
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            return self.actor_loss(params, potential_params, state, batch, weights, potential_mask, actor_key)

        (actor_aux, actor_grads) = jax.value_and_grad(
            actor_loss, argnums=(0, 1), has_aux=True
        )(state.actor_params, state.skill0_actor_params)
        (actor_loss_value, (log_prob, minimum_q)) = actor_aux
        actor_grad, potential_actor_grad = actor_grads
        potential_actor_updates, skill0_actor_opt_state = self.actor_optimizer.update(
            potential_actor_grad,
            state.skill0_actor_opt_state,
            state.skill0_actor_params,
        )
        skill0_actor_params = optax.apply_updates(
            state.skill0_actor_params, potential_actor_updates
        )
        actor_updates, actor_opt_state = self.actor_optimizer.update(
            actor_grad,
            state.actor_opt_state,
            state.actor_params,
        )
        actor_params = optax.apply_updates(state.actor_params, actor_updates)

        entropy = _weighted_mean(-log_prob, self._entropy_weights(batch.observations, weights))
        log_alpha, alpha_opt_state, alpha_loss_value = self._update_alpha(state, entropy, batch.observations, weights)
        target_critic_params = _polyak(state.target_critic_params, critic_params, self.config.tau)
        target_skill0_critic_params = _polyak(
            state.target_skill0_critic_params,
            skill0_critic_params,
            self.config.tau,
        )
        next_state = SACState(
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            skill0_actor_params=skill0_actor_params,
            skill0_critic_params=skill0_critic_params,
            target_skill0_critic_params=target_skill0_critic_params,
            log_alpha=log_alpha,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            skill0_actor_opt_state=skill0_actor_opt_state,
            skill0_critic_opt_state=skill0_critic_opt_state,
            alpha_opt_state=alpha_opt_state,
            step=state.step + 1,
        )
        potential_weights = weights * potential_mask.astype(weights.dtype)
        return next_state, SACMetrics(
            critic_loss=critic_loss_value,
            actor_loss=actor_loss_value,
            alpha_loss=alpha_loss_value,
            alpha=jnp.exp(log_alpha),
            entropy=entropy,
            q_mean=_weighted_mean(minimum_q, weights),
            target_q_mean=_weighted_mean(target_q, weights),
            replay_weight=jnp.sum(weights),
            potential_weight=jnp.sum(potential_weights),
            potential_skill_value_weight=(
                jnp.sum(potential_weights)
                if self.config.replace_skill_zero_reward_with_random_value
                else jnp.asarray(0.0, dtype=weights.dtype)
            ),
            potential_skill_value_total=jnp.sum(potential_value * potential_weights),
            potential_reward_total=jnp.sum(rewards * potential_weights),
            potential_q_total=jnp.sum(jnp.min(predicted_q, axis=-1) * potential_weights),
            potential_target_total=jnp.sum(target_q * potential_weights),
            potential_td_squared_total=jnp.sum(
                jnp.mean(jnp.square(predicted_q - target_q[:, None]), axis=-1)
                * potential_weights
            ),
            policy_reward_skill_on=(
                # potential_weights = weights * potential_mask, so the complement is
                # the skill-on share. `batch.rewards` is the intrinsic reward
                # handed to the learner.
                jnp.sum(
                    batch.rewards
                    * (weights - potential_weights)
                )
                / jnp.maximum(jnp.sum(weights - potential_weights), 1.0)
            ),
        )

    def update(self, state: SACState, batch: Transition, key: jax.Array) -> tuple[SACState, SACMetrics]:
        return self._update_jit(state, batch, key)


def _zero_sac_metrics() -> SACMetrics:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return SACMetrics(*(zero for _ in SACMetrics._fields))

def _zero_skill_metrics() -> SkillMetrics:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return SkillMetrics(*(zero for _ in SkillMetrics._fields))

def _zero_relabel_metrics() -> RelabelMetrics:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    one = jnp.asarray(1.0, dtype=jnp.float32)
    return RelabelMetrics(zero, zero, zero, zero, one, one)

def _zero_archive_metrics() -> ArchiveMetrics:
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return ArchiveMetrics(*(zero for _ in ArchiveMetrics._fields))

class SEEAgent:
    """SEE learning: sampled pairs, skill rewards, relabeling, and SAC updates."""

    def __init__(self, sac, skill, config, replay, representation_replay):
        self.sac = sac
        self.skill = skill
        self.config = config
        self.replay = replay
        self.representation_replay = representation_replay
        self.uses_marginal_branches = config.uses_marginal_branches

    def _switch_steps(self, episode_ids):
        return potential_switch_steps(
            episode_ids, self.config.collector_potential_steps,
            self.config.random_potential_switch_max_steps, self.config.potential_switch_seed,
        )

    def update(self, sac_state, skill_state, replay_state, representation_state,
               env_steps, update_root, ready):
        """Apply the configured learner updates to the current replay archives."""
        metric_shape = (self.config.updates_per_chunk,)

        def one_update(
            carry: tuple[SACState, SkillState, SkillMetrics, ArchiveMetrics, jax.Array],
            index: jax.Array,
        ) -> tuple[
            tuple[SACState, SkillState, SkillMetrics, ArchiveMetrics, jax.Array],
            tuple[
                SACMetrics,
                SkillMetrics,
                RelabelMetrics,
                ArchiveMetrics,
                AnchorReplayMetrics,
            ],
        ]:
            (
                sac_state,
                skill_state,
                last_skill_metrics,
                last_archive_metrics,
                update_key,
            ) = carry
            keys = jax.random.split(update_key, 8)
            (
                update_key,
                replay_sample_key,
                replay_anchor_key,
                reward_key,
                sac_key,
                representation_sample_key,
                representation_future_key,
                skill_key,
            ) = keys
            replay_indices = self.replay.sample_indices(
                replay_state,
                replay_sample_key,
                self.config.batch_size,
            )
            (
                replay_batch,
                reward_batch,
                relabel_allowed,
                relabel_outcomes,
                anchor_replay_metrics,
            ) = policy_hindsight_batch(
                self.config,
                self.replay,
                replay_state,
                replay_indices,
                replay_anchor_key,
                hindsight=True,
            )
            if self.config.achieved_skill_relabel_fraction < 1.0:
                # fold_in keeps every other RNG stream untouched, so the
                # default (1.0) draws nothing extra.
                relabel_allowed = relabel_allowed & jax.random.bernoulli(
                    jax.random.fold_in(replay_anchor_key, 1),
                    p=self.config.achieved_skill_relabel_fraction,
                    shape=relabel_allowed.shape,
                )
            replay_batch, relabel_metrics = self.skill.relabel_achieved_skills(
                skill_state,
                replay_batch,
                relabel_allowed,
                outcome_observations=relabel_outcomes,
            )
            reward_batch = reward_batch.replace(
                anchor_observations=replay_batch.anchor_observations,
                skills=replay_batch.skills,
            )
            if self.config.reward_partition_anneal_end_steps:
                intrinsic = self.skill.intrinsic_reward(
                    skill_state, reward_batch, reward_key,
                    partition_weight=self.config.reward_partition_weight(env_steps),
                )
            else:
                intrinsic = self.skill.intrinsic_reward(skill_state, reward_batch, reward_key)
            sac_batch = replay_batch.replace(rewards=intrinsic)
            sac_state, sac_metrics = self.sac.update(sac_state, sac_batch, sac_key)
            def encoder_step(
                state_in: SkillState,
            ) -> tuple[SkillState, SkillMetrics, ArchiveMetrics]:
                slots, rows, row_valid = self.representation_replay.sample_active(
                    representation_state,
                    representation_sample_key,
                    self.config.batch_size,
                )
                (
                    encoder_batch,
                    step_archive_metrics,
                ) = representation_archive_batch(
                    self.config,
                    self.representation_replay,
                    representation_state,
                    slots,
                    rows,
                    row_valid,
                    representation_future_key,
                )
                state_out, step_skill_metrics = self.skill.update_preprocessed(
                    state_in,
                    encoder_batch,
                    skill_key,
                )
                return state_out, step_skill_metrics, step_archive_metrics

            # The encoder does not need SAC's replay ratio. On skipped steps the
            # previous metrics are re-emitted so the segment mean stays the mean
            # over the encoder steps that actually ran (index 0 always runs, so
            # the carried value is never the initial zero).
            skill_state, skill_metrics, archive_metrics = jax.lax.cond(
                (index % self.config.encoder_update_stride) == 0,
                encoder_step,
                lambda state_in: (
                    state_in,
                    last_skill_metrics,
                    last_archive_metrics,
                ),
                skill_state,
            )
            return (
                sac_state,
                skill_state,
                skill_metrics,
                archive_metrics,
                update_key,
            ), (
                sac_metrics,
                skill_metrics,
                relabel_metrics,
                archive_metrics,
                anchor_replay_metrics,
            )

        def run_updates(
            update_state: tuple[SACState, SkillState, SkillMetrics, ArchiveMetrics, jax.Array],
        ) -> tuple[
            tuple[SACState, SkillState],
            tuple[
                SACMetrics,
                SkillMetrics,
                RelabelMetrics,
                ArchiveMetrics,
                AnchorReplayMetrics,
            ],
        ]:
            (sac_state, skill_state, _, _, _), metrics = jax.lax.scan(
                one_update,
                update_state,
                jnp.arange(self.config.updates_per_chunk, dtype=jnp.int32),
            )
            return (sac_state, skill_state), metrics

        def skip_updates(
            update_state: tuple[SACState, SkillState, SkillMetrics, ArchiveMetrics, jax.Array],
        ) -> tuple[
            tuple[SACState, SkillState],
            tuple[
                SACMetrics,
                SkillMetrics,
                RelabelMetrics,
                ArchiveMetrics,
                AnchorReplayMetrics,
            ],
        ]:
            sac_state, skill_state, _, _, _ = update_state
            broadcast = lambda value: jnp.broadcast_to(value, metric_shape)
            return (sac_state, skill_state), (
                jax.tree_util.tree_map(broadcast, _zero_sac_metrics()),
                jax.tree_util.tree_map(broadcast, _zero_skill_metrics()),
                jax.tree_util.tree_map(broadcast, _zero_relabel_metrics()),
                jax.tree_util.tree_map(broadcast, _zero_archive_metrics()),
                jax.tree_util.tree_map(broadcast, _zero_anchor_replay_metrics()),
            )

        return jax.lax.cond(
            ready,
            run_updates,
            skip_updates,
            (
                sac_state,
                skill_state,
                _zero_skill_metrics(),
                _zero_archive_metrics(),
                update_root,
            ),
        )


def create_learners(config: ExperimentConfig, env: Any) -> tuple[ContinuousSAC, SkillLearner, OffPolicyConfig]:
    """Construct the networks and update settings from one resolved recipe."""
    from empowerment_warp.training.sac.config import OffPolicyConfig

    observation_dim = int(env.observation_size)
    encoder_observation_indices = None
    encoder_observation_dim = observation_dim
    if hasattr(env, "encoder_observation_indices"):
        encoder_observation_indices = env.encoder_observation_indices(config.encoder_obs_mode)
        if encoder_observation_indices is not None:
            encoder_observation_dim = len(encoder_observation_indices)
    elif config.encoder_obs_mode != "full":
        raise ValueError(
            f"{config.environment} does not implement encoder_obs_mode={config.encoder_obs_mode!r}"
        )
    if config.uses_soccer_dynamics:
        expected_encoder_dim = 42 if config.encoder_obs_mode == "full" else 2
        if not env.soccer or observation_dim != 42 or encoder_observation_dim != expected_encoder_dim:
            raise RuntimeError(
                "AntSoccer observation contract failed: "
                f"soccer={env.soccer}, policy_dim={observation_dim}, "
                f"encoder_dim={encoder_observation_dim}; expected policy_dim=42 and "
                f"encoder_dim={expected_encoder_dim}"
            )
    skill_dim = config.skill_dim
    sac_config = SACConfig(
        episode_length=config.episode_length,
        observation_dim=observation_dim,
        action_dim=env.action_size,
        skill_dim=skill_dim,
        hidden_dims=config.hidden_dims,
        target_entropy=config.target_entropy,
        discount=config.discount,
        tau=config.critic_tau,
        actor_layer_norm=config.actor_layer_norm,
        critic_layer_norm=config.critic_layer_norm,
        actor_final_scale=config.actor_final_scale,
        critic_final_scale=config.critic_final_scale,
        skill_mean_init_std=config.skill_mean_init_std,
        state_dependent_std=config.state_dependent_std,
        use_anchor_in_policy=config.use_anchor_in_policy,
        use_anchor_in_critic=config.use_anchor_in_critic,
        skill0_value_num_skills=config.skill0_value_num_skills,
        replace_skill_zero_reward_with_random_value=(
            config.replace_skill_zero_reward_with_random_value
        ),
        separate_skill0_actor=config.separate_skill0_actor,
        separate_skill0_critic=config.separate_skill0_critic,
        potential_steps=config.potential_steps,
        random_potential_switch_max_steps=config.random_potential_switch_max_steps,
        potential_switch_seed=config.seed,
        potential_replay_mode=config.potential_replay_mode,
    )
    skill = SkillLearner(
        SkillConfig(
            method=config.algorithm,
            observation_dim=encoder_observation_dim,
            skill_dim=skill_dim,
            discrete_skills=config.discrete_skills,
            observation_indices=encoder_observation_indices,
            num_skills=config.num_skills if config.discrete_skills else 0,
            discrete_skill_library=config.discrete_skill_library,
            skill_library_seed=config.seed,
            hidden_dims=config.hidden_dims,
            target_tau=config.encoder_target_tau,
            temperature=config.encoder_temperature,
            encoder_objective=config.encoder_objective,
            phi_norm_cap=config.phi_norm_cap,
            phi_noise_std=config.phi_noise_std,
            reward_encoder=config.reward_encoder,
            discrete_reward_softmax=config.discrete_reward_softmax,
            clip_discrete_reward=config.clip_discrete_reward,
            achieved_skill_relabeling=config.achieved_skill_relabeling,
            achieved_skill_relabel_min_prob=config.achieved_skill_relabel_min_prob,
            achieved_skill_relabel_min_norm=config.achieved_skill_relabel_min_norm,
            encoder_skill_source=config.encoder_skill_source,
            pair_form=config.pair_form,
            achieved_skill_relabel_weight_mode=config.achieved_skill_relabel_weight_mode,
            achieved_skill_relabel_effective_beta=(
                config.achieved_skill_relabel_effective_beta
            ),
            achieved_skill_relabel_weight_max=config.achieved_skill_relabel_weight_max,
            layer_norm=config.encoder_layer_norm,
            potential_reward_reference=config.potential_reward_reference,
            potential_reward_scale=config.potential_reward_scale,
            continuous_reward_temperature=config.continuous_reward_temperature,
            continuous_reward_normalization=config.continuous_reward_normalization,
        ),
        ema_tau_one_shortcut=False,
    )
    sac = ContinuousSAC(sac_config)
    off_policy_config = OffPolicyConfig(
        random_potential_switch_max_steps=config.random_potential_switch_max_steps,
        potential_switch_seed=config.seed,
        reward_partition_anneal_start_steps=config.reward_partition_anneal_start_steps,
        reward_partition_anneal_end_steps=config.reward_partition_anneal_end_steps,
        rollout_steps=config.rollout_steps,
        updates_per_chunk=config.updates_per_chunk,
        encoder_updates_per_chunk=config.encoder_updates_per_chunk,
        branch_transitions_to_policy_replay=(
            config.branch_transitions_to_policy_replay
        ),
        branch_targets_per_anchor=config.branch_targets_per_anchor,
        branch_phase_groups_override=config.branch_phase_groups,
        live_potential_only=config.live_potential_only,
        branch_live_two_phase=config.branch_live_two_phase,
        batch_size=config.batch_size,
        replay_capacity=config.replay_capacity,
        episode_length=config.episode_length,
        representation_episodes=config.representation_episodes,
        future_gamma=config.future_gamma,
        max_future_steps=config.max_future_steps,
        sample_hindsight_anchors=config.sample_hindsight_anchors,
        potential_replay_mode=config.potential_replay_mode,
        achieved_skill_relabel_fraction=config.achieved_skill_relabel_fraction,
        achieved_skill_relabel_outcome=config.achieved_skill_relabel_outcome,
        achieved_skill_relabel_samples=config.achieved_skill_relabel_samples,
        encoder_skill_source=config.encoder_skill_source,
        potential_lane_fraction=config.potential_lane_fraction,
        potential_steps=config.potential_steps,
        skill_start_mode=config.skill_start_mode,
        stagger_reset_groups=(
            max(1, config.episode_length // config.rollout_steps)
            if config.staggered_resets
            else 0
        ),
    )
    return sac, skill, off_policy_config
