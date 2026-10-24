"""Representation and intrinsic reward learner for the potential method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

from flax import struct
import jax
import jax.numpy as jnp
import optax

from empowerment_warp.utils.objectives import in_batch_skill_accuracy, sphere_log_partition, sampled_infonce_log_partition, forward_infonce, vmf_forward_loss, vmf_posterior_kl
from empowerment_warp.utils.networks import MLP


SkillMethod = Literal["potential"]


@dataclass(frozen=True, slots=True)
class SkillConfig:
    method: SkillMethod
    observation_dim: int
    skill_dim: int
    discrete_skills: bool
    observation_indices: tuple[int, ...] | None = None
    num_skills: int = 0
    discrete_skill_library: str = "random"
    skill_library_seed: int = 0
    hidden_dims: tuple[int, ...] = (512, 512)
    learning_rate: float = 3e-4
    target_tau: float = 0.005
    temperature: float = 0.1
    encoder_objective: str = "auto"
    phi_norm_cap: float = 5.0
    phi_noise_std: float = 0.0
    # Which COPY of the encoder scores the reward / infers the achieved skill for relabeling:
    # "live" = the trained parameters (moves every encoder update), "slow" = the exponential
    # moving average of them. Named by the copy, not "target"/"EMA", because "target" is the
    # label's outcome state elsewhere and the SAC Q-target -- three meanings for one word.
    reward_encoder: str = "live"
    # Which encoder copy infers the achieved skill for relabeling ("live" or "slow").
    relabel_encoder: str = "live"
    continuous_reward_temperature: bool = False
    continuous_reward_normalization: str = "none"
    discrete_reward_softmax: bool = False
    clip_discrete_reward: bool = False
    achieved_skill_relabeling: bool = False
    achieved_skill_relabel_min_prob: float = 0.0
    # Continuous-skill relabel gate: rows whose representation displacement norm ||G|| is below
    # this keep their commanded skill (0 = reference parity, norm-eps validity only). The
    # discrete min_prob gate never applies to continuous draws, so this is the only gate there.
    achieved_skill_relabel_min_norm: float = 0.0
    # Which skill labels the ENCODER's InfoNCE rows: "commanded" (the reference: the skill the
    # policy was told to run) or "achieved_discounted" (the same HER label the actor/critic get:
    # z = unit(G), G = mean over M sampled futures of phi(s_{t+k}) - phi(s_t), Geometric(1-gamma)
    # inside the row's segment, stop-grad). The contrastive PAIR stays one sampled future, so the
    # positive cosine is the alignment of one future with the segment's mean direction -- not 1.
    # Continuous skills only (relabeling the K-way CE trains it on its own argmax: degenerate).
    encoder_skill_source: str = "commanded"
    # Pair form: "difference" = the bilinear form <phi(s') - phi(s), z> (METRA / CSF /
    # ours); "pair_mlp" = an arbitrary decoder q(z | s, s'): the logit is <f(s, s'), z> with f an
    # MLP over the concatenated pair (same InfoNCE, same reward, same HER label), so the
    # representation is no longer constrained to be a difference of per-state embeddings.
    pair_form: str = "difference"
    achieved_skill_relabel_weight_mode: str = "none"
    achieved_skill_relabel_effective_beta: float = 0.99
    achieved_skill_relabel_weight_max: float = 3.0
    # potential_reward_* (serialized names) configure the classifier's skill reward,
    # not the potential reward E_z[V(s, z)].
    # Reference the skill reward against the row's OWN state instead of the
    # fixed anchor. Sum_t <phi(s_t+1) - phi(s_t), z> telescopes to
    # <phi(s_T) - phi(s_0), z>, so the RETURN is the same quantity while the
    # per-step signal is dense -- better shaped for credit assignment.
    # NOTE the telescoping holds only for the RAW inner product; with
    # discrete_reward_softmax=True the reward is ln(K*p_true) of a ONE-STEP
    # displacement, which barely moves phi and reads ~0.
    potential_reward_reference: str = "anchor"
    # Multiply the skill reward (hence the critic targets and the potential
    # value) by this factor. The delta reward is ~1/(1-gamma) smaller than the
    # anchor reward, so at equal SAC settings its learning dynamics differ;
    # scaling it up matches the effective magnitude. Default 1.0 = unchanged.
    potential_reward_scale: float = 1.0
    layer_norm: bool = True
    # Global-norm clip on the encoder gradient. 0 = off (byte-identical; SAC default).
    # PQN/craftax enables it: phi saturates at phi_norm_cap on 1345-dim obs, so a
    # gradient spike on an unclipped Adam step can NaN the encoder.
    encoder_max_grad_norm: float = 0.0

    def __post_init__(self) -> None:
        # Only continuous skills are implemented. The field stays because saved run
        # configs carry it. num_skills, discrete_skill_library, skill_library_seed,
        # discrete_reward_softmax, clip_discrete_reward, achieved_skill_relabel_min_prob,
        # achieved_skill_relabel_weight_mode, achieved_skill_relabel_effective_beta and
        # achieved_skill_relabel_weight_max were read only by those branches; each engine's run
        # config pins them to the values its runs recorded.
        if self.discrete_skills:
            raise ValueError("discrete skills are not supported; discrete_skills must be False")
        if self.encoder_objective == "auto":
            object.__setattr__(self, "encoder_objective", "forward_infonce")
        if self.observation_indices is not None:
            object.__setattr__(self, "observation_indices", tuple(self.observation_indices))
        if self.method != "potential":
            raise ValueError(f"unsupported skill method: {self.method}")
        if self.observation_dim <= 0 or self.skill_dim <= 0:
            raise ValueError("observation_dim and skill_dim must be positive")
        if self.observation_indices is not None:
            if len(self.observation_indices) != self.observation_dim:
                raise ValueError("observation_indices length must equal observation_dim")
            if len(set(self.observation_indices)) != len(self.observation_indices):
                raise ValueError("observation_indices must be unique")
            if any(index < 0 for index in self.observation_indices):
                raise ValueError("observation_indices must be nonnegative")
        if self.num_skills != 0:
            raise ValueError("continuous skills use num_skills=0")
        if self.learning_rate <= 0 or not 0.0 < self.target_tau <= 1.0:
            raise ValueError("learning_rate must be positive and target_tau in (0,1]")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.encoder_objective not in {"symmetric_infonce", "forward_infonce", "vmf_forward"}:
            raise ValueError("unsupported encoder_objective")
        if self.encoder_objective == "symmetric_infonce":
            raise ValueError("symmetric continuous-skill CSF is not supported; select forward_infonce or vmf_forward")
        if self.encoder_objective == "forward_infonce" and (
            self.method != "potential"
            or self.encoder_skill_source not in {"commanded", "achieved_discounted"}
        ):
            raise ValueError("forward_infonce requires potential, continuous skills")
        if self.encoder_objective == "vmf_forward" and (
            self.method != "potential" or self.skill_dim not in (2, 4, 9, 16)
            or self.encoder_skill_source != "commanded"
        ):
            raise ValueError("vmf_forward requires potential, continuous 2/4/9/16-D commanded skills")
        if self.phi_noise_std < 0.0:
            raise ValueError("phi_noise_std must be nonnegative")
        if self.potential_reward_reference not in {"anchor", "current"}:
            raise ValueError("potential_reward_reference must be 'anchor' or 'current'")
        if self.continuous_reward_normalization not in {"none", "sphere", "sampled_infonce"}:
            raise ValueError("unsupported continuous_reward_normalization")
        if self.continuous_reward_normalization == "sampled_infonce" and (
            self.encoder_objective != "forward_infonce" or not self.continuous_reward_temperature
            or self.potential_reward_reference != "anchor"
        ):
            raise ValueError("sampled InfoNCE rewards require forward InfoNCE and temperature-scaled anchor rewards")
        if (self.pair_form == "pair_mlp" and self.continuous_reward_normalization == "sphere"
                and self.potential_reward_reference != "anchor"):
            raise ValueError("normalized pair_mlp requires anchor rewards")
        if self.continuous_reward_normalization == "sphere" and (
            self.method != "potential" or self.skill_dim not in (2, 4, 9, 16)
            or not self.continuous_reward_temperature
        ):
            raise ValueError("sphere reward requires continuous 2/4/9/16-D potential scores with temperature")
        if self.reward_encoder not in {"live", "slow"}:
            raise ValueError("reward_encoder must be 'live' or 'slow'")
        if self.relabel_encoder not in {"live", "slow"}:
            raise ValueError("relabel_encoder must be 'live' or 'slow'")
        if self.achieved_skill_relabel_min_norm < 0.0:
            raise ValueError("achieved_skill_relabel_min_norm must be nonnegative")
        if self.encoder_skill_source not in {"commanded", "achieved_discounted"}:
            raise ValueError("encoder_skill_source must be 'commanded' or 'achieved_discounted'")
        if self.pair_form not in {"difference", "pair_mlp"}:
            raise ValueError("pair_form must be 'difference' or 'pair_mlp'")


@struct.dataclass
class SkillBatch:
    observations: jax.Array
    next_observations: jax.Array
    future_observations: jax.Array
    anchor_observations: jax.Array
    skills: jax.Array
    future_valid: jax.Array
    # M sampled futures per row for the achieved-discounted ENCODER label, [rows, M, obs...];
    # None unless encoder_skill_source == "achieved_discounted".
    label_future_observations: jax.Array | None = None


@struct.dataclass
class SkillState:
    params: Any
    target_params: Any
    optimizer_state: optax.OptState
    log_dual: jax.Array
    dual_optimizer_state: optax.OptState
    input_mean: jax.Array
    input_var: jax.Array
    target_mean: jax.Array
    target_var: jax.Array
    step: jax.Array


class SkillMetrics(NamedTuple):
    loss: jax.Array
    intrinsic_reward_mean: jax.Array
    accuracy: jax.Array
    mi_lower_bound: jax.Array
    alignment: jax.Array
    log_partition: jax.Array


class RelabelMetrics(NamedTuple):
    allowed_fraction: jax.Array
    relabeled_fraction: jax.Array
    changed_fraction: jax.Array
    confidence_mean: jax.Array
    weight_mean: jax.Array
    weight_max: jax.Array


def sample_skills(config: SkillConfig, key: jax.Array, count: int) -> jax.Array:
    samples = jax.random.normal(key, (count, config.skill_dim))
    return samples / jnp.sqrt(jnp.sum(jnp.square(samples), axis=-1, keepdims=True) + 1e-8)


class SkillLearner:
    """One representation optimizer with method-specific pure losses and rewards."""

    def __init__(self, config: SkillConfig, *, ema_tau_one_shortcut: bool = True):
        self.config = config
        # Engine-specific slow-copy update at target_tau == 1.0: the PQN engine copies the live
        # parameters outright (True); the SAC engine always runs the Polyak expression (False).
        self.ema_tau_one_shortcut = ema_tau_one_shortcut
        output_dim = config.skill_dim
        self.model = MLP(config.hidden_dims, output_dim, config.layer_norm)
        # pair_mlp: f(s, s') over the concatenated pair; params live under params["pair"] with the
        # (unused for the logit) per-state trunk under params["phi"], so one optimizer covers both.
        self.pair_model = MLP(config.hidden_dims, output_dim, config.layer_norm) if config.pair_form == "pair_mlp" else None
        if config.encoder_max_grad_norm > 0.0:
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(config.encoder_max_grad_norm),
                optax.adam(config.learning_rate),
            )
        else:
            self.optimizer = optax.adam(config.learning_rate)
        self.dual_optimizer = optax.adam(config.learning_rate)
        self._update_jit = jax.jit(self._update_impl)
        self._intrinsic_jit = jax.jit(self._intrinsic_impl)

    def initialize(self, key: jax.Array) -> SkillState:
        example = jnp.zeros((1, self.config.observation_dim), dtype=jnp.float32)
        params = self.model.init(key, example)["params"]
        if self.pair_model is not None:
            pair_key = jax.random.fold_in(key, 11)
            pair_example = jnp.zeros((1, 2 * self.config.observation_dim), dtype=jnp.float32)
            params = {"phi": params, "pair": self.pair_model.init(pair_key, pair_example)["params"]}
        log_dual = jnp.log(jnp.asarray(30.0, dtype=jnp.float32))
        return SkillState(
            params=params,
            target_params=params,
            optimizer_state=self.optimizer.init(params),
            log_dual=log_dual,
            dual_optimizer_state=self.dual_optimizer.init(log_dual),
            input_mean=jnp.zeros((self.config.observation_dim,), dtype=jnp.float32),
            input_var=jnp.ones((self.config.observation_dim,), dtype=jnp.float32),
            target_mean=jnp.zeros((self.config.observation_dim,), dtype=jnp.float32),
            target_var=jnp.ones((self.config.observation_dim,), dtype=jnp.float32),
            step=jnp.asarray(0, dtype=jnp.int32),
        )

    def sample_skills(self, key: jax.Array, count: int) -> jax.Array:
        samples = jax.random.normal(key, (count, self.config.skill_dim))
        return samples / jnp.sqrt(
            jnp.sum(jnp.square(samples), axis=-1, keepdims=True) + 1e-8
        )

    def displacement(self, params: Any, anchor_observations: jax.Array, outcome_observations: jax.Array) -> jax.Array:
        """The representation displacement the logit <., z> is taken against: phi(s') - phi(s) for
        the difference form, f(s, s') for the pair-MLP decoder (norm-capped like phi)."""
        if self.config.pair_form == "pair_mlp":
            pair = jnp.concatenate((anchor_observations, outcome_observations), axis=-1)
            return self._cap(self.pair_model.apply({"params": params["pair"]}, pair))
        return self._apply(params, outcome_observations) - self._apply(params, anchor_observations)

    def _cap(self, value: jax.Array) -> jax.Array:
        if self.config.phi_norm_cap <= 0:
            return value
        # Safe norm: jnp.linalg.norm has a 0/0 = NaN GRADIENT at value==0. Zero-vector
        # rows reach here from the encoder archive's masked filler (beyond-episode-length
        # slots are zeros) and NaN the backward via `0 * NaN` in the where-masked loss.
        # sqrt(sum(v^2)+eps) has a finite (0) gradient at v==0; eps=1e-12 is below float32
        # ULP for realistic norms, so non-zero rows (all of SAC's) are numerically unchanged.
        norm = jnp.sqrt(jnp.sum(jnp.square(value), axis=-1, keepdims=True) + 1e-12)
        scale = jnp.minimum(1.0, self.config.phi_norm_cap / jnp.maximum(norm, 1e-8))
        return value * scale

    def _apply(self, params: Any, observations: jax.Array) -> jax.Array:
        if isinstance(params, dict) and "phi" in params and "pair" in params:
            params = params["phi"]
        return self._cap(self.model.apply({"params": params}, observations))

    def encoder_observations(self, observations: jax.Array) -> jax.Array:
        if self.config.observation_indices is None:
            return observations
        return observations[..., self.config.observation_indices]


    def preprocess_batch(self, batch: SkillBatch) -> SkillBatch:
        """Gather the intrinsic view while leaving replay and SAC observations full."""

        return batch.replace(
            observations=self.encoder_observations(batch.observations),
            next_observations=self.encoder_observations(batch.next_observations),
            future_observations=self.encoder_observations(batch.future_observations),
            anchor_observations=self.encoder_observations(batch.anchor_observations),
            label_future_observations=(
                None
                if batch.label_future_observations is None
                else self.encoder_observations(batch.label_future_observations)
            ),
        )


    def _loss(
        self,
        params: Any,
        log_dual: jax.Array,
        state: SkillState,
        batch: SkillBatch,
        key: jax.Array,
        *,
        apply_phi_noise: bool = True,
    ) -> tuple[jax.Array, SkillMetrics]:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        accuracy = mi_lb = alignment = log_partition = zero
        valid = batch.future_valid.astype(jnp.bool_)
        method = self.config.method
        encoder_obs = batch.observations
        encoder_next = batch.next_observations
        encoder_future = batch.future_observations

        if apply_phi_noise and self.config.phi_noise_std > 0.0:
            obs_key, next_key, future_key, _adj_key, key = jax.random.split(key, 5)
            noise = jnp.asarray(self.config.phi_noise_std, encoder_obs.dtype)
            encoder_obs = encoder_obs + noise * jax.random.normal(obs_key, encoder_obs.shape)
            encoder_next = encoder_next + noise * jax.random.normal(next_key, encoder_next.shape)
            encoder_future = encoder_future + noise * jax.random.normal(
                future_key, encoder_future.shape
            )
        if method == "potential":
            delta = self.displacement(params, encoder_obs, encoder_future)
            encoder_labels = batch.skills
            if self.config.encoder_skill_source == "achieved_discounted":
                if batch.label_future_observations is None:
                    raise ValueError("encoder_skill_source='achieved_discounted' needs "
                                     "label_future_observations on the encoder batch")
                # HER label for the encoder: unit(G), G = mean_m phi(s_{t+k_m}) - phi(s_t),
                # stop-grad (the label is a target, not a path for gradients), no phi noise.
                lf = batch.label_future_observations
                rows, samples = lf.shape[0], lf.shape[1]
                anchors_rep = jnp.repeat(batch.observations, samples, axis=0)
                d_lf = self.displacement(params, anchors_rep, lf.reshape((rows * samples,) + lf.shape[2:]))
                g = jax.lax.stop_gradient(jnp.mean(d_lf.reshape((rows, samples) + d_lf.shape[1:]), axis=1))
                g_norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(g), axis=-1), 0.0))
                skill_on = jnp.linalg.norm(batch.skills, axis=-1) > 1e-6
                g_valid = skill_on & (g_norm > 1e-6) & jnp.all(jnp.isfinite(g), axis=-1)
                encoder_labels = jnp.where(
                    g_valid[:, None], g / jnp.maximum(g_norm[:, None], 1e-6), batch.skills
                ).astype(batch.skills.dtype)
            if self.config.encoder_objective == "vmf_forward":
                loss, diagnostics = vmf_forward_loss(
                    delta, encoder_labels,
                    temperature=self.config.temperature, valid=valid,
                )
                accuracy = in_batch_skill_accuracy(
                    delta, encoder_labels,
                    temperature=self.config.temperature, valid=valid,
                )
                mi_lb = -loss
                alignment = diagnostics["alignment"]
                log_partition = diagnostics["log_partition"]
            else:
                nce = forward_infonce(
                    delta, encoder_labels,
                    temperature=self.config.temperature, valid=valid,
                )
                loss, accuracy, mi_lb = nce.loss, nce.accuracy, nce.mi_lower_bound
                count = jnp.maximum(jnp.sum(valid.astype(delta.dtype)), 1.0)
                positive = jnp.sum(delta * encoder_labels, axis=-1) / self.config.temperature
                alignment = jnp.sum(jnp.where(valid, positive, 0.0)) / count
                # Empirical log-mean-exp, including each row's positive candidate.
                log_partition = alignment - mi_lb
        rewards = self._intrinsic_impl(state, batch, key)
        metrics = SkillMetrics(
            loss=loss,
            intrinsic_reward_mean=jnp.mean(rewards),
            accuracy=accuracy,
            mi_lower_bound=mi_lb,
            alignment=alignment,
            log_partition=log_partition,
        )
        return loss, metrics

    def _update_impl(
        self,
        state: SkillState,
        batch: SkillBatch,
        key: jax.Array,
    ) -> tuple[SkillState, SkillMetrics]:
        def objective(params: Any, log_dual: jax.Array) -> tuple[jax.Array, SkillMetrics]:
            return self._loss(params, log_dual, state, batch, key)

        (loss, metrics), (param_grads, dual_grad) = jax.value_and_grad(
            objective,
            argnums=(0, 1),
            has_aux=True,
        )(state.params, state.log_dual)
        del loss
        updates, optimizer_state = self.optimizer.update(param_grads, state.optimizer_state, state.params)
        params = optax.apply_updates(state.params, updates)
        dual_updates, dual_optimizer_state = self.dual_optimizer.update(
            dual_grad,
            state.dual_optimizer_state,
            state.log_dual,
        )
        log_dual = optax.apply_updates(state.log_dual, dual_updates)
        target_params = params if self.ema_tau_one_shortcut and self.config.target_tau == 1.0 else jax.tree_util.tree_map(
            lambda target, online: (1.0 - self.config.target_tau) * target + self.config.target_tau * online,
            state.target_params,
            params,
        )
        input_mean, input_var = state.input_mean, state.input_var
        target_mean, target_var = state.target_mean, state.target_var
        return SkillState(
            params=params,
            target_params=target_params,
            optimizer_state=optimizer_state,
            log_dual=log_dual,
            dual_optimizer_state=dual_optimizer_state,
            input_mean=input_mean,
            input_var=input_var,
            target_mean=target_mean,
            target_var=target_var,
            step=state.step + 1,
        ), metrics

    def _intrinsic_impl(
        self,
        state: SkillState,
        batch: SkillBatch,
        key: jax.Array,
        partition_weight: jax.Array | float = 1.0,
    ) -> jax.Array:
        encoder_obs = batch.observations
        encoder_next = batch.next_observations
        reward_params = state.params if self.config.reward_encoder == "live" else state.target_params
        reference_obs = (
            batch.anchor_observations
            if self.config.potential_reward_reference == "anchor"
            else encoder_obs
        )
        delta = self.displacement(reward_params, reference_obs, encoder_next)
        reward = jnp.sum(delta * batch.skills, axis=-1)   # <displacement, z>; == potential_reward(phi_ref, phi_next, z) for the difference form
        if self.config.continuous_reward_temperature:
            reward = reward / jnp.asarray(self.config.temperature, reward.dtype)
        if self.config.continuous_reward_normalization == "sphere":
            # Both endpoints use the SAME replay anchor and encoder copy. Normalizing
            # a one-step displacement instead would define a different objective.
            anchor_next = self.displacement(reward_params, batch.anchor_observations, encoder_next)
            correction = sphere_log_partition(anchor_next, self.config.temperature)
            if self.config.potential_reward_reference == "current":
                anchor_current = self.displacement(reward_params, batch.anchor_observations, encoder_obs)
                correction = correction - sphere_log_partition(anchor_current, self.config.temperature)
            reward = reward - partition_weight * correction
        if self.config.continuous_reward_normalization == "sampled_infonce":
            # Never use replay/HER labels as the prior. Include the positive
            # beside 1023 fresh uniform-sphere negatives (1024 candidates).
            negatives = self.sample_skills(key, 1023)
            correction = sampled_infonce_log_partition(
                delta, batch.skills, negatives, self.config.temperature,
            )
            reward = reward - partition_weight * correction
        return self.config.potential_reward_scale * reward

    def relabel_achieved_skills(
        self,
        state: SkillState,
        batch: Any,
        allowed: jax.Array,
        outcome_observations: jax.Array | None = None,
    ) -> tuple[Any, RelabelMetrics]:
        """Relabel allowed rows from their selected anchor to the supplied outcome."""

        dtype = batch.skills.dtype
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        if not self.config.achieved_skill_relabeling:
            return batch, RelabelMetrics(zero, zero, zero, zero, one, one)

        allowed = jnp.asarray(allowed, dtype=jnp.bool_).reshape((-1,))
        skill_on = jnp.linalg.norm(batch.skills, axis=-1) > 1e-6
        relabel_params = (
            state.target_params if self.config.relabel_encoder == "slow" else state.params
        )
        anchor_enc = self.encoder_observations(batch.anchor_observations)
        outcomes = batch.observations if outcome_observations is None else outcome_observations
        sampled = outcomes.ndim == batch.observations.ndim + 1
        if sampled:
            rows, samples = outcomes.shape[0], outcomes.shape[1]
            flat = outcomes.reshape((rows * samples,) + outcomes.shape[2:])
            sample_delta = self.displacement(
                relabel_params, jnp.repeat(anchor_enc, samples, axis=0), self.encoder_observations(flat)
            )
            displacement = jnp.mean(sample_delta.reshape((rows, samples) + sample_delta.shape[1:]), axis=1)
        else:
            displacement = self.displacement(relabel_params, anchor_enc, self.encoder_observations(outcomes))
        # Continuous (sym-CSF) achieved skill: the achieved skill is the UNIT DIRECTION of the representation displacement
        # anchor -> outcome (stop-grad); rows whose displacement is below the norm eps or
        # non-finite are not relabeled; confidence := displacement norm; class-balanced
        # weighting has no classes so weights == 1; "same skill" for next-row propagation
        # is vector equality.
        delta = jax.lax.stop_gradient(displacement)
        finite = jnp.all(jnp.isfinite(delta), axis=-1)
        delta = jnp.where(finite[:, None], delta, jnp.zeros_like(delta))
        norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(delta), axis=-1), 0.0))
        norm_eps = jnp.asarray(1e-6, dtype=norm.dtype)
        min_norm = jnp.asarray(max(self.config.achieved_skill_relabel_min_norm, 1e-6), dtype=norm.dtype)
        valid = (norm > norm_eps) & (norm >= min_norm) & finite
        achieved_skills = jnp.where(
            valid[:, None],
            delta / jnp.maximum(norm[:, None], norm_eps),
            jnp.zeros_like(delta),
        ).astype(dtype)
        confidence = norm.astype(dtype)
        # Reference: relabel_mask = skill_on & achieved_valid -- the discrete min-prob
        # confidence gate does NOT apply to continuous draws (norm-eps validity only).
        relabel_mask = allowed & skill_on & (batch.loss_weights > 0.0) & valid
        relabeled_skills = jnp.where(relabel_mask[:, None], achieved_skills, batch.skills)
        next_skill_on = jnp.linalg.norm(batch.next_skills, axis=-1) > 1e-6
        same_segment = next_skill_on & jnp.all(batch.next_skills == batch.skills, axis=-1)
        changed_flag = jnp.linalg.norm(achieved_skills - batch.skills, axis=-1) > 1e-4
        next_relabel_mask = relabel_mask & same_segment
        relabeled_next_skills = jnp.where(
            next_relabel_mask[:, None], achieved_skills, batch.next_skills
        )
        next_anchor_mask = next_relabel_mask.reshape(
            (next_relabel_mask.shape[0],)
            + (1,) * (batch.next_anchor_observations.ndim - 1)
        )
        relabeled_next_anchors = jnp.where(
            next_anchor_mask,
            batch.anchor_observations,
            batch.next_anchor_observations,
        )

        relabel_float = relabel_mask.astype(dtype)
        relabel_count = jnp.sum(relabel_float)
        # Class-balanced weighting has no classes for continuous skills, so every weight is 1.
        weights = jnp.ones_like(batch.loss_weights, dtype=dtype)
        loss_weights = batch.loss_weights * weights
        relabeled = batch.replace(
            skills=relabeled_skills,
            next_skills=relabeled_next_skills,
            next_anchor_observations=relabeled_next_anchors,
            loss_weights=loss_weights,
        )

        denom = jnp.maximum(jnp.sum(skill_on.astype(dtype)), one)
        changed = relabel_mask & changed_flag
        weight_denom = jnp.maximum(relabel_count, one)
        metrics = RelabelMetrics(
            allowed_fraction=jnp.mean(allowed.astype(dtype)),
            relabeled_fraction=jnp.sum(relabel_float) / denom,
            changed_fraction=jnp.sum(changed.astype(dtype)) / denom,
            confidence_mean=jnp.sum(jnp.where(relabel_mask, confidence, zero)) / weight_denom,
            weight_mean=jnp.sum(jnp.where(relabel_mask, weights, zero)) / weight_denom,
            weight_max=jnp.where(
                relabel_count > 0,
                jnp.max(jnp.where(relabel_mask, weights, zero)),
                one,
            ),
        )
        return relabeled, metrics

    def update(self, state: SkillState, batch: SkillBatch, key: jax.Array) -> tuple[SkillState, SkillMetrics]:
        return self.update_preprocessed(state, self.preprocess_batch(batch), key)

    def update_preprocessed(
        self,
        state: SkillState,
        batch: SkillBatch,
        key: jax.Array,
    ) -> tuple[SkillState, SkillMetrics]:
        return self._update_jit(state, batch, key)

    def evaluate_preprocessed(
        self,
        state: SkillState,
        batch: SkillBatch,
        key: jax.Array,
        *,
        apply_phi_noise: bool = True,
    ) -> SkillMetrics:
        """Score an encoder-view batch under the current params without updating.

        The default matches the training loss. Callers may disable only the
        encoder input noise to obtain a same-batch no-augmentation companion.
        Left untraced so it can be called from inside an enclosing jit.
        """

        return self._loss(
            state.params,
            state.log_dual,
            state,
            batch,
            key,
            apply_phi_noise=apply_phi_noise,
        )[1]

    def vmf_reward_terms(self, state: SkillState, batch: SkillBatch) -> tuple[jax.Array, jax.Array]:
        """Clean executed-command logit and log partition, both in nats."""
        if self.config.encoder_objective != "vmf_forward":
            raise ValueError("vMF reward terms require the vMF decoder")
        batch = self.preprocess_batch(batch)
        params = state.params if self.config.reward_encoder == "live" else state.target_params
        delta = self.displacement(params, batch.anchor_observations, batch.next_observations)
        positive = jnp.sum(delta * batch.skills, axis=-1) / self.config.temperature
        partition = sphere_log_partition(delta, self.config.temperature)
        return jax.lax.stop_gradient(positive), jax.lax.stop_gradient(partition)

    def executed_information(self, state: SkillState, batch: SkillBatch) -> jax.Array:
        """Signed log q(executed skill | anchor, successor) / prior; not posterior KL."""
        positive, partition = self.vmf_reward_terms(state, batch)
        return positive - partition

    def posterior_information(self, state: SkillState, batch: SkillBatch) -> jax.Array:
        """Clean, unscaled posterior KL for the supplied physical anchor/outcome."""
        if self.config.encoder_objective != "vmf_forward":
            raise ValueError("posterior information requires the vMF decoder")
        batch = self.preprocess_batch(batch)
        params = state.params if self.config.reward_encoder == "live" else state.target_params
        delta = self.displacement(params, batch.anchor_observations, batch.next_observations)
        return jax.lax.stop_gradient(vmf_posterior_kl(delta, self.config.temperature))

    def intrinsic_reward(
        self, state: SkillState, batch: SkillBatch, key: jax.Array,
        partition_weight: jax.Array | float = 1.0,
    ) -> jax.Array:
        return self._intrinsic_jit(state, self.preprocess_batch(batch), key, partition_weight)
