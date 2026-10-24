"""Flax network modules: every ``nn.Module`` used by the SEE agents.

``default_init``/``MLP`` (encoder and critic heads), the SAC actor/critic modules
and the recurrent PQN modules live here; class names, attribute
names and explicit ``name=`` arguments fix the checkpoint parameter paths.

``BatchRenorm`` follows PureJaxQL, so the recurrent PQN controller uses PureJaxQL's
normalization.

The VIM, LatentGCE and PPO baseline modules (``VectorMLP``, ``VectorLatentGCE``,
``AutoregressiveGaussian``, ``SourceValueNetwork``, ``DecoderNetwork``,
``ActorCritic`` and their
``BaselineMLP``/``baseline_default_init``) follow at the end of the file.
"""

from functools import partial
import math
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import distrax
from flax import linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from jax.nn import initializers
import numpy as np

# Leaf modules (stdlib/JAX only): the baseline modules' config fields and default.
from empowerment_warp.training.baselines.config import EnvironmentConfig, EstimatorConfig, VIMModelConfig


PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any
Axes = Union[int, Sequence[int]]


class BatchRenorm(Module):
    """Flax BatchRenorm matching the PureJaxQL recurrent Craftax controller."""

    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.999
    epsilon: float = 0.001
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        use_running_average = merge_param(
            'use_running_average', self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[axis] for axis in feature_axes]

        running_mean = self.variable(
            'batch_stats', 'mean', lambda shape: jnp.zeros(shape, jnp.float32), feature_shape
        )
        running_var = self.variable(
            'batch_stats', 'var', lambda shape: jnp.ones(shape, jnp.float32), feature_shape
        )
        r_max = self.variable('batch_stats', 'r_max', lambda value: value, 3)
        d_max = self.variable('batch_stats', 'd_max', lambda value: value, 5)
        steps = self.variable('batch_stats', 'steps', lambda value: value, 0)

        if use_running_average:
            mean, var = running_mean.value, running_var.value
            custom_mean, custom_var = mean, var
        else:
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean, custom_var = mean, var
            if not self.is_initializing():
                std = jnp.sqrt(var + self.epsilon)
                running_std = jnp.sqrt(running_var.value + self.epsilon)
                r = jnp.clip(jax.lax.stop_gradient(std / running_std), 1 / r_max.value, r_max.value)
                d = jnp.clip(
                    jax.lax.stop_gradient((mean - running_mean.value) / running_std),
                    -d_max.value,
                    d_max.value,
                )
                renorm_var = var / (r**2)
                renorm_mean = mean - d * jnp.sqrt(custom_var) / r
                warmed_up = jnp.greater_equal(steps.value, 1000).astype(jnp.float32)
                custom_var = warmed_up * renorm_var + (1.0 - warmed_up) * custom_var
                custom_mean = warmed_up * renorm_mean + (1.0 - warmed_up) * custom_mean
                running_mean.value = (
                    self.momentum * running_mean.value + (1 - self.momentum) * mean
                )
                running_var.value = (
                    self.momentum * running_var.value + (1 - self.momentum) * var
                )
                steps.value += 1

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )


def default_init(scale: float = 1.0):
    """Match empowerment.utils.networks.default_init."""

    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: tuple[int, ...]
    output_dim: int
    layer_norm: bool
    final_scale: float = 1.0

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        for width in self.hidden_dims:
            value = nn.Dense(width, kernel_init=default_init())(value)
            value = nn.gelu(value)
            if self.layer_norm:
                value = nn.LayerNorm()(value)
        return nn.Dense(self.output_dim, kernel_init=default_init(self.final_scale))(value)


class SkillMeanAdapter(nn.Module):
    skill_dim: int
    action_dim: int
    init_std: float

    @nn.compact
    def __call__(self, skills: jax.Array) -> jax.Array:
        kernel = self.param(
            "kernel",
            nn.initializers.normal(stddev=self.init_std),
            (self.skill_dim, self.action_dim),
        )
        return skills @ (kernel - kernel.mean(axis=0, keepdims=True))


class GaussianActor(nn.Module):
    hidden_dims: tuple[int, ...]
    action_dim: int
    layer_norm: bool
    final_scale: float
    skill_dim: int
    skill_mean_init_std: float
    state_dependent_std: bool
    use_anchor: bool

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        anchor_observations: jax.Array,
        skills: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        parts = (observations, anchor_observations, skills) if self.use_anchor else (observations, skills)
        value = jnp.concatenate(parts, axis=-1)
        for width in self.hidden_dims:
            value = nn.Dense(width, kernel_init=default_init())(value)
            value = nn.gelu(value)
            if self.layer_norm:
                value = nn.LayerNorm()(value)
        mean = nn.Dense(
            self.action_dim,
            kernel_init=default_init(self.final_scale),
            name="mean",
        )(value)
        if self.skill_mean_init_std > 0.0:
            mean = mean + SkillMeanAdapter(
                self.skill_dim,
                self.action_dim,
                self.skill_mean_init_std,
                name="skill_mean_adapter",
            )(skills)
        if self.state_dependent_std:
            log_std = nn.Dense(
                self.action_dim,
                kernel_init=default_init(self.final_scale),
                name="log_std",
            )(value)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
            log_std = jnp.broadcast_to(log_stds, mean.shape)
        return mean, jnp.clip(log_std, -5.0, 2.0)


class TwinContinuousCritic(nn.Module):
    hidden_dims: tuple[int, ...]
    layer_norm: bool
    final_scale: float
    use_anchor: bool

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
        anchor_observations: jax.Array,
        skills: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        parts = (
            (observations, anchor_observations, actions, skills)
            if self.use_anchor
            else (observations, actions, skills)
        )
        value = jnp.concatenate(parts, axis=-1)
        q1 = MLP(self.hidden_dims, 1, self.layer_norm, self.final_scale, name="q1")(value)[..., 0]
        q2 = MLP(self.hidden_dims, 1, self.layer_norm, self.final_scale, name="q2")(value)[..., 0]
        return jnp.stack((q1, q2), axis=-1)


class PotentialActor(nn.Module):
    """Skill-free actor for z = 0 rows (checkpoint field ``skill0_actor_params``)."""

    hidden_dims: tuple[int, ...]
    action_dim: int
    layer_norm: bool
    final_scale: float
    state_dependent_std: bool

    @nn.compact
    def __call__(self, observations: jax.Array) -> tuple[jax.Array, jax.Array]:
        value = observations
        for width in self.hidden_dims:
            value = nn.Dense(width, kernel_init=default_init())(value)
            value = nn.gelu(value)
            if self.layer_norm:
                value = nn.LayerNorm()(value)
        mean = nn.Dense(
            self.action_dim,
            kernel_init=default_init(self.final_scale),
            name="mean",
        )(value)
        if self.state_dependent_std:
            log_std = nn.Dense(
                self.action_dim,
                kernel_init=default_init(self.final_scale),
                name="log_std",
            )(value)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
            log_std = jnp.broadcast_to(log_stds, mean.shape)
        return mean, jnp.clip(log_std, -5.0, 2.0)


class TwinPotentialCritic(nn.Module):
    """Skill-free twin Q for z = 0 rows (checkpoint field ``skill0_critic_params``)."""

    hidden_dims: tuple[int, ...]
    layer_norm: bool
    final_scale: float

    @nn.compact
    def __call__(self, observations: jax.Array, actions: jax.Array) -> jax.Array:
        value = jnp.concatenate((observations, actions), axis=-1)
        q1 = MLP(self.hidden_dims, 1, self.layer_norm, self.final_scale, name="q1")(value)[..., 0]
        q2 = MLP(self.hidden_dims, 1, self.layer_norm, self.final_scale, name="q2")(value)[..., 0]
        return jnp.stack((q1, q2), axis=-1)


class ScannedLSTM(nn.Module):
    """LSTM scanned over time with an explicit reset mask."""

    @partial(
        nn.scan,
        variable_broadcast='params',
        in_axes=0,
        out_axes=0,
        split_rngs={'params': False},
    )
    @nn.compact
    def __call__(self, carry, inputs):
        embeddings, resets = inputs
        hidden_size = carry[0].shape[-1]
        zero_carry = self.initialize_carry(hidden_size, *resets.shape)
        carry = jax.tree_util.tree_map(
            lambda zero, old: jnp.where(resets[..., None], zero, old),
            zero_carry,
            carry,
        )
        new_carry, output = nn.OptimizedLSTMCell(hidden_size)(carry, embeddings)
        # Potential-value probes recover each pre-step carry from these outputs.
        return new_carry, (output, new_carry)

    @staticmethod
    def initialize_carry(hidden_size, *batch_shape):
        return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_shape, hidden_size)
        )


class SkillLogitAdapter(nn.Module):
    """Centered, state-independent skill bias on the action logits / Q-values.

    Port of the reference `SkillLogitAdapter` (empowerment utils/networks.py): a learnable
    kernel K ~ N(0, init_std^2) of shape [skill_dim, action_dim], double-centered
    (C = K - row mean - column mean + grand mean) so the bias has zero mean over skills
    and over actions, applied as z^T C.  At init this is the policy-side SYMMETRY BREAKING:
    a random, skill-dependent action preference so that behaviour differs across z before
    any learning, giving the encoder something to fit.  z = 0 (potential rows) gets 0.
    """

    skill_dim: int
    action_dim: int
    init_std: float

    @nn.compact
    def __call__(self, skills):
        kernel = self.param(
            'kernel', nn.initializers.normal(stddev=self.init_std), (self.skill_dim, self.action_dim)
        )
        centered = (
            kernel
            - kernel.mean(axis=0, keepdims=True)
            - kernel.mean(axis=1, keepdims=True)
            + kernel.mean()
        )
        return skills @ centered


class RecurrentQNetwork(nn.Module):
    """PureJaxQL recurrent Q-network: normalized MLP, last action, and LSTM."""

    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    num_rnn_layers: int = 1
    norm_input: bool = False
    add_last_action: bool = True
    reset_last_action: bool = False
    recurrent_context_dim: int = 0
    # Skill-logit adapter (symmetry breaking): reads the trailing `skill_dim` dims of the
    # raw input (the effective input is [obs, anchor, skills]) and adds z^T C to the
    # Q-values. 0.0 = off = byte-identical (no parameter is created).
    skill_dim: int = 0
    skill_logit_init_std: float = 0.0
    # potential-PPO: append one state-value column after the `action_dim` policy logits
    # (Dense(1) on the same recurrent features). False = PQN, byte-identical (no parameter).
    value_head: bool = False

    @nn.compact
    def __call__(self, hidden, observations, resets, last_actions, train=False, return_carries=False):
        """Returns ``(next_hidden, outputs)``; with ``return_carries`` also the list (per RNN
        layer) of post-step carries stacked over time, ``[(c, h)]`` each ``[T, B, hidden]``."""
        raw_skills = None
        if self.skill_logit_init_std > 0.0:
            if self.skill_dim <= 0 or self.skill_dim >= observations.shape[-1]:
                raise ValueError('skill_logit_init_std > 0 needs 0 < skill_dim < input width')
            raw_skills = observations[..., -self.skill_dim:]
        context = None
        if self.recurrent_context_dim:
            if self.recurrent_context_dim >= observations.shape[-1]:
                raise ValueError(
                    'recurrent_context_dim must leave at least one recurrent input.'
                )
            observations, context = jnp.split(
                observations,
                [observations.shape[-1] - self.recurrent_context_dim],
                axis=-1,
            )

        if self.norm_input and context is None:
            observations = BatchRenorm(use_running_average=not train)(observations)
        elif self.norm_input:
            observations = BatchRenorm(
                use_running_average=not train, name='recurrent_input_norm'
            )(observations)
            context = BatchRenorm(
                use_running_average=not train, name='context_input_norm'
            )(context)
        else:
            # The reference creates this state even when input normalization is off.
            _ = BatchRenorm(use_running_average=not train)(observations)
            if context is not None:
                _ = BatchRenorm(
                    use_running_average=not train, name='context_input_norm'
                )(context)

        x = observations
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)

        if self.add_last_action:
            if self.reset_last_action:
                last_actions = jnp.where(resets, 0, last_actions)
            previous = jax.nn.one_hot(last_actions, self.action_dim)
            x = jnp.concatenate([x, previous], axis=-1)

        next_hidden = []
        carries = []
        for layer_index in range(self.num_rnn_layers):
            hidden_layer, (x, step_carries) = ScannedLSTM(name=f'lstm_{layer_index}')(
                hidden[layer_index], (x, resets)
            )
            next_hidden.append(hidden_layer)
            carries.append(step_carries)
        if context is not None:
            # relabelable context affects Q without entering the recurrent state
            context = nn.Dense(self.hidden_size, name='context_projection')(context)
            x = nn.LayerNorm(name='context_fusion_norm')(x + context)
            x = nn.relu(x)
        q_values = nn.Dense(self.action_dim, name='q_values')(x)
        if raw_skills is not None:
            q_values = q_values + SkillLogitAdapter(
                self.skill_dim, self.action_dim, self.skill_logit_init_std, name='skill_logit_adapter'
            )(raw_skills)
        if self.value_head:
            # [.., action_dim] logits (adapter-biased) followed by one value column.
            q_values = jnp.concatenate([q_values, nn.Dense(1, name='value')(x)], axis=-1)
        if return_carries:
            return next_hidden, q_values, carries
        return next_hidden, q_values

    def initialize_carry(self, batch_size):
        return [
            ScannedLSTM.initialize_carry(self.hidden_size, batch_size)
            for _ in range(self.num_rnn_layers)
        ]

# ---------------------------------------------------------------------------
# VIM / LatentGCE / PPO baseline modules (extracted from empowerment; see
# training/baselines/reference_manifest.json and LICENSE). The extracted
# MLP/default_init are renamed BaselineMLP/baseline_default_init to avoid the
# SEE MLP above; VIM attaches them as setup attributes (backbone,
# value_network), so parameter paths do not contain the class name.
# ---------------------------------------------------------------------------

def baseline_default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')

class BaselineMLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = baseline_default_init()
    final_kernel_init: Any = None
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            kernel_init = (
                self.final_kernel_init
                if i == len(self.hidden_dims) - 1 and self.final_kernel_init is not None
                else self.kernel_init
            )
            x = nn.Dense(size, kernel_init=kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class VectorMLP(nn.Module):
    width: int
    depth: int
    output_dim: int

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for _ in range(self.depth):
            x = nn.relu(nn.Dense(self.width, kernel_init=nn.initializers.glorot_uniform())(x))
        return nn.Dense(self.output_dim, kernel_init=nn.initializers.glorot_uniform())(x)


class VectorLatentGCE(nn.Module):
    """Encode vector states and action sequences before the linear latent channel.

    Ant shares its start/future state encoder. AntSoccer retains the existing
    ball-xy outcome, so its 42D context and 2D outcome need distinct encoders.
    """

    observation_dim: int
    target_dim: int
    action_sequence_dim: int
    latent_dim: int = 32
    hidden_dim: int = 512
    hidden_layers: int = 3
    bound_action_decoder: bool = False

    def setup(self):
        self.context_encoder = VectorMLP(self.hidden_dim, self.hidden_layers, self.latent_dim)
        self.context_decoder = VectorMLP(self.hidden_dim, 2, self.observation_dim)
        if self.observation_dim == self.target_dim:
            self.target_encoder = self.context_encoder
            self.target_decoder = self.context_decoder
        else:
            self.target_encoder = VectorMLP(self.hidden_dim, self.hidden_layers, self.latent_dim)
            self.target_decoder = VectorMLP(self.hidden_dim, 2, self.target_dim)
        self.action_encoder = VectorMLP(self.hidden_dim, self.hidden_layers, self.latent_dim)
        self.action_decoder = VectorMLP(self.hidden_dim, 2, self.action_sequence_dim)
        self.matrix_network = VectorMLP(self.hidden_dim, self.hidden_layers,
                                        self.latent_dim * self.latent_dim)

    def channel(self, observations):
        context = self.context_encoder(observations)
        matrix = self.matrix_network(context).reshape(
            observations.shape[:-1] + (self.latent_dim, self.latent_dim))
        return context, matrix

    def decode_action(self, latent_action):
        action = self.action_decoder(latent_action)
        return jnp.tanh(action) if self.bound_action_decoder else action

    def predict_target(self, observations, actions):
        """Decode the channel's predicted outcome back into the target space."""
        _, matrix = self.channel(observations)
        action = self.action_encoder(actions.reshape((actions.shape[0], -1)))
        return self.target_decoder(jnp.einsum('...ij,...j->...i', matrix, action))

    def __call__(self, observations, actions, targets):
        context, matrix = self.channel(observations)
        target = self.target_encoder(targets)
        action = self.action_encoder(actions.reshape((actions.shape[0], -1)))
        prediction = jnp.einsum('...ij,...j->...i', matrix, action)
        return dict(
            latent_prediction=jnp.mean(jnp.square(prediction - target), axis=-1),
            observation_reconstruction=jnp.mean(jnp.square(
                self.context_decoder(context) - observations), axis=-1),
            target_reconstruction=jnp.mean(jnp.square(
                self.target_decoder(target) - targets), axis=-1)
                if self.observation_dim != self.target_dim else jnp.zeros(observations.shape[0]),
            prediction_reconstruction=jnp.mean(jnp.square(
                self.target_decoder(prediction) - targets), axis=-1),
            action_reconstruction=jnp.mean(jnp.square(
                self.decode_action(action) - actions.reshape((actions.shape[0], -1))), axis=-1),
            context_squared=jnp.mean(jnp.square(context), axis=-1),
            target_squared=jnp.mean(jnp.square(target), axis=-1),
            action_squared=jnp.mean(jnp.square(action), axis=-1),
        )


def _tanh_log_prob(base: distrax.Distribution, pre_tanh_actions: jax.Array) -> jax.Array:
    """Evaluate the action density without inverting rounded tanh outputs."""

    log_jacobian = 2.0 * (
        jnp.log(2.0) - pre_tanh_actions - jax.nn.softplus(-2.0 * pre_tanh_actions)
    )
    return base.log_prob(pre_tanh_actions) - jnp.sum(log_jacobian, axis=-1)


class AutoregressiveGaussian(nn.Module):
    """Tanh-Gaussian sequence density conditioned on a fixed context."""

    horizon: int
    action_dim: int
    hidden_dim: int
    hidden_layers: int
    log_std_min: float
    log_std_max: float
    initial_std: float | None = None

    def setup(self):
        self.backbone = BaselineMLP(
            (self.hidden_dim,) * self.hidden_layers,
            activations=nn.relu,
            activate_final=True,
        )
        self.mean_head = nn.Dense(
            self.action_dim,
            kernel_init=(baseline_default_init(1e-2) if self.initial_std is None
                         else nn.initializers.zeros_init()),
            bias_init=nn.initializers.zeros_init(),
        )
        self.log_std_head = nn.Dense(
            self.action_dim,
            kernel_init=(baseline_default_init(1e-2) if self.initial_std is None
                         else nn.initializers.zeros_init()),
            bias_init=nn.initializers.constant(
                0.0 if self.initial_std is None else math.log(self.initial_std)
            ),
        )

    def _distribution(
        self,
        context: jax.Array,
        action_prefix: jax.Array,
        step_index: int,
    ) -> distrax.Distribution:
        prefix_mask = (
            jnp.arange(self.horizon, dtype=jnp.int32) < step_index
        )[None, :, None]
        masked_prefix = jnp.where(prefix_mask, action_prefix, 0.0)
        timestep = jax.nn.one_hot(
            jnp.full(context.shape[:-1], step_index, dtype=jnp.int32),
            self.horizon,
        )
        inputs = jnp.concatenate(
            (context, masked_prefix.reshape((context.shape[0], -1)), timestep),
            axis=-1,
        )
        features = self.backbone(inputs)
        means = self.mean_head(features)
        log_stds = jnp.clip(self.log_std_head(features), self.log_std_min, self.log_std_max)
        return distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))

    def __call__(
        self, context: jax.Array, actions: jax.Array, pre_tanh_actions: jax.Array
    ) -> jax.Array:
        """Return ``log p(u[1:K] | context)`` in natural-log units."""

        def step(module, prefix, step_index):
            distribution = module._distribution(context, prefix, step_index)
            action = actions[:, step_index, :]
            log_probability = _tanh_log_prob(distribution, pre_tanh_actions[:, step_index, :])
            prefix = prefix.at[:, step_index, :].set(action)
            return prefix, log_probability

        scan = nn.scan(step, variable_broadcast='params', split_rngs={'params': False}, out_axes=1)
        _, log_probabilities = scan(self, jnp.zeros_like(actions), jnp.arange(self.horizon))
        return jnp.sum(log_probabilities, axis=1)

    def sample(self, context: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample one normalized open-loop sequence per context row."""

        actions = jnp.zeros((context.shape[0], self.horizon, self.action_dim), dtype=context.dtype)
        pre_tanh_actions = jnp.zeros_like(actions)
        def step(module, carry, step_index):
            actions, pre_tanh_actions, key = carry
            key, sample_key = jax.random.split(key)
            distribution = module._distribution(context, actions, step_index)
            pre_tanh_action = distribution.sample(seed=sample_key)
            action = jnp.tanh(pre_tanh_action)
            log_probability = _tanh_log_prob(distribution, pre_tanh_action)
            actions = actions.at[:, step_index, :].set(action)
            pre_tanh_actions = pre_tanh_actions.at[:, step_index, :].set(pre_tanh_action)
            return (actions, pre_tanh_actions, key), log_probability

        scan = nn.scan(step, variable_broadcast='params', split_rngs={'params': False}, out_axes=1)
        (actions, pre_tanh_actions, _), log_probabilities = scan(
            self, (actions, pre_tanh_actions, key), jnp.arange(self.horizon))
        return actions, pre_tanh_actions, jnp.sum(log_probabilities, axis=1)


class SourceValueNetwork(nn.Module):
    """VIM source distribution and independent scalar ``psi(s)`` branch."""

    environment: EnvironmentConfig
    config: EstimatorConfig
    model_config: VIMModelConfig = VIMModelConfig()

    def setup(self):
        self.source = AutoregressiveGaussian(
            horizon=self.environment.horizon,
            action_dim=self.environment.action_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_layers=self.config.hidden_layers,
            log_std_min=self.model_config.log_std_min,
            log_std_max=self.model_config.log_std_max,
            initial_std=self.model_config.source_initial_std,
        )
        self.value_network = BaselineMLP(
            (self.config.scalar_hidden_dim,) * self.config.hidden_layers + (1,),
            activations=nn.relu,
        )

    def __call__(
        self, observations: jax.Array, actions: jax.Array, pre_tanh_actions: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        log_h = self.source(observations, actions, pre_tanh_actions)
        psi = jnp.squeeze(self.value_network(observations), axis=-1)
        return log_h, psi

    def sample(self, observations: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return self.source.sample(observations, key)

    def empowerment(self, observations: jax.Array, beta: float) -> jax.Array:
        psi = jnp.squeeze(self.value_network(observations), axis=-1)
        return psi / beta


class DecoderNetwork(nn.Module):
    """VIM action-sequence decoder ``q(u[1:K] | s, s')``."""

    environment: EnvironmentConfig
    config: EstimatorConfig
    model_config: VIMModelConfig = VIMModelConfig()

    def setup(self):
        self.decoder = AutoregressiveGaussian(
            horizon=self.environment.horizon,
            action_dim=self.environment.action_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_layers=self.config.hidden_layers,
            log_std_min=self.model_config.log_std_min,
            log_std_max=self.model_config.log_std_max,
        )

    def __call__(
        self,
        observations: jax.Array,
        targets: jax.Array,
        actions: jax.Array,
        pre_tanh_actions: jax.Array,
    ) -> jax.Array:
        context = jnp.concatenate((observations, targets), axis=-1)
        return self.decoder(context, actions, pre_tanh_actions)


class ActorCritic(nn.Module):
    """Separate 2x256 networks; actor returns the pre-tanh Gaussian."""

    action_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 2

    @nn.compact
    def __call__(self, observations: jax.Array):
        actor = observations
        for _ in range(self.hidden_layers):
            actor = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2.0)),
                    bias_init=constant(0.0),
                )(actor)
            )
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        log_std = self.param('log_std', nn.initializers.zeros_init(), (self.action_dim,))

        critic = observations
        for _ in range(self.hidden_layers):
            critic = nn.tanh(
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=orthogonal(np.sqrt(2.0)),
                    bias_init=constant(0.0),
                )(critic)
            )
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)
        return distrax.MultivariateNormalDiag(actor_mean, jnp.exp(log_std)), jnp.squeeze(value, axis=-1)
