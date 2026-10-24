"""State-only variational information maximization and probe replay."""

from __future__ import annotations

import math
from functools import partial

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from empowerment_warp.baselines.toy_baselines.config import EnvironmentConfig
from empowerment_warp.baselines.toy_baselines.ppo import ControllerState, RolloutBatch
from empowerment_warp.baselines.toy_baselines.config import EstimatorConfig, VIMModelConfig, optimizer_kwargs
from empowerment_warp.baselines.toy_baselines.networks import MLP, default_init


from empowerment_warp.baselines.toy_baselines.probes import uniform_actions
from empowerment_warp.baselines.toy_baselines.replay import masked_mean as _masked_mean


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
        self.backbone = MLP(
            (self.hidden_dim,) * self.hidden_layers,
            activations=nn.relu,
            activate_final=True,
        )
        self.mean_head = nn.Dense(
            self.action_dim,
            kernel_init=(default_init(1e-2) if self.initial_std is None
                         else nn.initializers.zeros_init()),
            bias_init=nn.initializers.zeros_init(),
        )
        self.log_std_head = nn.Dense(
            self.action_dim,
            kernel_init=(default_init(1e-2) if self.initial_std is None
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

        prefix = jnp.zeros_like(actions)
        log_probabilities = []
        for step_index in range(self.horizon):
            distribution = self._distribution(context, prefix, step_index)
            action = actions[:, step_index, :]
            log_probabilities.append(_tanh_log_prob(distribution, pre_tanh_actions[:, step_index, :]))
            prefix = prefix.at[:, step_index, :].set(action)
        return jnp.sum(jnp.stack(log_probabilities, axis=1), axis=1)

    def sample(self, context: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample one normalized open-loop sequence per context row."""

        actions = jnp.zeros((context.shape[0], self.horizon, self.action_dim), dtype=context.dtype)
        pre_tanh_actions = jnp.zeros_like(actions)
        log_probabilities = []
        for step_index in range(self.horizon):
            key, sample_key = jax.random.split(key)
            distribution = self._distribution(context, actions, step_index)
            pre_tanh_action = distribution.sample(seed=sample_key)
            action = jnp.tanh(pre_tanh_action)
            log_probabilities.append(_tanh_log_prob(distribution, pre_tanh_action))
            actions = actions.at[:, step_index, :].set(action)
            pre_tanh_actions = pre_tanh_actions.at[:, step_index, :].set(pre_tanh_action)
        return actions, pre_tanh_actions, jnp.sum(jnp.stack(log_probabilities, axis=1), axis=1)


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
        self.value_network = MLP(
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



class FixedUniformSourceValue(SourceValueNetwork):
    """Exact uniform action density; initialize/train only the scalar branch."""
    def __call__(self, observations, actions, pre_tanh_actions):
        inside = jnp.all(jnp.abs(actions) <= 1., axis=(1, 2))
        log_h = -self.environment.horizon * self.environment.action_dim * jnp.log(2.)
        return jnp.where(inside, log_h, -jnp.inf), self.empowerment(observations, 1.)

    def sample(self, observations, key):
        shape = (len(observations), self.environment.horizon, self.environment.action_dim)
        actions = uniform_actions(key, shape, observations.dtype)
        return actions, jnp.arctanh(actions), jnp.full(len(observations), -shape[1] * shape[2] * jnp.log(2.))


def create_vim_train_states(
    source_key: jax.Array,
    decoder_key: jax.Array,
    environment: EnvironmentConfig,
    config: EstimatorConfig,
    model_config: VIMModelConfig = VIMModelConfig(),
    *,
    fixed_uniform: bool = False,
) -> tuple[SourceValueNetwork, TrainState, DecoderNetwork, TrainState]:
    """Initialize VIM networks and independent states for the selected optimizer."""

    observations = jnp.zeros((1, environment.observation_dim), dtype=jnp.float32)
    actions = jnp.zeros((1, environment.horizon, environment.action_dim), dtype=jnp.float32)
    source_type = FixedUniformSourceValue if fixed_uniform else SourceValueNetwork
    source_value = source_type(environment=environment, config=config, model_config=model_config)
    source_variables = source_value.init(source_key, observations, actions, actions)
    decoder = DecoderNetwork(environment=environment, config=config, model_config=model_config)
    decoder_variables = decoder.init(decoder_key, observations, observations, actions, actions)
    optimizer = optax.adam(**optimizer_kwargs(config))
    source_state = TrainState.create(
        apply_fn=source_value.apply,
        params=source_variables['params'],
        tx=optimizer,
    )
    decoder_state = TrainState.create(
        apply_fn=decoder.apply,
        params=decoder_variables['params'],
        tx=optimizer,
    )
    return source_value, source_state, decoder, decoder_state


def rollout_successor_observations(
    rollout: RolloutBatch,
    controller_state: ControllerState,
) -> jax.Array:
    """Align each acted transition with its reached state before any reset."""

    continued_successors = jnp.concatenate(
        (rollout.observations[1:], controller_state.observation[None, ...]),
        axis=0,
    )
    return jnp.where(
        rollout.dones[..., None],
        rollout.terminal_observations,
        continued_successors,
    )


def vim_diagnostics(
    source_value: SourceValueNetwork,
    source_params,
    decoder: DecoderNetwork,
    decoder_params,
    observations: jax.Array,
    actions: jax.Array,
    pre_tanh_actions: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
    *,
    beta: float,
) -> dict[str, jax.Array]:
    """Evaluate the bound and energy-matching relation on fixed tuples."""

    log_q = decoder.apply({'params': decoder_params}, observations, targets, actions, pre_tanh_actions)
    log_h, psi = source_value.apply({'params': source_params}, observations, actions, pre_tanh_actions)
    direct_bound = log_q - log_h
    residual = beta * log_q - log_h - psi
    return {
        'decoder_nll': -_masked_mean(log_q, mask),
        'log_q_mean': _masked_mean(log_q, mask),
        'log_h_mean': _masked_mean(log_h, mask),
        'psi_mean': _masked_mean(psi, mask),
        'direct_bound_nats': _masked_mean(direct_bound, mask),
        'psi_bound_gap_nats': _masked_mean(psi / beta - direct_bound, mask),
        'energy_mse': _masked_mean(jnp.square(residual), mask),
        'action_saturation_fraction': _masked_mean(
            jnp.mean((jnp.abs(actions) > 0.99).astype(jnp.float32), axis=(1, 2)),
            mask,
        ),
    }



VIM_METRICS = (
    'decoder_nll', 'source_value_loss', 'log_q_mean', 'log_h_mean', 'psi_mean',
    'direct_bound_nats', 'psi_bound_gap_nats', 'energy_mse', 'action_saturation_fraction',
    'decoder_gradient_norm', 'source_value_gradient_norm',
)


def make_vim_train_step(source_value, decoder):
    """One decoder update followed by the unchanged detached energy regression."""
    def step(states, batch, mask):
        source_train_state, decoder_train_state = states
        observations, actions, pre_tanh_actions, targets = batch
        def decoder_loss_fn(params):
            log_q = decoder.apply({'params': params}, observations, targets, actions, pre_tanh_actions)
            return -_masked_mean(log_q, mask)

        decoder_loss, decoder_grads = jax.value_and_grad(decoder_loss_fn)(
            decoder_train_state.params
        )
        decoder_grad_norm = optax.global_norm(decoder_grads)
        decoder_train_state = decoder_train_state.apply_gradients(grads=decoder_grads)
        fixed_log_q = jax.lax.stop_gradient(
            decoder.apply(
                {'params': decoder_train_state.params},
                observations,
                targets,
                actions,
                pre_tanh_actions,
            )
        )

        def source_loss_fn(params):
            log_h, psi = source_value.apply({'params': params}, observations, actions, pre_tanh_actions)
            residual = source_value.model_config.beta * fixed_log_q - log_h - psi
            return _masked_mean(jnp.square(residual), mask), (log_h, psi, residual)

        (source_loss, (log_h, psi, residual)), source_grads = jax.value_and_grad(
            source_loss_fn,
            has_aux=True,
        )(source_train_state.params)
        source_grad_norm = optax.global_norm(source_grads)
        source_train_state = source_train_state.apply_gradients(grads=source_grads)
        direct_bound = fixed_log_q - log_h
        metrics = jnp.stack(
            (
                decoder_loss,
                source_loss,
                _masked_mean(fixed_log_q, mask),
                _masked_mean(log_h, mask),
                _masked_mean(psi, mask),
                _masked_mean(direct_bound, mask),
                _masked_mean(psi / source_value.model_config.beta - direct_bound, mask),
                _masked_mean(jnp.square(residual), mask),
                _masked_mean(
                    jnp.mean(
                        (jnp.abs(actions) > 0.99).astype(jnp.float32),
                        axis=(1, 2),
                    ),
                    mask,
                ),
                decoder_grad_norm,
                source_grad_norm,
            )
        )
        return (source_train_state, decoder_train_state), metrics

    return step


@partial(jax.jit, static_argnames=('source_value',))
def empowerment_from_observations(
    source_value: SourceValueNetwork,
    params,
    observations: jax.Array,
    *,
    beta: float,
) -> jax.Array:
    """Return the raw VIM estimate ``psi(s) / beta`` in nats."""

    return source_value.apply(
        {'params': params},
        observations,
        beta,
        method=source_value.empowerment,
    )
