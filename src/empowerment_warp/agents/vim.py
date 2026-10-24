# Extracted from empowerment; see training/baselines/reference_manifest.json and LICENSE.
"""State-only variational information maximization and probe replay.

The source/value and decoder networks are ``SourceValueNetwork`` and
``DecoderNetwork`` in ``utils/networks.py``.
"""

from __future__ import annotations

from functools import partial

import jax

import jax.numpy as jnp

import optax

from flax.training.train_state import TrainState

from empowerment_warp.training.baselines.config import EnvironmentConfig

from empowerment_warp.agents.ppo import ControllerState, RolloutBatch

from empowerment_warp.training.baselines.config import EstimatorConfig, VIMModelConfig, optimizer_kwargs

from empowerment_warp.training.baselines.replay import masked_mean as _masked_mean

from empowerment_warp.utils.networks import DecoderNetwork, SourceValueNetwork

def create_vim_train_states(
    source_key: jax.Array,
    decoder_key: jax.Array,
    environment: EnvironmentConfig,
    config: EstimatorConfig,
    model_config: VIMModelConfig = VIMModelConfig(),
) -> tuple[SourceValueNetwork, TrainState, DecoderNetwork, TrainState]:
    """Initialize VIM networks and independent states for the selected optimizer."""

    observations = jnp.zeros((1, environment.observation_dim), dtype=jnp.float32)
    actions = jnp.zeros((1, environment.horizon, environment.action_dim), dtype=jnp.float32)
    source_value = SourceValueNetwork(environment=environment, config=config, model_config=model_config)
    source_variables = source_value.init(source_key, observations, actions, actions)
    decoder = DecoderNetwork(environment=environment, config=config, model_config=model_config)
    targets = jnp.zeros((1, environment.target_dim or environment.observation_dim), dtype=jnp.float32)
    decoder_variables = decoder.init(decoder_key, observations, targets, actions, actions)
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
    """Nonnegative state empowerment for controller rewards/evaluation.

    The source/decoder training objective continues to use the raw psi.
    """

    return jnp.maximum(source_value.apply(
        {'params': params},
        observations,
        beta,
        method=source_value.empowerment,
    ), 0.0)
