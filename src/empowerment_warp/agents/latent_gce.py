# Extracted from empowerment; see training/baselines/reference_manifest.json and LICENSE.
"""LatentGCE estimator: the vector-observation adaptation of the learned channel.

Zhao's water filling over the channel's singular values (from ``gce.py``) and the
vector-observation adaptation of the released LatentGCE learned channel (from
``gce_vector.py``). The network is ``VectorLatentGCE`` in ``utils/networks.py``.
"""

from __future__ import annotations

from functools import partial

from typing import NamedTuple

import jax

import jax.numpy as jnp

import optax

from flax.training.train_state import TrainState

from empowerment_warp.training.baselines.config import (
    VECTOR_GCE_MIN_STD, EnvironmentConfig, EstimatorConfig, optimizer_kwargs,
)

from empowerment_warp.training.baselines.replay import masked_mean

from empowerment_warp.utils.networks import VectorLatentGCE

@partial(
    jax.jit,
    static_argnames=('max_iterations',),
)
def water_filling_from_singular_values(
    singular_values: jax.Array,
    *,
    power: float = 1.0,
    tolerance: float = 1e-3,
    max_iterations: int = 10_000,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run Zhao's iterative water filling on a batch of singular values."""

    noise = 1.0 / (singular_values + 1e-6)
    dimension = singular_values.shape[-1]
    initial_water = jnp.min(noise, axis=-1, keepdims=True) + power / dimension

    def condition(loop_state):
        iteration, water, max_error = loop_state
        return jnp.logical_and(iteration < max_iterations, max_error >= tolerance)

    def body(loop_state):
        iteration, water, _ = loop_state
        allocated = jnp.maximum(water - noise, 0.0)
        current_power = jnp.sum(allocated, axis=-1, keepdims=True)
        error = power - current_power
        water = water + error / dimension
        return iteration + 1, water, jnp.max(jnp.abs(error))

    initial_error = jnp.max(
        jnp.abs(power - jnp.sum(jnp.maximum(initial_water - noise, 0.0), axis=-1, keepdims=True))
    )
    iterations, water, _ = jax.lax.while_loop(
        condition,
        body,
        (jnp.asarray(0, dtype=jnp.int32), initial_water, initial_error),
    )
    allocated_power = jnp.maximum(water - noise, 0.0)
    capacity = 0.5 * jnp.sum(jnp.log2(allocated_power / noise + 1.0), axis=-1)
    return capacity, singular_values, allocated_power, iterations

def channel_singular_values(matrices: jax.Array) -> jax.Array:
    """Recover the descending spectrum from the smaller channel Gram matrix.

    Eigenvalues of G G^T (or G^T G for tall channels) are squared singular
    values. Form and decompose the Gram matrix in float64: float32 can invent
    active modes in large, nearly rank-deficient channels. Return the original
    dtype without changing global precision or regularizing the channel.
    """
    dtype = matrices.dtype
    with jax.enable_x64():
        precise = matrices.astype(jnp.float64)
        transposed = jnp.swapaxes(precise, -1, -2)
        left, right = ((precise, transposed) if precise.shape[-2] <= precise.shape[-1]
                       else (transposed, precise))
        gram = jnp.matmul(left, right, precision=jax.lax.Precision.HIGHEST)
        # CUDA's batched solver can return NaNs for an all-zero matrix. Its
        # spectrum is known exactly; substitute identity only inside the solver.
        zero = jnp.trace(gram, axis1=-2, axis2=-1) == 0
        safe_gram = jnp.where(zero[..., None, None], jnp.eye(gram.shape[-1], dtype=gram.dtype), gram)
        eigenvalues = jnp.linalg.eigvalsh(safe_gram)
        eigenvalues = jnp.where(zero[..., None], 0.0, eigenvalues)
        return jnp.sqrt(jnp.maximum(eigenvalues[..., ::-1], 0.0)).astype(dtype)


# Vector-observation LatentGCE.


class InputMoments(NamedTuple):
    """Per-dimension location and scale of estimator states and outcomes."""
    observation_mean: jax.Array
    observation_std: jax.Array
    target_mean: jax.Array
    target_std: jax.Array


def identity_moments(environment: EnvironmentConfig) -> InputMoments:
    target_dim = environment.target_dim or environment.observation_dim
    return InputMoments(jnp.zeros(environment.observation_dim), jnp.ones(environment.observation_dim),
                        jnp.zeros(target_dim), jnp.ones(target_dim))


@jax.jit
def replay_moments(buffer) -> InputMoments:
    """Moments of the valid replay rows; identity until the replay holds data.

    Standardizing plays the role of the released model's [0, 1] pixel scaling,
    so its reconstruction coefficients act on comparable magnitudes.
    """
    valid = (jnp.arange(buffer.observations.shape[0]) < buffer.count)[:, None]
    count = jnp.maximum(buffer.count, 1)
    def moments(values, mask=valid, sample_count=count):
        mean = jnp.sum(values * mask, axis=0) / sample_count
        variance = jnp.sum(jnp.square(values - mean) * mask, axis=0) / sample_count
        std = jnp.maximum(jnp.sqrt(variance), VECTOR_GCE_MIN_STD)
        empty = buffer.count == 0
        return jnp.where(empty, 0., mean), jnp.where(empty, 1., std)
    if buffer.observations.shape[-1] == buffer.targets.shape[-1]:
        # A shared encoder must see one coordinate system for starts and outcomes.
        shared = moments(jnp.concatenate((buffer.observations, buffer.targets)),
                         jnp.concatenate((valid, valid)), 2 * count)
        return InputMoments(*shared, *shared)
    return InputMoments(*moments(buffer.observations), *moments(buffer.targets))


def standardize(moments: InputMoments, observations, targets=None):
    observations = (observations - moments.observation_mean) / moments.observation_std
    if targets is None:
        return observations
    return observations, (targets - moments.target_mean) / moments.target_std


VECTOR_GCE_METRICS = (
    'total_loss', 'latent_prediction_loss', 'observation_reconstruction_loss',
    'prediction_reconstruction_loss', 'target_reconstruction_loss', 'action_reconstruction_loss',
    'latent_regularization', 'channel_gradient_norm',
)


def vector_losses(network, params, observations, actions, targets, mask):
    rows = network.apply({'params': params}, observations, actions, targets)
    prediction = masked_mean(rows['latent_prediction'], mask)
    observation = masked_mean(rows['observation_reconstruction'], mask)
    reconstruction = masked_mean(rows['prediction_reconstruction'], mask)
    target_reconstruction = masked_mean(rows['target_reconstruction'], mask)
    action = masked_mean(rows['action_reconstruction'], mask)
    context_norm = masked_mean(rows['context_squared'], mask)
    target_norm = masked_mean(rows['target_squared'], mask)
    action_norm = masked_mean(rows['action_squared'], mask)
    regularization = jnp.abs(1.0 - context_norm) + jnp.abs(1.0 - action_norm)
    if network.observation_dim != network.target_dim:
        regularization += jnp.abs(1.0 - target_norm)
    # Executable LatentGCEImage coefficients, with MLPs replacing image convnets.
    # Separate outcome encoders additionally need their own autoencoding term.
    total = (prediction + 500.0 * (observation + reconstruction + target_reconstruction)
             + 10.0 * action + regularization)
    return total, (prediction, observation, reconstruction, target_reconstruction,
                   action, regularization)


def create_vector_gce_train_state(key: jax.Array, environment: EnvironmentConfig,
                                  config: EstimatorConfig, *, latent_dim: int = 32,
                                  bound_action_decoder: bool = False):
    network = VectorLatentGCE(
        observation_dim=environment.observation_dim,
        target_dim=environment.target_dim or environment.observation_dim,
        action_sequence_dim=environment.horizon * environment.action_dim,
        latent_dim=latent_dim, hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers, bound_action_decoder=bound_action_decoder,
    )
    variables = network.init(
        key, jnp.zeros((1, environment.observation_dim), jnp.float32),
        jnp.zeros((1, environment.horizon, environment.action_dim), jnp.float32),
        jnp.zeros((1, environment.target_dim or environment.observation_dim), jnp.float32),
    )
    state = TrainState.create(apply_fn=network.apply, params=variables['params'],
                              tx=optax.adam(**optimizer_kwargs(config)))
    return network, state


def make_vector_gce_train_step(network):
    def step(state, batch, mask):
        def loss_fn(params):
            return vector_losses(network, params, batch.observations, batch.actions,
                                 batch.targets, mask)
        (total, components), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        metrics = jnp.stack((total, *components, optax.global_norm(grads)))
        return state.apply_gradients(grads=grads), metrics
    return step


@partial(jax.jit, static_argnames=('network', 'max_iterations'))
def vector_capacity_from_observations(network, params, observations, *, power=1.0,
                                      tolerance=1e-3, max_iterations=10_000):
    _, matrices = network.apply({'params': params}, observations, method=network.channel)
    singular_values = channel_singular_values(matrices)
    return water_filling_from_singular_values(
        singular_values, power=power, tolerance=tolerance, max_iterations=max_iterations)
