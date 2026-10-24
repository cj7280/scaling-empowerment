"""State-conditioned linear channel and unchanged Zhao water filling."""
from __future__ import annotations
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from empowerment_warp.baselines.toy_baselines.config import EnvironmentConfig, EstimatorConfig, optimizer_kwargs
from empowerment_warp.baselines.toy_baselines.replay import masked_mean

class LatentGCENetwork(nn.Module):
    """Predict ``s[t+H] = G(s[t]) @ U + b(s[t])``."""

    observation_dim: int
    action_sequence_dim: int
    matrix_hidden_dim: int = 512
    matrix_hidden_layers: int = 3
    bias_hidden_dim: int = 256
    bias_hidden_layers: int = 3

    @nn.compact
    def __call__(self, observations: jax.Array, actions: jax.Array | None = None):
        matrix_features = observations
        for _ in range(self.matrix_hidden_layers):
            matrix_features = nn.relu(
                nn.Dense(
                    self.matrix_hidden_dim,
                    kernel_init=nn.initializers.glorot_uniform(),
                    bias_init=nn.initializers.zeros_init(),
                )(matrix_features)
            )
        matrix_flat = nn.Dense(
            self.observation_dim * self.action_sequence_dim,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )(matrix_features)
        matrix = matrix_flat.reshape(
            observations.shape[:-1] + (self.observation_dim, self.action_sequence_dim)
        )

        bias = observations
        for _ in range(self.bias_hidden_layers):
            bias = nn.relu(
                nn.Dense(
                    self.bias_hidden_dim,
                    kernel_init=nn.initializers.glorot_uniform(),
                    bias_init=nn.initializers.zeros_init(),
                )(bias)
            )
        bias = nn.Dense(
            self.observation_dim,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )(bias)
        prediction = None if actions is None else jnp.einsum('...ij,...j->...i', matrix, actions) + bias
        return matrix, bias, prediction


def create_gce_train_state(
    key: jax.Array,
    environment: EnvironmentConfig,
    config: EstimatorConfig,
) -> tuple[LatentGCENetwork, TrainState]:
    """Initialize Zhao's matrix and bias MLPs with TF-compatible Glorot kernels."""

    action_sequence_dim = environment.horizon * environment.action_dim
    network = LatentGCENetwork(
        observation_dim=environment.observation_dim,
        action_sequence_dim=action_sequence_dim,
        matrix_hidden_dim=config.hidden_dim,
        matrix_hidden_layers=config.hidden_layers,
        bias_hidden_dim=config.scalar_hidden_dim,
        bias_hidden_layers=config.hidden_layers,
    )
    variables = network.init(
        key,
        jnp.zeros((1, environment.observation_dim), dtype=jnp.float32),
        jnp.zeros((1, action_sequence_dim), dtype=jnp.float32),
    )
    state = TrainState.create(
        apply_fn=network.apply,
        params=variables['params'],
        tx=optax.adam(**optimizer_kwargs(config)),
    )
    return network, state


@partial(jax.jit, static_argnames=('network',))
def prediction_errors(
    network: LatentGCENetwork,
    params,
    observations: jax.Array,
    actions: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """Return Zhao's per-tuple mean squared prediction error."""

    _, _, predictions = network.apply({'params': params}, observations, actions.reshape((observations.shape[0], -1)))
    return jnp.mean(jnp.square(predictions - targets), axis=-1)



def make_gce_train_step(network):
    """One channel-MSE update; shared replay code owns masking and cadence."""
    def step(state, batch, mask):
        def loss_fn(params):
            errors = prediction_errors(network, params, batch.observations, batch.actions, batch.targets)
            return masked_mean(errors, mask)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), jnp.stack((loss, optax.global_norm(grads)))
    return step


GCE_METRICS = ('channel_loss', 'channel_gradient_norm')

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


@partial(
    jax.jit,
    static_argnames=('network', 'max_iterations'),
)
def capacity_from_observations(
    network: LatentGCENetwork,
    params,
    observations: jax.Array,
    *,
    power: float = 1.0,
    tolerance: float = 1e-3,
    max_iterations: int = 10_000,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Evaluate ``G(s)`` and compute its closed-form channel capacity."""

    matrices, _, _ = network.apply({'params': params}, observations, None)
    singular_values = jnp.linalg.svd(matrices, compute_uv=False)
    return water_filling_from_singular_values(
        singular_values,
        power=power,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
