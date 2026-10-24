# Extracted from empowerment; see reference_manifest.json and LICENSE.
"""Shared all-example retention, masked minibatches and replay cadence."""

import math
from functools import partial

from typing import NamedTuple

import jax

import jax.numpy as jnp

from empowerment_warp.training.baselines.config import EnvironmentConfig, EstimatorConfig

class ProbeBatch(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    pre_tanh_actions: jax.Array
    targets: jax.Array

class ProbeBuffer(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    pre_tanh_actions: jax.Array
    targets: jax.Array
    count: jax.Array

def empty_probe_buffer(environment: EnvironmentConfig, config: EstimatorConfig) -> ProbeBuffer:
    states = jnp.zeros((config.buffer_size, environment.observation_dim), jnp.float32)
    actions = jnp.zeros((config.buffer_size, environment.horizon, environment.action_dim), jnp.float32)
    targets = jnp.zeros((config.buffer_size, environment.target_dim or environment.observation_dim), jnp.float32)
    return ProbeBuffer(states, actions, actions, targets, jnp.asarray(0, jnp.int32))

def uniform_replay_indices(old_count, candidate_valid, key, *, capacity):
    """Uniformly retain valid old and new rows, packing valid rows before padding."""
    valid = jnp.concatenate((jnp.arange(capacity) < old_count, candidate_valid))
    scores = jnp.where(valid, jax.random.uniform(key, valid.shape), -jnp.inf)
    _, selected = jax.lax.top_k(scores, capacity)
    count = jnp.minimum(jnp.sum(valid, dtype=jnp.int32), capacity)
    return selected[jnp.argsort(~valid[selected])], count

@jax.jit
def insert_probe_batch(buffer: ProbeBuffer, batch: ProbeBatch, key) -> ProbeBuffer:
    capacity = buffer.observations.shape[0]
    indices, count = uniform_replay_indices(
        buffer.count, jnp.ones(batch.observations.shape[0], dtype=jnp.bool_), key, capacity=capacity)
    arrays = [jnp.concatenate((old, new))[indices] for old, new in zip(buffer[:4], batch)]
    return ProbeBuffer(*arrays, count)

def masked_mean(values, mask):
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)

def make_replay_epoch(update_batch, metric_names, config: EstimatorConfig):
    """Wrap a method's minibatch update in one common masked replay pass."""
    capacity, batch_size = config.buffer_size, config.batch_size
    padded_size = math.ceil(capacity / batch_size) * batch_size

    @jax.jit
    def epoch(state, buffer: ProbeBuffer, key):
        valid = jnp.arange(capacity) < buffer.count
        scores = jnp.where(valid, jax.random.uniform(key, (capacity,)), -jnp.inf)
        permutation = jnp.argsort(scores)[::-1]
        permutation = jnp.pad(permutation, (0, padded_size - capacity))
        batches = permutation.reshape((-1, batch_size))
        active_batches = (buffer.count + batch_size - 1) // batch_size

        def step(carry, indices):
            current, metric_sum, index = carry
            mask = (index * batch_size + jnp.arange(batch_size) < buffer.count).astype(jnp.float32)
            batch = ProbeBatch(*(array[indices] for array in buffer[:4]))
            current, metrics = jax.lax.cond(
                index < active_batches,
                lambda s: update_batch(s, batch, mask),
                lambda s: (s, jnp.zeros(len(metric_names), jnp.float32)), current)
            return (current, metric_sum + metrics, index + 1), None

        initial = state, jnp.zeros(len(metric_names), jnp.float32), jnp.asarray(0)
        (state, sums, _), _ = jax.lax.scan(step, initial, batches)
        means = sums / jnp.maximum(active_batches, 1)
        metrics = dict(zip(metric_names, means))
        metrics['active_minibatches'] = active_batches
        return state, metrics
    return epoch

@partial(jax.jit, static_argnames=('epoch', 'passes'))
def train_replay(epoch, state, buffer, key, passes):
    """Use the same cadence on every collection, including the first."""
    def step(carry, _):
        state, key = carry
        key, epoch_key = jax.random.split(key)
        state, metrics = epoch(state, buffer, epoch_key)
        return (state, key), metrics

    (state, key), history = jax.lax.scan(step, (state, key), None, length=passes)
    metrics = jax.tree.map(lambda values: values[-1], history)
    return state, dict(metrics, replay_passes=jnp.asarray(passes),
                       gradient_minibatches=metrics['active_minibatches'] * passes), key
