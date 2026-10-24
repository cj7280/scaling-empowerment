"""Pure objective kernels shared by the potential skill learner."""

from __future__ import annotations

from typing import NamedTuple
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import i0e, i1e


class InfoNCEMetrics(NamedTuple):
    loss: jax.Array
    forward_loss: jax.Array
    mi_lower_bound: jax.Array
    accuracy: jax.Array






def potential_reward(phi_reference: jax.Array, phi_next: jax.Array, skills: jax.Array) -> jax.Array:
    """Inner-product potential reward, `<phi(s') - phi(ref), z>`."""

    return jnp.sum((phi_next - phi_reference) * skills, axis=-1)


def forward_infonce(
    deltas: jax.Array,
    skills: jax.Array,
    *,
    temperature: float,
    valid: jax.Array | None = None,
) -> InfoNCEMetrics:
    """Forward in-batch skill InfoNCE with explicit invalid-row masking."""

    if deltas.shape != skills.shape:
        raise ValueError(f"deltas and skills must have equal shape, got {deltas.shape} and {skills.shape}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch_size = deltas.shape[0]
    valid = jnp.ones((batch_size,), dtype=jnp.bool_) if valid is None else valid.astype(jnp.bool_)
    logits = deltas @ skills.T / jnp.asarray(temperature, deltas.dtype)
    pair_valid = valid[:, None] & valid[None, :]
    masked_logits = jnp.where(pair_valid, logits, jnp.asarray(-1e9, logits.dtype))
    positive = jnp.diag(logits)
    count = jnp.maximum(jnp.sum(valid.astype(logits.dtype)), 1.0)
    forward = jnp.sum(jnp.where(valid, jax.nn.logsumexp(masked_logits, axis=-1) - positive, 0.0)) / count
    accuracy = jnp.sum(
        jnp.where(valid, (jnp.argmax(masked_logits, axis=-1) == jnp.arange(batch_size)).astype(logits.dtype), 0.0)
    ) / count
    return InfoNCEMetrics(
        loss=forward,
        forward_loss=forward,
        mi_lower_bound=jnp.log(count + 1e-8) - forward,
        accuracy=accuracy,
    )


def in_batch_skill_accuracy(
    deltas: jax.Array,
    skills: jax.Array,
    *,
    temperature: float,
    valid: jax.Array,
) -> jax.Array:
    """Diagnostic skill retrieval accuracy, without either contrastive loss."""

    logits = deltas @ skills.T / jnp.asarray(temperature, deltas.dtype)
    pair_valid = valid[:, None] & valid[None, :]
    masked = jnp.where(pair_valid, logits, jnp.asarray(-1e9, logits.dtype))
    correct = jnp.argmax(masked, axis=-1) == jnp.arange(deltas.shape[0])
    count = jnp.maximum(jnp.sum(valid.astype(logits.dtype)), 1.0)
    return jnp.sum(jnp.where(valid, correct.astype(logits.dtype), 0.0)) / count




def sampled_infonce_log_partition(
    deltas: jax.Array, skills: jax.Array, negative_skills: jax.Array, temperature: float,
) -> jax.Array:
    """Log-mean-exp over the executed skill plus independent prior negatives.

    Chunk rows so PQN's whole-rollout reward query never materializes an
    episode-by-candidate matrix. The positive is included, as in InfoNCE.
    """
    if deltas.shape != skills.shape or deltas.shape[-1] != negative_skills.shape[-1]:
        raise ValueError("InfoNCE candidates and displacements must have matching skill dimensions")
    if temperature <= 0 or negative_skills.shape[0] < 1:
        raise ValueError("InfoNCE needs positive temperature and at least one negative")
    count = negative_skills.shape[0] + 1

    def row(pair):
        delta, skill = pair
        positive = jnp.sum(delta * skill) / temperature
        negative = jax.nn.logsumexp(negative_skills @ delta / temperature)
        return jnp.logaddexp(positive, negative) - jnp.log(jnp.asarray(count, delta.dtype))

    return jax.lax.map(row, (deltas, skills), batch_size=256)


def sphere4_log_partition(deltas: jax.Array, temperature: float) -> jax.Array:
    """Log E[exp(delta.z / temperature)] for uniform unit 4-D skills."""

    if deltas.shape[-1] != 4 or temperature <= 0:
        raise ValueError("sphere4_log_partition requires 4-D deltas and positive temperature")
    squared = jnp.sum(jnp.square(deltas / temperature), axis=-1)
    # A positive Bessel series avoids cancellation in the derivative near zero.
    # Clamping the inactive branch also keeps derivatives finite at delta = 0.
    radius = jnp.sqrt(jnp.maximum(squared, 0.25))
    closed_form = radius + jnp.log(2.0 * i1e(radius) / radius)
    # Keep the unused polynomial finite too: where's transpose can otherwise
    # multiply zero by an overflowing derivative for an uncapped displacement.
    series_squared = jnp.minimum(squared, 1.0)
    series = series_squared * (1 / 8 + series_squared * (1 / 192 + series_squared * (
        1 / 9216 + series_squared * (1 / 737280 + series_squared / 88473600)
    )))
    return jnp.where(squared < 1.0, jnp.log1p(series), closed_form)


def sphere_log_partition(deltas: jax.Array, temperature: float) -> jax.Array:
    """Log E exp(delta.z / temperature) for the supported uniform-sphere priors.

    A positive Bessel series handles small radii; scaled Bessel recurrence handles
    large radii without exponentiating them. The 4-D path is unchanged.
    """
    dimension = deltas.shape[-1]
    if dimension not in (2, 4, 9, 16) or temperature <= 0:
        raise ValueError("sphere_log_partition supports dimensions 2, 4, 9, 16 and positive temperature")
    if dimension == 4:
        return sphere4_log_partition(deltas, temperature)
    squared = jnp.sum(jnp.square(deltas / temperature), axis=-1)
    # M_d(k) = sum_m (k²/4)^m / (m! (d/2)_m). Scaling the polynomial's
    # argument to [0,1] keeps its coefficients representable in float32.
    threshold = 20.0
    argument = jnp.minimum(squared / threshold**2, 1.0)
    half_dimension = dimension / 2.0
    coefficients = [math.exp(
        m * math.log(threshold**2 / 4) + math.lgamma(half_dimension)
        - math.lgamma(m + 1) - math.lgamma(half_dimension + m)
    ) for m in range(1, 33)]
    polynomial = jnp.asarray(coefficients[-1], dtype=deltas.dtype)
    for coefficient in reversed(coefficients[:-1]):
        polynomial = polynomial * argument + coefficient
    series = jnp.log1p(argument * polynomial)

    # Both branches stay finite even when inactive, including derivatives at 0.
    radius = jnp.sqrt(jnp.maximum(squared, threshold**2 / 4))
    order = half_dimension - 1
    if dimension % 2 == 0:
        previous, current = i0e(radius), i1e(radius)
        for n in range(1, int(order)):
            previous, current = current, previous - (2 * n / radius) * current
        scaled_bessel = previous if order == 0 else current
    else:
        factor = jax.lax.rsqrt(2 * jnp.pi * radius)
        exponential = jnp.exp(-2 * radius)
        previous, current = factor * (1 + exponential), factor * (1 - exponential)
        for n in range(int(order - 0.5)):
            previous, current = current, previous - ((2 * n + 1) / radius) * current
        scaled_bessel = current
    closed_form = (radius + jnp.log(scaled_bessel) + math.lgamma(half_dimension)
                   + order * (math.log(2) - jnp.log(radius)))
    return jnp.where(squared < threshold**2, series, closed_form)


def vmf_posterior_kl(deltas: jax.Array, temperature: float) -> jax.Array:
    """KL(vMF(delta / temperature) || uniform), in nats per state pair."""
    # Directional differentiation gives kappa A'(kappa) without a norm at zero.
    # Bound the inactive branch to avoid cancellation/overflow for large norms.
    squared = jnp.sum(jnp.square(deltas / temperature), axis=-1)
    radius = jnp.sqrt(jnp.maximum(squared, 1.0))
    bounded = deltas * jnp.minimum(1.0, 100.0 / radius)[..., None]
    partition, radial_derivative = jax.jvp(
        lambda x: sphere_log_partition(x, temperature), (bounded,), (bounded,)
    )
    regular = radial_derivative - partition

    # Scaled-Bessel asymptotics evaluate the same KL without subtracting two
    # O(kappa) terms. Twelve terms suffice at kappa >= 100 for supported d <= 16.
    order = deltas.shape[-1] / 2.0 - 1.0
    inverse = 1.0 / jnp.maximum(radius, 100.0)
    term = jnp.ones_like(radius)
    polynomial, radial_polynomial = term, jnp.zeros_like(radius)
    for n in range(1, 13):
        term = term * (-(4 * order**2 - (2 * n - 1)**2) / (8 * n)) * inverse
        polynomial = polynomial + term
        radial_polynomial = radial_polynomial - n * term
    constant = order * math.log(2.0) + math.lgamma(order + 1) - 0.5 * math.log(2 * math.pi)
    large = ((order + 0.5) * (jnp.log(jnp.maximum(radius, 100.0)) - 1.0)
             - constant - jnp.log(polynomial) + radial_polynomial / polynomial)
    return jnp.maximum(jnp.where(squared < 100.0**2, regular, large), 0.0)


def vmf_forward_loss(
    deltas: jax.Array,
    skills: jax.Array,
    *,
    temperature: float,
    valid: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Exact spherical decoder likelihood ratio; no sampled skill normalizer."""

    if deltas.shape != skills.shape:
        raise ValueError("deltas and skills must have equal shape")
    valid = jnp.ones(deltas.shape[0], dtype=jnp.bool_) if valid is None else valid.astype(jnp.bool_)
    count = jnp.maximum(jnp.sum(valid.astype(deltas.dtype)), 1.0)
    positive = jnp.sum(deltas * skills, axis=-1) / temperature
    partition = sphere_log_partition(deltas, temperature)
    alignment = jnp.sum(jnp.where(valid, positive, 0.0)) / count
    log_partition = jnp.sum(jnp.where(valid, partition, 0.0)) / count
    loss = log_partition - alignment
    return loss, {"alignment": alignment, "log_partition": log_partition}
