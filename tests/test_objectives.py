"""Classifier objectives against explicit computations at small sizes."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ive, logsumexp

from empowerment_warp.utils.objectives import (
    forward_infonce, sampled_infonce_log_partition, sphere4_log_partition, vmf_forward_loss,
)


def test_forward_infonce_matches_explicit_cross_entropy_and_masks_invalid_rows():
    delta = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
    skills = jax.random.normal(jax.random.PRNGKey(1), (5, 3))
    valid = jnp.array([True, True, True, True, False])
    logits = np.asarray(delta[:4] @ skills[:4].T) / .3
    expected = -np.mean(np.diag(logits) - logsumexp(logits, axis=1))
    np.testing.assert_allclose(forward_infonce(delta, skills, temperature=.3, valid=valid).loss,
                               expected, rtol=1e-5)
    grad = jax.grad(lambda x: forward_infonce(x, skills, temperature=.3, valid=valid).loss)(delta)
    np.testing.assert_array_equal(grad[-1], jnp.zeros(3))
    aligned = forward_infonce(jnp.eye(4), jnp.eye(4), temperature=.1)
    assert float(aligned.accuracy) == 1.0


@pytest.mark.parametrize('rows', [1, 17])
def test_sampled_log_partition_matches_explicit_candidates(rows):
    delta = jax.random.normal(jax.random.PRNGKey(0), (rows, 4))
    skills = jnp.broadcast_to(jnp.eye(4)[0], delta.shape)
    negatives = jax.random.normal(jax.random.PRNGKey(1), (15, 4))
    negatives /= jnp.linalg.norm(negatives, axis=-1, keepdims=True)
    logits = np.concatenate([np.sum(delta * skills, axis=-1)[:, None], delta @ negatives.T], axis=1) / .1
    actual = jax.jit(lambda d: sampled_infonce_log_partition(d, skills, negatives, .1))(delta)
    np.testing.assert_allclose(actual, logsumexp(logits, axis=-1) - np.log(16), atol=1e-5)


def test_sphere4_log_partition_matches_bessel_closed_form():
    kappa = np.array([0, 1e-7, 0.02, 0.5, 1, 4, 20, 1e3, 1e5], dtype=np.float32)
    delta = jnp.asarray(kappa[:, None] * np.array([[1, 0, 0, 0]], dtype=np.float32))
    nonzero = kappa[1:].astype(np.float64)
    expected = np.zeros_like(kappa, dtype=np.float64)
    expected[1:] = nonzero + np.log(2 * ive(1, nonzero) / nonzero)
    np.testing.assert_allclose(sphere4_log_partition(delta, 1), expected, atol=2e-6, rtol=2e-6)
    grad = jax.grad(lambda x: sphere4_log_partition(x, 1).sum())(delta)
    expected_grad = np.zeros((len(kappa), 4))
    expected_grad[1:, 0] = ive(2, nonzero) / ive(1, nonzero)
    np.testing.assert_allclose(grad, expected_grad, atol=3e-6, rtol=1e-4)


def test_vmf_forward_loss_is_per_row_and_masks_invalid_rows():
    delta = jnp.array([[0.2, 0, 0, 0], [0, 0.4, 0, 0], [9, 8, 7, 6]], dtype=jnp.float32)
    skill = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=jnp.float32)
    valid = jnp.array([True, True, False])
    loss, metrics = vmf_forward_loss(delta, skill, temperature=0.1, valid=valid)
    single = [vmf_forward_loss(delta[i:i+1], skill[i:i+1], temperature=0.1)[0] for i in range(2)]
    np.testing.assert_allclose(loss, np.mean(single), atol=1e-6)
    np.testing.assert_allclose(loss, metrics['log_partition'] - metrics['alignment'])
    grad = jax.grad(lambda x: vmf_forward_loss(x, skill, temperature=0.1, valid=valid)[0])(delta)
    np.testing.assert_array_equal(grad[-1], np.zeros(4))
