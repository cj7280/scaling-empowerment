"""Saved Ball-in-Box/Pendulum GCE and VIM controller baselines."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from .config import ExperimentConfig, config_from_dict, effective_scale


@contextmanager
def _reference_prng():
    """Pin the non-partitionable Threefry layout (the JAX 0.4.26 default) for this baseline."""
    import jax

    previous = jax.config.jax_threefry_partitionable
    jax.config.update('jax_threefry_partitionable', False)
    try:
        yield
    finally:
        jax.config.update('jax_threefry_partitionable', previous)


def run(config: ExperimentConfig, *, max_updates: int | None = None) -> dict:
    """Run the controller/probe/replay loop at the resolved budget."""
    with _reference_prng():
        from .runner import run_experiment

        return run_experiment(config, max_updates=max_updates)


def evaluate_checkpoint(config: ExperimentConfig, checkpoint_path: str | Path) -> dict:
    """Evaluate a saved checkpoint into ``config.output_dir`` without media."""
    with _reference_prng():
        import jax.numpy as jnp

        from .checkpoint import restore_checkpoint
        from .config import behavior_fingerprint
        from .runner import evaluate, initialize_runtime

        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        state, functions = initialize_runtime(config)
        state = restore_checkpoint(checkpoint_path, state)
        if state['config_fingerprint'] != behavior_fingerprint(config):
            raise ValueError('checkpoint recipe differs from evaluation config')
        visitation = jnp.empty((0, config.environment.observation_dim), jnp.float32)
        metrics, _ = evaluate(config, state, functions, visitation, media=False)
        return metrics


__all__ = ['ExperimentConfig', 'config_from_dict', 'effective_scale', 'run', 'evaluate_checkpoint']
