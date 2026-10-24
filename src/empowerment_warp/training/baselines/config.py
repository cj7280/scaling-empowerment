# Extracted from empowerment; see reference_manifest.json and LICENSE.
from __future__ import annotations

import argparse

import dataclasses

import hashlib

import json

import math

from dataclasses import dataclass, field

from pathlib import Path

from typing import Sequence

@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    observation_dim: int
    action_dim: int
    action_limit: float
    episode_steps: int
    horizon: int
    probe_trace_steps: int
    gamma: float
    target_dim: int = 0

@dataclass(frozen=True)
class PPOConfig:
    """Unchanged shared PPO recipe; estimator LR is configured separately."""
    num_envs: int = 5
    rollout_steps: int = 1000
    hidden_dim: int = 256
    hidden_layers: int = 2
    learning_rate: float = 1e-3
    update_epochs: int = 4
    num_minibatches: int = 4
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    adam_epsilon: float = 1e-5
    normalize_actor_observations: bool = False

@dataclass(frozen=True)
class EstimatorConfig:
    """Shared network sizes, Adam settings and replay-training budget."""
    hidden_dim: int = 512
    scalar_hidden_dim: int = 256
    hidden_layers: int = 3
    learning_rate: float = 3e-4
    buffer_size: int = 10_000
    batch_size: int = 1024
    replay_passes: int = 10

# Lower bound on per-dimension std for standardized vector-GCE inputs, in native
# simulator units. Measured outcomes are near-constant in visited states
# (BuilderBench cubes 0, AntSoccer ball xy about 0.026), so a pure epsilon would
# amplify physics jitter into unit-scale targets.
VECTOR_GCE_MIN_STD = 1e-2

@dataclass(frozen=True)
class GCEConfig:
    channel_power: float = 1.0
    water_filling_tolerance: float = 1e-3
    water_filling_max_iterations: int = 10_000

@dataclass(frozen=True)
class VIMModelConfig:
    beta: float = 1.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    source_initial_std: float = 1.0

def optimizer_kwargs(config: EstimatorConfig) -> dict[str, float]:
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError('estimator learning rate must be finite and positive')
    return dict(learning_rate=config.learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0)
