"""Shared, import-light recipes and simulator-step accounting for GCE/VIM."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

REFERENCE_REPO = 'https://github.com/PhilipZRH/latent-gce'
REFERENCE_COMMIT = '16eae22a2f8edb8965ba3623e1513ebdfb37e811'
PUREJAXRL_SOURCE = 'https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_continuous_action.py'


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


@dataclass(frozen=True)
class EvaluationConfig:
    """Events cross training-step thresholds; final evaluation is always saved."""
    interval_env_steps: int = 100_000
    media_interval_env_steps: int = 500_000
    samples_per_anchor: int = 128
    deterministic_episodes: int = 5
    stochastic_episodes: int = 5
    grid_bins: int = 65


ENVIRONMENT_PRESETS = {
    'ball_in_box': EnvironmentConfig('ball_in_box', 2, 2, 1.0, 100, 12, 24, 0.95),
    'pendulum': EnvironmentConfig('pendulum', 3, 1, 2.0, 200, 8, 8, 0.5),
}


@dataclass(frozen=True)
class ExperimentConfig:
    method: str
    mode: str
    environment: EnvironmentConfig
    env_steps: int
    output_dir: str
    seed: int = 0
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    gce: GCEConfig = field(default_factory=GCEConfig)
    vim: VIMModelConfig = field(default_factory=VIMModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb_mode: str = 'disabled'
    wandb_project: str = 'empowerment-warp'
    wandb_group: str = ''
    wandb_name: str = ''
    resume: str = ''
    config_only: bool = False

    @property
    def windows_per_trace(self) -> int:
        return self.environment.probe_trace_steps - self.environment.horizon + 1

    @property
    def root_cost(self) -> int:
        if self.method == 'vim':
            return (self.environment.probe_trace_steps - self.environment.horizon
                    + self.windows_per_trace * self.environment.horizon)
        return self.environment.probe_trace_steps

    @property
    def probe_steps_per_update(self) -> int:
        # Complete groups in every method: Ball LCM(24,168)=168 -> 5040;
        # Pendulum LCM(8,8)=8 -> 5000. No masked or uncharged rollouts.
        env = self.environment
        learned_cost = env.probe_trace_steps - env.horizon + self.windows_per_trace * env.horizon
        quantum = math.lcm(env.probe_trace_steps, learned_cost)
        return math.ceil(5000 / quantum) * quantum

    @property
    def roots_per_update(self) -> int:
        return self.probe_steps_per_update // self.root_cost

    @property
    def tuples_per_update(self) -> int:
        return self.roots_per_update * self.windows_per_trace

    @property
    def controller_steps_per_update(self) -> int:
        return self.ppo.num_envs * self.ppo.rollout_steps if self.mode == 'controller' else 0

    @property
    def steps_per_update(self) -> int:
        return self.probe_steps_per_update + self.controller_steps_per_update

    @property
    def updates(self) -> int:
        return self.env_steps // self.steps_per_update

    @property
    def effective_env_steps(self) -> int:
        return self.updates * self.steps_per_update


def optimizer_kwargs(config: EstimatorConfig) -> dict[str, float]:
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError('estimator learning rate must be finite and positive')
    return dict(learning_rate=config.learning_rate, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0)


def config_dict(config: ExperimentConfig) -> dict:
    return dataclasses.asdict(config)


def config_from_dict(settings: Mapping[str, object]) -> ExperimentConfig:
    """Reconstruct the typed configuration from a saved resolved_config.json."""
    values = dict(settings)
    nested_types = {
        'environment': EnvironmentConfig,
        'estimator': EstimatorConfig,
        'ppo': PPOConfig,
        'gce': GCEConfig,
        'vim': VIMModelConfig,
        'evaluation': EvaluationConfig,
    }
    for key, config_type in nested_types.items():
        if key in values:
            values[key] = config_type(**values[key])
    config = ExperimentConfig(**values)
    validate_config(config)
    return config


def behavior_fingerprint(config: ExperimentConfig) -> str:
    payload = config_dict(config)
    # Budget may be extended on resume, but the data/update/evaluation recipe may not.
    for key in ('env_steps', 'output_dir', 'resume', 'config_only', 'wandb_mode',
                'wandb_project', 'wandb_group', 'wandb_name'):
        payload.pop(key)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def event_updates(config: ExperimentConfig, interval: int) -> list[int]:
    return [u for u in range(1, config.updates + 1)
            if u == config.updates or (u * config.steps_per_update // interval
                                      > (u - 1) * config.steps_per_update // interval)]


def effective_scale(config: ExperimentConfig) -> dict:
    cfg = config.estimator
    counts = [min(u * config.tuples_per_update, cfg.buffer_size) for u in range(1, config.updates + 1)]
    batches = [math.ceil(n / cfg.batch_size) for n in counts]
    return dict(
        requested_env_steps=config.env_steps, effective_env_steps=config.effective_env_steps,
        unused_env_steps=config.env_steps - config.effective_env_steps, updates=config.updates,
        probe_steps_per_update=config.probe_steps_per_update,
        controller_steps_per_update=config.controller_steps_per_update,
        roots_per_update=config.roots_per_update, tuples_per_update=config.tuples_per_update,
        windows_per_trace=config.windows_per_trace,
        total_probe_steps=config.updates * config.probe_steps_per_update,
        total_controller_steps=config.updates * config.controller_steps_per_update,
        total_collected_tuples=config.updates * config.tuples_per_update,
        first_update_gradient_minibatches_per_component=batches[0] * cfg.replay_passes,
        total_gradient_minibatches_per_component=sum(batches) * cfg.replay_passes,
        total_valid_replay_row_uses_per_component=sum(counts) * cfg.replay_passes,
        train_components=['channel'] if config.method == 'gce' else ['decoder', 'source_value'],
        ppo_gradient_minibatches_per_update=(config.ppo.update_epochs * config.ppo.num_minibatches
                                            if config.mode == 'controller' else 0),
        evaluation_events=1 + len(set(event_updates(config, config.evaluation.interval_env_steps))
                                  | set(event_updates(config, config.evaluation.media_interval_env_steps))),
        media_events=1 + len(event_updates(config, config.evaluation.media_interval_env_steps)),
        replay_capacity_tuples=cfg.buffer_size, evaluation_steps_counted_in_budget=False,
        root_distribution=('uniform physical roots; intermediate starts follow random walks'
                           if config.mode == 'estimator' else 'controller observations plus uniform random walks'),
        reward=('exp(capacity_bits(current_state))' if config.method == 'gce'
                else 'raw_empowerment_nats(reached_state)'),
    )


def resolve_config(argv: Sequence[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description='Shared GCE/VIM training with an explicit simulator-step cap.')
    parser.add_argument('--env', choices=tuple(ENVIRONMENT_PRESETS), required=True)
    parser.add_argument('--method', choices=('gce', 'vim', 'vim_uniform'), required=True)
    parser.add_argument('--mode', choices=('estimator', 'controller'), required=True)
    parser.add_argument('--env-steps', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--buffer-size', type=int, default=10_000)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--replay-passes', type=int, default=10)
    parser.add_argument('--eval-interval-env-steps', type=int, default=100_000)
    parser.add_argument('--media-interval-env-steps', type=int, default=500_000)
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='disabled')
    parser.add_argument('--wandb-project', default='empowerment-warp')
    parser.add_argument('--wandb-group', default='')
    parser.add_argument('--wandb-name', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--config-only', action='store_true')
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        method=args.method, mode=args.mode, environment=ENVIRONMENT_PRESETS[args.env],
        env_steps=args.env_steps, output_dir=args.output_dir, seed=args.seed,
        estimator=EstimatorConfig(learning_rate=args.learning_rate, buffer_size=args.buffer_size,
                                  batch_size=args.batch_size, replay_passes=args.replay_passes),
        evaluation=EvaluationConfig(interval_env_steps=args.eval_interval_env_steps,
                                    media_interval_env_steps=args.media_interval_env_steps),
        wandb_mode=args.wandb_mode, wandb_project=args.wandb_project,
        wandb_group=args.wandb_group, wandb_name=args.wandb_name, resume=args.resume,
        config_only=args.config_only)
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.method not in ('gce', 'vim', 'vim_uniform') or config.mode not in ('estimator', 'controller'):
        raise ValueError('unknown method or training mode')
    optimizer_kwargs(config.estimator)
    positive = (config.estimator.buffer_size, config.estimator.batch_size,
                config.estimator.replay_passes, config.evaluation.interval_env_steps,
                config.evaluation.media_interval_env_steps, config.evaluation.samples_per_anchor,
                config.evaluation.grid_bins, config.ppo.num_envs, config.ppo.rollout_steps)
    positive += (config.ppo.num_minibatches, config.ppo.update_epochs,
                 config.evaluation.deterministic_episodes, config.evaluation.stochastic_episodes)
    if any(n <= 0 for n in positive) or config.seed < 0:
        raise ValueError('counts must be positive and seed nonnegative')
    if not 0 < config.environment.horizon <= config.environment.probe_trace_steps:
        raise ValueError('probe trace must cover a positive horizon')
    if config.evaluation.samples_per_anchor < 2:
        raise ValueError('at least two anchor samples are needed for standard errors')
    if config.updates < 1:
        raise ValueError(f'--env-steps must cover at least one complete update ({config.steps_per_update} steps)')
    if config.mode == 'controller':
        if config.roots_per_update > config.controller_steps_per_update:
            raise ValueError('controller rollout cannot supply the requested distinct probe roots')
        if config.controller_steps_per_update % config.ppo.num_minibatches:
            raise ValueError('PPO rollout must divide into complete minibatches')


def write_config_artifacts(config: ExperimentConfig) -> None:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if config.config_only and (output / 'run_manifest.json').exists():
        raise FileExistsError('config-only must not overwrite the configuration of an existing run')
    for name, payload in (('resolved_config', config_dict(config)), ('effective_scale', effective_scale(config))):
        (output / f'{name}.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
