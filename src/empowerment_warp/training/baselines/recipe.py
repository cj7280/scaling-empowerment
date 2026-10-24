"""Explicit Warp transfer recipe; resolving it does not initialize JAX or physics."""
from dataclasses import asdict, dataclass, field, replace
import math

from .config import (VECTOR_GCE_MIN_STD, EnvironmentConfig, EstimatorConfig, GCEConfig, PPOConfig,
                     VIMModelConfig)


@dataclass(frozen=True)
class ControllerPPOConfig(PPOConfig):
    anneal_learning_rate: bool = False
    normalization_epsilon: float = 1e-8
    normalization_clip: float = 10.0


@dataclass(frozen=True)
class Recipe:
    environment: str
    method: str
    horizon: int
    total_env_steps: int
    run_dir: str
    seed: int = 0
    reset_preset: str | None = None
    num_envs: int = 64
    rollout_steps: int = 80
    episode_steps: int = 400
    gamma: float = 0.99
    eval_every_updates: int = 20
    eval_every_env_steps: int | None = None
    max_updates: int = 0
    resume: bool = False
    config_only: bool = False
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    checkpoint_keep: int = 0
    vector_gce_latent_dim: int = 32
    vector_gce_learning_rate: float = 1e-4
    vector_gce_normalize_inputs: bool = False
    # Defaults preserve recipes/checkpoints saved before these choices were explicit.
    vector_gce_bound_action_decoder: bool = False
    exponentiate_empowerment: bool = True
    ppo: ControllerPPOConfig | None = None

    def __post_init__(self):
        ppo = self.ppo
        if isinstance(ppo, dict):
            ppo = ControllerPPOConfig(**ppo)
        if ppo is None:
            ppo = ControllerPPOConfig()
        if not isinstance(ppo, ControllerPPOConfig):
            raise TypeError('ppo must be a ControllerPPOConfig or mapping')
        object.__setattr__(self, 'ppo', replace(ppo, num_envs=self.num_envs,
                                              rollout_steps=self.rollout_steps))
        if ppo.normalization_epsilon <= 0 or ppo.normalization_clip <= 0:
            raise ValueError('normalization epsilon and clipping must be positive')
        if self.reset_preset is None:
            object.__setattr__(self, 'reset_preset', 'closer_ball' if self.environment == 'ant_soccer' else 'standard')
        if self.reset_preset not in {'standard', 'closer_ball', 'center_ball', 'past_center_ball'}:
            raise ValueError('unknown reset preset')
        if self.environment not in ('ant', 'ant_soccer', 'builderbench_corner3'):
            raise ValueError('only continuous Warp benchmarks are supported')
        if self.eval_every_env_steps is None:
            object.__setattr__(self, 'eval_every_env_steps',
                               5_000_000 if self.environment == 'builderbench_corner3' else 50_000_000)
        if self.method not in ('gce_vector', 'vim'):
            raise ValueError("baseline method must be 'gce_vector' or 'vim'")
        if min(self.horizon, self.num_envs, self.rollout_steps, self.episode_steps,
               self.eval_every_updates, self.eval_every_env_steps, self.estimator.buffer_size,
               self.estimator.batch_size, self.estimator.replay_passes) < 1:
            raise ValueError('all sizes and cadences must be positive')
        if self.seed < 0 or self.max_updates < 0 or not 0 < self.gamma <= 1:
            raise ValueError('invalid seed, smoke limit or discount')
        if self.checkpoint_keep < 0:
            raise ValueError('checkpoint_keep must be nonnegative; zero keeps all')
        if (self.vector_gce_latent_dim < 1 or not math.isfinite(self.vector_gce_learning_rate)
                or self.vector_gce_learning_rate <= 0):
            raise ValueError('vector GCE dimension and learning rate must be positive')
        if self.vector_gce_normalize_inputs and self.method != 'gce_vector':
            raise ValueError('estimator input normalization applies only to gce_vector')
        if self.num_envs * self.rollout_steps % self.ppo.num_minibatches:
            raise ValueError('PPO rollout must divide into its minibatches')
        if self.total_env_steps < self.steps_per_update:
            raise ValueError('budget must cover a complete controller-plus-probe update')

    @property
    def steps_per_update(self):
        return self.num_envs * (self.rollout_steps + self.horizon)

    @property
    def updates(self):
        return self.total_env_steps // self.steps_per_update

    @property
    def effective_estimator(self):
        return (replace(self.estimator, learning_rate=self.vector_gce_learning_rate)
                if self.method == 'gce_vector' else self.estimator)

    @property
    def outcome_indices(self):
        return {'ant': None, 'ant_soccer': (15, 16),
                'builderbench_corner3': tuple(range(10, 19))}[self.environment]

    @property
    def model_environment(self):
        obs, action = (51, 5) if self.environment == 'builderbench_corner3' else (42, 8)
        target = obs if self.outcome_indices is None else len(self.outcome_indices)
        return EnvironmentConfig(self.environment, obs, action, 1., self.episode_steps,
                                 self.horizon, self.horizon, self.gamma, target)

    def resolved(self):
        result = asdict(self)
        result['estimator'] = asdict(self.effective_estimator)
        ppo = asdict(self.ppo)
        if not (self.ppo.anneal_learning_rate or self.ppo.normalize_actor_observations):
            for key in ('anneal_learning_rate', 'normalization_epsilon', 'normalization_clip'):
                ppo.pop(key)
        result.update(ppo=ppo, gce=asdict(GCEConfig()), vim=asdict(VIMModelConfig()),
                      model_environment=asdict(self.model_environment),
                      outcome_indices=None if self.outcome_indices is None else list(self.outcome_indices),
                      controller_action_units='tanh Gaussian action in [-1,1], passed directly to Warp',
                      controller_distribution='tanh_normal',
                      controller_entropy='squashed differential entropy; analytic Gaussian plus sampled log Jacobian',
                      probe_action_units='normalized [-1,1] passed directly to Warp',
                      probe_reset_semantics='H raw physics steps; no resets or time-limit shortening',
                      root_distribution='one uniformly chosen controller time per lane',
                      reward=(('exp(capacity_bits(current))' if self.exponentiate_empowerment
                               else 'capacity_bits(current)') if self.method == 'gce_vector'
                              else 'max(0, psi(reached)/beta)'),
                      vector_gce_model=(f'learned vector state/action encoders; '
                                        f'{self.vector_gce_latent_dim}x{self.vector_gce_latent_dim} latent channel')
                                       if self.method == 'gce_vector' else None,
                      vector_gce_inputs=(('replay-standardized states and outcomes; '
                                          f'std floor {VECTOR_GCE_MIN_STD}')
                                         if self.vector_gce_normalize_inputs else 'raw')
                                        if self.method == 'gce_vector' else None,
                      algorithm=self.method)
        e = self.estimator
        counts = [min(i * self.num_envs, e.buffer_size) for i in range(1, self.updates + 1)]
        result['effective_scale'] = dict(
            controller_steps_per_update=self.num_envs*self.rollout_steps,
            probe_steps_per_update=self.num_envs*self.horizon,
            tuples_per_update=self.num_envs, updates=self.updates,
            total_training_steps=self.updates*self.steps_per_update,
            total_controller_steps=self.updates*self.num_envs*self.rollout_steps,
            total_probe_steps=self.updates*self.num_envs*self.horizon,
            unused_budget=self.total_env_steps-self.updates*self.steps_per_update,
            replay_capacity_tuples=e.buffer_size,
            first_estimator_minibatches=e.replay_passes*math.ceil(counts[0]/e.batch_size),
            total_estimator_minibatches_per_component=e.replay_passes*sum(math.ceil(n/e.batch_size) for n in counts),
            total_estimator_valid_row_uses_per_component=e.replay_passes*sum(counts),
            ppo_minibatches_per_update=self.ppo.num_minibatches*self.ppo.update_epochs,
            ppo_samples_per_controller_transition=self.ppo.update_epochs,
            evaluation_worlds=128,
            evaluation_horizon_actions=100,
            evaluation_cadence_env_steps=self.eval_every_env_steps,
            evaluation_steps_per_event=128*100+self.num_envs*self.horizon,
            evaluation_steps_counted_in_training_budget=False,
            smoke_max_updates=self.max_updates,
        )
        return result
