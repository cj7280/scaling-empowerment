"""Typed OGBench-derived Ant and AntSoccer configurations."""

from __future__ import annotations

from dataclasses import dataclass


SOCCER_TASKS = (
    ((1, 6), (2, 3), (5, 2)),
    ((2, 2), (5, 5), (2, 2)),
    ((6, 1), (2, 3), (6, 6)),
    ((6, 6), (1, 1), (6, 1)),
    ((4, 6), (6, 2), (1, 6)),
)


@dataclass(frozen=True, slots=True)
class AntConfig:
    num_envs: int = 1024
    episode_length: int = 400
    sim_dt: float = 0.02
    ctrl_dt: float = 0.1
    integrator: str = "rk4"
    graph_mode: str = "warp"
    naconmax_per_env: int = 64
    njmax_per_env: int = 128
    reset_noise_scale: float = 0.1
    terminate_on_nonfinite: bool = True

    def __post_init__(self) -> None:
        if self.num_envs <= 0 or self.episode_length <= 0:
            raise ValueError("num_envs and episode_length must be positive")
        if self.sim_dt <= 0 or self.ctrl_dt <= 0:
            raise ValueError("simulation and control timesteps must be positive")
        ratio = self.ctrl_dt / self.sim_dt
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("ctrl_dt must be an integer multiple of sim_dt")
        if self.integrator not in {"euler", "rk4", "implicit", "implicitfast"}:
            raise ValueError(f"unsupported MuJoCo integrator: {self.integrator}")
        if self.graph_mode not in {"none", "warp", "warp_staged", "warp_staged_ex"}:
            raise ValueError(f"unsupported Warp graph mode: {self.graph_mode}")
        if self.naconmax_per_env <= 0 or self.njmax_per_env <= 0:
            raise ValueError("contact and constraint capacities must be positive")
        if self.reset_noise_scale < 0:
            raise ValueError("reset_noise_scale must be nonnegative")

    @property
    def substeps(self) -> int:
        return int(round(self.ctrl_dt / self.sim_dt))

    @property
    def naconmax(self) -> int:
        return self.naconmax_per_env * self.num_envs


@dataclass(frozen=True, slots=True)
class AntSoccerConfig(AntConfig):
    task_id: int = 4
    spawn_noise_scale: float = 1.0
    add_noise_to_goal: bool = False
    spawn_mode: str = "standard"
    goal_tolerance: float = 0.5
    success_timing: str = "pre"

    def __post_init__(self) -> None:
        AntConfig.__post_init__(self)
        if not 1 <= self.task_id <= 5:
            raise ValueError("task_id must be in [1, 5]")
        if self.spawn_noise_scale < 0:
            raise ValueError("spawn_noise_scale must be nonnegative")
        if self.spawn_mode not in {"standard", "closer_ball", "center_ball", "past_center_ball"}:
            raise ValueError(f"unsupported AntSoccer spawn mode: {self.spawn_mode}")
        if self.goal_tolerance <= 0:
            raise ValueError("goal_tolerance must be positive")
        if self.success_timing not in {"pre", "post"}:
            raise ValueError("success_timing must be 'pre' or 'post'")
