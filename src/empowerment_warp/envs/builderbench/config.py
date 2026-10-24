"""Typed BuilderBench Corner3 configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BuilderBenchConfig:
    """Static physics and batching choices for Corner3."""

    num_envs: int = 512
    episode_length: int = 200
    sim_dt: float = 0.001
    ctrl_dt: float = 0.01
    integrator: str = "implicitfast"
    solver_iterations: int | None = None
    solver_ls_iterations: int | None = None
    graph_mode: str = "warp"
    naconmax_per_env: int = 64
    njmax_per_env: int = 128
    cube_density: float = 1240.0
    density_choice: bool = False
    centered_start: bool = False
    density_min: float = 124.0
    density_max: float = 12400.0
    workspace_expand_xy: float = 0.05
    wall_restitution: float = 0.0
    corner_x: float = 0.40
    corner_y: float = 0.24
    corner_spacing: float = 0.055
    early_termination: bool = True
    permutation_invariant_reward: bool = True

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not 0.0 < self.density_min < self.density_max:
            raise ValueError("density bounds must satisfy 0 < density_min < density_max")
        if self.cube_density <= 0:
            raise ValueError("cube_density must be positive")
        if self.density_choice and not self.density_min <= self.cube_density <= self.density_max:
            raise ValueError("cube_density must lie inside density choice bounds")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.sim_dt <= 0 or self.ctrl_dt <= 0:
            raise ValueError("simulation and control timesteps must be positive")
        ratio = self.ctrl_dt / self.sim_dt
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("ctrl_dt must be an integer multiple of sim_dt")
        if not 0.0 <= self.wall_restitution <= 1.0:
            raise ValueError("wall_restitution must be in [0, 1]")
        if self.naconmax_per_env <= 0 or self.njmax_per_env <= 0:
            raise ValueError("contact and constraint capacities must be positive")
        if self.graph_mode not in {"none", "warp", "warp_staged", "warp_staged_ex"}:
            raise ValueError(f"unsupported Warp graph mode: {self.graph_mode}")

    @property
    def substeps(self) -> int:
        return int(round(self.ctrl_dt / self.sim_dt))

    @property
    def naconmax(self) -> int:
        return self.naconmax_per_env * self.num_envs
