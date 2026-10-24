"""Environment construction for the continuous (SAC) recipes."""

from __future__ import annotations

from typing import Any

from empowerment_warp.envs.builderbench import BuilderBenchConfig, BuilderBenchWarpEnv
from empowerment_warp.envs.locomotion import (
    AntConfig,
    AntSoccerConfig,
    AntSoccerWarpEnv,
    AntWarpEnv,
)
from empowerment_warp.training.sac.config import ExperimentConfig, SAC_ENVIRONMENTS


def make_environment(config: ExperimentConfig) -> Any:
    name = config.environment
    if name not in SAC_ENVIRONMENTS:
        raise ValueError(f"unknown environment: {name}")
    if not config.allow_unpromoted_physics:
        raise RuntimeError(
            f"{name} Warp status is 'validation'; "
            "set allow_unpromoted_physics only for validation runs"
        )
    if name == "builderbench_corner3":
        return BuilderBenchWarpEnv(
            BuilderBenchConfig(
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
    if name == "ant":
        if config.uses_soccer_dynamics:
            return AntSoccerWarpEnv(
                AntSoccerConfig(
                    num_envs=config.num_envs,
                    episode_length=config.episode_length,
                    spawn_mode=config.antsoccer_preset,
                )
            )
        return AntWarpEnv(
            AntConfig(num_envs=config.num_envs, episode_length=config.episode_length)
        )
    if name == "ant_soccer":
        return AntSoccerWarpEnv(
            AntSoccerConfig(
                num_envs=config.num_envs,
                episode_length=config.episode_length,
                spawn_mode=config.antsoccer_preset,
            )
        )
    raise ValueError(f"SAC supports Ant, AntSoccer, and BuilderBench; got {name}")
