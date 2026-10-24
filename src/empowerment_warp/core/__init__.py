"""Shared typed, device-resident training primitives."""

from empowerment_warp.core.replay import (
    ReplayBuffer,
    ReplayState,
    RepresentationTransition,
    EpisodeArchive,
    EpisodeArchiveState,
    Transition,
)
from empowerment_warp.core.rollout import (
    COVERAGE_TRACKS,
    CollectorState,
    ExplorationMetrics,
    coverage_counts,
    coverage_track_specs,
    initialize_collector,
    make_rollout_chunk,
)

__all__ = [
    "COVERAGE_TRACKS",
    "CollectorState",
    "ExplorationMetrics",
    "ReplayBuffer",
    "ReplayState",
    "RepresentationTransition",
    "EpisodeArchive",
    "EpisodeArchiveState",
    "Transition",
    "coverage_counts",
    "coverage_track_specs",
    "initialize_collector",
    "make_rollout_chunk",
]
