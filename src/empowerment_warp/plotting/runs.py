"""Discover the seed runs that make up one experiment group."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


ARTIFACT_RUN_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "runs"

_SEED_DIR = re.compile(r"^seed(?P<seed>\d+)_(?P<job>[^_]+)(?:_(?P<task>\d+))?$")


@dataclass(frozen=True)
class SeedRun:
    """One seed's run folder, with its resolved config."""

    path: Path
    group: str
    environment: str
    seed: int
    job: str
    config: dict

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.jsonl"

    @property
    def evaluation_path(self) -> Path:
        return self.path / "evaluation" / "deterministic.json"

    def get(self, key: str, default=None):
        return self.config.get(key, default)


def discover_runs(
    group: str,
    environment: str,
    *,
    root: Path | str = ARTIFACT_RUN_ROOT,
) -> list[SeedRun]:
    """Return every seed run in `<root>/<group>/<environment>/`, seed-ordered.

    A seed folder with no `metrics.jsonl` is skipped rather than silently
    contributing an empty curve: a run that never logged is not a seed of zero,
    it is a seed that is missing, and the caller must see a smaller n.
    """
    group_dir = Path(root) / group / environment
    if not group_dir.is_dir():
        raise FileNotFoundError(f"no such run group: {group_dir}")

    runs: list[SeedRun] = []
    for candidate in sorted(group_dir.iterdir()):
        if not candidate.is_dir():
            continue
        match = _SEED_DIR.match(candidate.name)
        if match is None:
            continue
        if not (candidate / "metrics.jsonl").is_file():
            continue
        config_path = candidate / "config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        runs.append(
            SeedRun(
                path=candidate,
                group=group,
                environment=environment,
                seed=int(match.group("seed")),
                job=str(match.group("job")),
                config=config,
            )
        )
    if not runs:
        raise FileNotFoundError(f"no seed runs with metrics.jsonl under {group_dir}")
    return sorted(runs, key=lambda run: run.seed)
