"""metrics.jsonl -> per-seed series -> seed aggregate.

Two rules here exist because breaking either produces a plausible wrong curve:

1. **Union keys across rows, never trust row 0.** Episode-completion metrics
   (`exploration/episode_*`) are absent until the first episodes finish, so the
   first few rows of every run lack them. Reading only the first row concludes
   the metric does not exist. It does.
2. **Complete-case across seeds.** Seeds are aggregated only over the step
   range every seed reached. A mean whose n silently drops partway along the
   x-axis puts a discontinuity in the curve that reads as a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np


DEFAULT_EMA_ALPHA = 0.15
DEFAULT_GRID_POINTS = 400


@dataclass(frozen=True)
class SeedSeries:
    """One seed's (steps, values) for one metric key, NaN rows dropped."""

    seed: int
    key: str
    steps: np.ndarray
    values: np.ndarray

    def __len__(self) -> int:
        return int(self.steps.size)


@dataclass(frozen=True)
class AggregateSeries:
    """Mean +/- standard error across seeds on a shared step grid."""

    key: str
    steps: np.ndarray
    mean: np.ndarray
    stderr: np.ndarray
    per_seed: dict[int, np.ndarray]
    n_seeds: int
    ema_alpha: float
    min_step: float
    max_step: float
    dropped_seeds: tuple[int, ...] = field(default=())

    @property
    def provenance(self) -> dict:
        """Everything needed to regenerate these marks, for the plot_data JSON."""
        return {
            "metric_key": self.key,
            "n_seeds": self.n_seeds,
            "seeds": sorted(self.per_seed),
            "dropped_seeds": list(self.dropped_seeds),
            "ema_alpha": self.ema_alpha,
            "min_step": self.min_step,
            "max_step": self.max_step,
            "grid_points": int(self.steps.size),
            "statistic": "mean +/- standard error across seeds",
            "aggregation": "complete-case on the step range every seed reached",
        }


def read_metric_rows(path: Path | str) -> list[dict]:
    """Parse a metrics.jsonl, skipping blank and truncated trailing lines."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write leaves one partial final line.
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows




def seed_series(
    rows: list[dict],
    key: str,
    *,
    seed: int,
    step_key: str = "env_steps",
) -> SeedSeries | None:
    """Extract one metric as a series, or None when the run never logged it."""
    steps: list[float] = []
    values: list[float] = []
    for row in rows:
        if key not in row or step_key not in row:
            continue
        value = row[key]
        step = row[step_key]
        if value is None or step is None:
            continue
        value = float(value)
        step = float(step)
        if not np.isfinite(value) or not np.isfinite(step):
            continue
        steps.append(step)
        values.append(value)
    if not steps:
        return None
    order = np.argsort(np.asarray(steps, dtype=np.float64), kind="stable")
    return SeedSeries(
        seed=seed,
        key=key,
        steps=np.asarray(steps, dtype=np.float64)[order],
        values=np.asarray(values, dtype=np.float64)[order],
    )


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average, applied PER SEED before aggregation.

    Smoothing after averaging would blur the seed spread into the mean and
    shrink the error band, which is the quantity the reader is judging.
    """
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"ema alpha must be in (0, 1], got {alpha}")
    if alpha == 1.0 or values.size == 0:
        return np.asarray(values, dtype=np.float64)
    smoothed = np.empty_like(values, dtype=np.float64)
    running = float(values[0])
    smoothed[0] = running
    for index in range(1, values.size):
        running = alpha * float(values[index]) + (1.0 - alpha) * running
        smoothed[index] = running
    return smoothed


def aggregate_seeds(
    series: list[SeedSeries],
    *,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    grid_points: int = DEFAULT_GRID_POINTS,
    max_step: float | None = None,
    dropped_seeds: tuple[int, ...] = (),
) -> AggregateSeries:
    """Mean +/- s.e. across seeds, complete-case, on a shared step grid."""
    if not series:
        raise ValueError("cannot aggregate an empty list of seed series")
    key = series[0].key
    if any(item.key != key for item in series):
        raise ValueError("cannot aggregate series for different metric keys")

    # Complete case: only the range EVERY seed covers.
    low = max(float(item.steps[0]) for item in series)
    high = min(float(item.steps[-1]) for item in series)
    if max_step is not None:
        high = min(high, float(max_step))
    if not high > low:
        raise ValueError(
            f"{key}: seeds share no overlapping step range "
            f"(low={low:g}, high={high:g}); they cannot be aggregated"
        )

    grid = np.linspace(low, high, int(grid_points), dtype=np.float64)
    per_seed: dict[int, np.ndarray] = {}
    for item in series:
        smoothed = ema(item.values, ema_alpha)
        per_seed[item.seed] = np.interp(grid, item.steps, smoothed)

    stacked = np.vstack([per_seed[seed] for seed in sorted(per_seed)])
    n_seeds = int(stacked.shape[0])
    mean = stacked.mean(axis=0)
    if n_seeds > 1:
        stderr = stacked.std(axis=0, ddof=1) / np.sqrt(n_seeds)
    else:
        # One seed has no spread to report. Zero would draw a band implying
        # certainty that was never measured.
        stderr = np.full_like(mean, np.nan)

    return AggregateSeries(
        key=key,
        steps=grid,
        mean=mean,
        stderr=stderr,
        per_seed=per_seed,
        n_seeds=n_seeds,
        ema_alpha=float(ema_alpha),
        min_step=float(low),
        max_step=float(high),
        dropped_seeds=tuple(dropped_seeds),
    )


def aggregate_metric(
    runs,
    key: str,
    *,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    grid_points: int = DEFAULT_GRID_POINTS,
    max_step: float | None = None,
    rows_by_seed: dict[int, list[dict]] | None = None,
) -> AggregateSeries | None:
    """Read every seed run and aggregate one metric, or None if nobody logged it."""
    collected: list[SeedSeries] = []
    dropped: list[int] = []
    for run in runs:
        rows = (rows_by_seed or {}).get(run.seed)
        if rows is None:
            rows = read_metric_rows(run.metrics_path)
        item = seed_series(rows, key, seed=run.seed)
        if item is None:
            dropped.append(run.seed)
            continue
        collected.append(item)
    if not collected:
        return None
    return aggregate_seeds(
        collected,
        ema_alpha=ema_alpha,
        grid_points=grid_points,
        max_step=max_step,
        dropped_seeds=tuple(dropped),
    )
