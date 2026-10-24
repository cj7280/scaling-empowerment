"""Run figures using the bundled publication style and rendering contracts."""

from empowerment_warp.plotting.panels import PanelSpec, board_panels
from empowerment_warp.plotting.runs import SeedRun, discover_runs
from empowerment_warp.plotting.series import (
    AggregateSeries,
    SeedSeries,
    aggregate_seeds,
    read_metric_rows,
    seed_series,
)

__all__ = [
    "AggregateSeries",
    "PanelSpec",
    "SeedRun",
    "SeedSeries",
    "aggregate_seeds",
    "board_panels",
    "discover_runs",
    "read_metric_rows",
    "seed_series",
]
