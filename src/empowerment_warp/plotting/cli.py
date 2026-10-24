"""Render the metric-overview ("board" tier) figures for one run group.

    python -m empowerment_warp.plotting.cli <RUN_GROUP> \
        --env ant_soccer --tier board --pt-per-inch 3.6 --output-dir <DIR>

Publication styling uses the bundled plotting helpers.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _bootstrap_matplotlib():
    """Cluster-safe font/cache setup BEFORE Matplotlib is imported."""
    from empowerment_warp._vendor.plotting.plot_style import apply_pub_style, configure_rendering_environment

    configure_rendering_environment()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_pub_style(plt, compact=True, use_tex=False)
    return plt


def build_figure(
    plt,
    *,
    group: str,
    environment: str,
    build,
    max_step: float | None = None,
    ema_alpha: float = 0.15,
    keys: list[str] | None = None,
    ncols: int = 1,
    run_root: Path | None = None,
):
    """Aggregate the group's seeds and lay out its board panels."""
    from empowerment_warp.plotting.panels import board_panels
    from empowerment_warp.plotting.render import PanelData, render_metric_stack
    from empowerment_warp.plotting.runs import ARTIFACT_RUN_ROOT, discover_runs
    from empowerment_warp.plotting.series import aggregate_metric, read_metric_rows

    runs = discover_runs(
        group, environment, root=run_root or ARTIFACT_RUN_ROOT
    )
    num_skills = int(runs[0].get("num_skills", 30))
    specs = board_panels(environment, num_skills=num_skills)
    if keys:
        wanted = set(keys)
        specs = [spec for spec in specs if spec.key in wanted]
        if not specs:
            raise ValueError(f"none of {sorted(wanted)} is a board panel for {environment}")

    # Read each seed's metrics ONCE, not once per panel.
    rows_by_seed = {run.seed: read_metric_rows(run.metrics_path) for run in runs}

    panels = []
    missing = []
    for index, spec in enumerate(specs):
        if spec.terminal_only:
            continue
        aggregate = aggregate_metric(
            runs,
            spec.key,
            ema_alpha=ema_alpha,
            max_step=max_step,
            rows_by_seed=rows_by_seed,
        )
        if aggregate is None:
            missing.append(spec.key)
            continue
        panels.append(PanelData(spec=spec, aggregate=aggregate, color_index=index))

    if not panels:
        raise ValueError(f"no panel in {group}/{environment} had any data")

    figure = render_metric_stack(plt, panels, build, ncols=ncols)
    figure.plot_data["group"] = group
    figure.plot_data["environment"] = environment
    figure.plot_data["num_skills"] = num_skills
    figure.plot_data["missing_metric_keys"] = missing
    figure.plot_data["seed_runs"] = [
        {"seed": run.seed, "job": run.job, "path": str(run.path)} for run in runs
    ]
    return figure, runs, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group")
    parser.add_argument("--env", required=True)
    parser.add_argument("--tier", choices=("board", "paper"), default="board")
    parser.add_argument("--width-in", type=float, default=None)
    parser.add_argument("--pt-per-inch", type=float, default=None)
    parser.add_argument("--max-step", type=float, default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.15)
    parser.add_argument("--key", action="append", dest="keys")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ncols", type=int, default=1)
    parser.add_argument("--stem", default=None)
    args = parser.parse_args(argv)

    plt = _bootstrap_matplotlib()
    from empowerment_warp.plotting.render import (
        DEFAULT_BOARD_WIDTH_IN,
        DEFAULT_PAPER_WIDTH_IN,
        board_build,
        finish,
        paper_build,
    )

    if args.tier == "board":
        build = board_build(
            args.width_in or DEFAULT_BOARD_WIDTH_IN, pt_per_inch=args.pt_per_inch
        )
    else:
        build = paper_build(args.width_in or DEFAULT_PAPER_WIDTH_IN)

    figure, runs, missing = build_figure(
        plt,
        group=args.group,
        environment=args.env,
        build=build,
        max_step=args.max_step,
        ema_alpha=args.ema_alpha,
        keys=args.keys,
        ncols=args.ncols,
    )
    stem = args.stem or f"{args.group}__{args.env}__{build.name}"
    paths = finish(figure, Path(args.output_dir) / stem)

    print(f"group        {args.group}/{args.env}")
    print(f"seeds        n={len(runs)}  {[run.seed for run in runs]}")
    print(f"build        {build.describe()}")
    if missing:
        print(f"MISSING      {missing}")
    for path in paths:
        print(f"wrote        {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
