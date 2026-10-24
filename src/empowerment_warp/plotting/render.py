"""One figure spec -> a build at any reading tier.

The tier fixes the figure WIDTH, because the width is a property of the surface
the figure is read on, not a free parameter. Everything else -- gutters, panel
size, band heights -- is then solved inside that width from measured text.

This is the opposite of growing the figure until the text fits. Growing is what
let a long title push a metric stack out to 7.86 in and drop its type scale
below the floor; at a fixed width an over-long title is a loud failure instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from empowerment_warp._vendor.plotting.contracts.typography import (
    pub_axis_label_gutter_inches,
    pub_em_inches,
    pub_measure_text_inches,
    pub_text_band_height_inches,
)
from empowerment_warp._vendor.plotting.plot_style import (
    PUB_TIERS,
    apply_pub_legend_box,
    assert_pub_artists_inside_figure,
    assert_pub_type_scale,
    assert_pub_uniform_text_size,
    add_pub_fixed_axes,
    high_contrast_rgb01,
    pub_body_pt_for_width,
    pub_tier,
    pub_uniform_typography_for_width,
    save_pub_figure,
)


DEFAULT_BOARD_WIDTH_IN = 6.5
DEFAULT_PAPER_WIDTH_IN = 3.3
DEFAULT_PANEL_ASPECT = 0.62      # panel height / panel width
BAND_COLOR_ALPHA = 0.22
REFERENCE_COLOR = "#6b6b6b"


@dataclass(frozen=True)
class TierBuild:
    """A reading surface: how wide the figure is and how big its type is."""

    name: str
    width_in: float
    pt_per_inch: float
    placement_in: float
    floor: float
    aspect: float = DEFAULT_PANEL_ASPECT
    tier_for_assert: str = "paper"

    @property
    def body_pt(self) -> float:
        return pub_body_pt_for_width(self.width_in, pt_per_inch=self.pt_per_inch)

    @property
    def typography(self):
        return pub_uniform_typography_for_width(
            self.width_in, pt_per_inch=self.pt_per_inch
        )

    def describe(self) -> dict:
        return {
            "tier": self.name,
            "figure_width_in": self.width_in,
            "pt_per_fig_inch": self.pt_per_inch,
            "body_pt": self.body_pt,
            "reading_width_in": self.placement_in,
            "ratio_floor": self.floor,
        }


def paper_build(width_in: float = DEFAULT_PAPER_WIDTH_IN, **kwargs) -> TierBuild:
    tier = pub_tier("paper")
    return TierBuild(
        name="paper",
        width_in=float(width_in),
        pt_per_inch=float(tier.pt_per_inch),
        placement_in=float(tier.placement_in),
        floor=float(tier.floor),
        tier_for_assert="paper",
        **kwargs,
    )


def board_build(
    width_in: float = DEFAULT_BOARD_WIDTH_IN,
    *,
    pt_per_inch: float | None = None,
    floor: float | None = None,
    **kwargs,
) -> TierBuild:
    """The "board"-tier (HTML overview) build.

    `pt_per_inch` may be passed explicitly while the board tier is still
    uncalibrated -- that is exactly what the type-scale bake-off does.
    """
    tier = pub_tier("board")
    resolved_pt = pt_per_inch if pt_per_inch is not None else tier.pt_per_inch
    if resolved_pt is None:
        raise ValueError(
            "the board tier has no pt/in yet. Pass pt_per_inch= for a bake-off "
            "rung, or calibrate PUB_TIERS['board'] once a rung has been chosen."
        )
    resolved_floor = floor if floor is not None else tier.floor
    if resolved_floor is None:
        # Before calibration, hold the rung itself as the floor: a bake-off
        # variant must at least be as large as it claims to be.
        resolved_floor = float(resolved_pt) - 1e-9
    return TierBuild(
        name="board",
        width_in=float(width_in),
        pt_per_inch=float(resolved_pt),
        placement_in=float(tier.placement_in),
        floor=float(resolved_floor),
        tier_for_assert="board",
        **kwargs,
    )


def _format_steps(value: float) -> str:
    """Environment steps as a short human label: 0, 250M, 1.0B."""
    value = float(value)
    if value >= 1e9:
        return f"{value / 1e9:.1f}B".replace(".0B", "B")
    if value >= 1e6:
        return f"{value / 1e6:.0f}M"
    if value >= 1e3:
        return f"{value / 1e3:.0f}k"
    return f"{value:.0f}"


def _format_value(value: float, span: float) -> str:
    if span >= 100:
        return f"{value:.0f}"
    if span >= 10:
        return f"{value:.1f}"
    if span >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


@dataclass
class PanelData:
    """Everything one panel draws, already aggregated.

    `arms` turns the panel into a comparison. It stays EMPTY for the ordinary
    one-series case, so a single-arm figure renders exactly as before. When it
    is set, `aggregate` is arm 0 -- the axis solve and the reference line still
    read from it, and every arm is drawn.

    An arm is not a seed. Seeds are averaged inside one `AggregateSeries`
    (complete-case, so the band's n never changes mid-curve); arms are separate
    configurations and are aggregated separately, so two arms may legitimately
    end at different steps. That is what makes this honest for LIVE runs.
    """

    spec: object              # PanelSpec
    aggregate: object         # AggregateSeries -- arm 0
    color_index: int = 0
    arms: tuple[tuple[str, object], ...] = ()   # (label, AggregateSeries)

    @property
    def all_aggregates(self) -> list:
        """Every series this panel draws, arm 0 first."""
        if not self.arms:
            return [self.aggregate]
        return [aggregate for _label, aggregate in self.arms]


@dataclass
class MetricFigure:
    """A solved figure: the Matplotlib objects plus the provenance to save."""

    fig: object
    axes: list
    build: TierBuild
    plot_data: dict = field(default_factory=dict)
    # A figure-level artist sits outside `finish`'s per-axes containment sweep,
    # so anything drawn on the FIGURE has to be handed over explicitly or it can
    # overflow the page unchecked.
    legend_artist: object = None


def _y_tick_label_candidates(panels: list[PanelData]) -> list[str]:
    labels: list[str] = []
    for panel in panels:
        lows, highs = [], []
        for aggregate in panel.all_aggregates:
            lows.append(float(np.nanmin(aggregate.mean)))
            highs.append(float(np.nanmax(aggregate.mean)))
        low, high = min(lows), max(highs)
        reference = panel.spec.resolved_reference()
        if reference is not None:
            low = min(low, reference[0])
            high = max(high, reference[0])
        span = max(high - low, 1e-9)
        labels += [_format_value(low, span), _format_value(high, span)]
    return labels


def _x_tick_label_candidates(panels: list[PanelData]) -> list[str]:
    """The x tick strings that will actually be drawn, for margin measurement."""
    labels: list[str] = []
    for panel in panels:
        for aggregate in panel.all_aggregates:
            steps = aggregate.steps
            labels += [_format_steps(float(steps[0])), _format_steps(float(steps[-1]))]
    return labels


def fit_text_to_width(plt, text: str, *, style, max_width_in: float) -> str:
    """Wrap `text` onto as few lines as fit within `max_width_in`.

    At a fixed page width, a title that does not fit must WRAP. Widening the
    page instead is what dropped the metric stack below its type-scale floor,
    and shrinking the type is what this whole contract exists to prevent.
    """
    import textwrap

    words = str(text).split()
    if not words:
        return str(text)
    for lines in range(1, len(words) + 1):
        # Aim for `lines` roughly equal lines, then measure the real result.
        budget = max(1, -(-len(str(text)) // lines))
        for width in range(budget, len(str(text)) + 1):
            # `break_long_words=False`: a run identifier is one token and must
            # never be split mid-word, which would read as two run names.
            candidate = "\n".join(
                textwrap.wrap(
                    str(text), width=width,
                    break_long_words=False, break_on_hyphens=False,
                )
            )
            if candidate.count("\n") + 1 > lines:
                continue
            measured = pub_measure_text_inches(plt, candidate, style=style)[0]
            if measured <= max_width_in:
                return candidate
            break
    # Nothing fits even one word per line; hand back the tightest wrap and let
    # the containment assert reject it loudly rather than shipping an overflow.
    return "\n".join(words)


def _nullable(values: np.ndarray) -> list:
    return [float(value) if np.isfinite(value) else None for value in values]


def render_metric_stack(
    plt,
    panels: list[PanelData],
    build: TierBuild,
    *,
    ncols: int = 1,
) -> MetricFigure:
    """Lay out N panels in a grid inside the tier's FIXED figure width.

    `ncols` defaults to 1 because a board figure stacks. At board type scale the
    y-label gutter is ~1.8 in per column, so two columns of labelled panels do
    not fit in a 6.5 in page at all -- the solver says so rather than silently
    shrinking the type or widening the page. A vertically scrolling HTML board
    wants a stack anyway.

    Only the bottom panel of each column carries the x-axis label: repeating
    "Environment steps" under every panel spends vertical space on a string the
    reader has already read.
    """
    if not panels:
        raise ValueError("render_metric_stack needs at least one panel")
    ncols = max(int(ncols), 1)
    nrows = -(-len(panels) // ncols)
    orphans = nrows * ncols - len(panels)
    if orphans and ncols > 1:
        raise ValueError(
            f"{len(panels)} panels into {ncols} columns leaves {orphans} empty "
            f"cell(s) of dead canvas. Pick a column count that fills the last row."
        )

    typography = build.typography
    body_pt = build.body_pt
    em = pub_em_inches(body_pt)

    gutter = pub_axis_label_gutter_inches(
        plt,
        y_labels=[panel.spec.ylabel for panel in panels],
        y_tick_labels=_y_tick_label_candidates(panels),
        typography=typography,
    )
    # The LAST x tick label is centred on the panel's right edge, so half of it
    # hangs past the data area. A fixed em margin does not know how wide "500M"
    # is at this type size, and the containment assert catches the overhang --
    # so derive the margin from the measured label instead of guessing.
    half_last_tick = 0.5 * max(
        pub_measure_text_inches(plt, label, style=typography.tick)[0]
        for label in _x_tick_label_candidates(panels)
    )
    right_margin = max(0.9 * em, half_last_tick + 0.25 * em)
    inter_col = 1.1 * em if ncols > 1 else 0.0

    panel_w = (
        build.width_in - ncols * gutter - right_margin - (ncols - 1) * inter_col
    ) / ncols
    min_panel_w = 6.0 * em
    if panel_w < min_panel_w:
        raise ValueError(
            f"{ncols} column(s) of {body_pt:.1f} pt type do not fit in a "
            f"{build.width_in:.2f} in figure: each panel would be "
            f"{panel_w:.2f} in, below the {min_panel_w:.2f} in minimum "
            f"(the y-label gutter alone is {gutter:.2f} in). "
            f"Use fewer columns, a wider tier, or a smaller pt/in -- "
            f"do NOT widen the figure past its reading width."
        )
    panel_h = panel_w * build.aspect

    title_budget = panel_w + right_margin
    titles = [
        fit_text_to_width(
            plt, panel.spec.title, style=typography.title, max_width_in=title_budget
        )
        for panel in panels
    ]
    title_h = max(
        pub_text_band_height_inches(plt, title, style=typography.title, pad_em=0.45)
        for title in titles
    )
    xlabel_h = pub_text_band_height_inches(
        plt, "Environment steps", style=typography.axis_label, pad_em=0.4
    )
    tick_h = pub_measure_text_inches(plt, "250M", style=typography.tick)[1]
    # Every row pays for its tick labels; only the bottom row pays for the
    # x-axis label, so it is added once outside the per-row height.
    row_h = title_h + panel_h + tick_h + 0.45 * em
    # 0.15 em of rounding headroom for the descender.
    #
    # NOTE the band heights above are only trustworthy because every wrapped
    # string below is DRAWN with the same `linespacing` it was MEASURED with.
    # `PubTextStyle.linespacing` is 1.0 and Matplotlib's default is 1.2, so a
    # text drawn without it stands 20% taller than the band solved for it, and
    # the containment assert reports the overflow. Pass `linespacing=` on every
    # multi-line text artist here.
    bottom = 0.55 * em + xlabel_h + 0.15 * em
    top_margin = 0.5 * em

    # Every panel compares the SAME arms, so one legend serves the figure. It
    # sits in its own band above the panels rather than inside an axes: a box
    # floating over a curve hides the data.
    arm_labels = [label for label, _ in (panels[0].arms or ())]
    legend_h = (
        len(arm_labels) * 1.9 * em + 1.1 * em if len(arm_labels) > 1 else 0.0
    )

    fig_h = bottom + nrows * row_h + legend_h + top_margin
    fig = plt.figure(figsize=(build.width_in, fig_h))

    palette = high_contrast_rgb01(max(len(panels), 3))
    axes = []
    for index, panel in enumerate(panels):
        row, col = divmod(index, ncols)
        # Row 0 is the TOP row, so its y origin is the highest.
        rows_below = nrows - 1 - row
        y0 = bottom + rows_below * row_h + tick_h + 0.45 * em
        x0 = gutter + col * (panel_w + gutter + inter_col)
        ax = add_pub_fixed_axes(fig, build.width_in, fig_h, x0, y0, panel_w, panel_h)
        handles = _draw_panel(
            ax,
            panel,
            color=palette[panel.color_index % len(palette)],
            typography=typography,
            body_pt=body_pt,
            show_xlabel=(row == nrows - 1),
        )
        ax.set_title(
            titles[index],
            fontsize=body_pt,
            color=typography.title.color,
            linespacing=typography.title.linespacing,
        )
        axes.append(ax)

    if legend_h:
        legend_ax = fig.add_axes(
            [gutter / build.width_in,
             (bottom + nrows * row_h) / fig_h,
             panel_w / build.width_in,
             legend_h / fig_h],
        )
        legend_ax.axis("off")
        legend = legend_ax.legend(
            handles=handles, loc="center left", ncol=1,
            fontsize=body_pt, frameon=True, borderaxespad=0.0,
        )
        apply_pub_legend_box(legend, style="white")
        for text in legend.get_texts():
            text.set_fontsize(body_pt)
        legend_artist = legend
    else:
        legend_artist = None

    plot_data = {
        "build": build.describe(),
        "figure_size_in": [build.width_in, fig_h],
        "panel_size_in": [panel_w, panel_h],
        "grid": {"nrows": nrows, "ncols": ncols},
        "left_gutter_in": gutter,
        "arms": arm_labels,
        "panels": [
            {
                "metric_key": panel.spec.key,
                "title": panel.spec.title,
                "ylabel": panel.spec.ylabel,
                "tier": panel.spec.tier,
                "reference": panel.spec.reference,
                "reference_label": panel.spec.reference_label,
                **panel.aggregate.provenance,
                "steps": panel.aggregate.steps.tolist(),
                "mean": panel.aggregate.mean.tolist(),
                "stderr": _nullable(panel.aggregate.stderr),
                "per_seed": {
                    str(seed): values.tolist()
                    for seed, values in sorted(panel.aggregate.per_seed.items())
                },
                "arms": [
                    {
                        "label": label,
                        **aggregate.provenance,
                        "steps": aggregate.steps.tolist(),
                        "mean": aggregate.mean.tolist(),
                        "stderr": _nullable(aggregate.stderr),
                    }
                    for label, aggregate in panel.arms
                ],
            }
            for panel in panels
        ],
    }
    return MetricFigure(
        fig=fig, axes=axes, build=build, plot_data=plot_data,
        legend_artist=legend_artist,
    )


def _axes_height_inches(ax) -> float:
    """The axes' drawn height, in inches."""
    figure = ax.get_figure()
    return float(ax.get_position().height) * float(figure.get_figheight())


def plt_of(ax):
    """The pyplot module, reached from an artist rather than re-imported."""
    import matplotlib.pyplot as plt

    return plt


def _draw_panel(
    ax,
    panel: PanelData,
    *,
    color,
    typography,
    body_pt: float,
    show_xlabel: bool = True,
) -> None:
    from matplotlib.ticker import FuncFormatter

    # One arm or several. The single-arm path is the `arms == ()` case and is
    # unchanged: same colour, same z-order, same band.
    if panel.arms:
        drawn = [(label, aggregate) for label, aggregate in panel.arms]
        arm_colors = high_contrast_rgb01(max(len(drawn), 3))
    else:
        drawn = [(None, panel.aggregate)]
        arm_colors = [color]

    handles = []
    lows, highs = [], []
    for index, (label, aggregate) in enumerate(drawn):
        steps = aggregate.steps
        mean = aggregate.mean
        stderr = aggregate.stderr
        arm_color = arm_colors[index % len(arm_colors)]

        # A single seed has no standard error to draw. `aggregate_seeds` reports
        # that as a non-finite band rather than a zero-width one, so a band of
        # zero is never mistaken for seeds that agreed.
        if np.all(np.isfinite(stderr)):
            ax.fill_between(
                steps,
                mean - stderr,
                mean + stderr,
                color=arm_color,
                alpha=BAND_COLOR_ALPHA,
                linewidth=0.0,
            )
            lows.append(float(np.nanmin(mean - stderr)))
            highs.append(float(np.nanmax(mean + stderr)))
        else:
            lows.append(float(np.nanmin(mean)))
            highs.append(float(np.nanmax(mean)))
        line = ax.plot(
            steps, mean, color=arm_color, linewidth=1.9, solid_capstyle="round",
            label=label,
        )[0]
        if label:
            handles.append(line)

    # The axis solve reads arm 0's last point for the reference-label side, so
    # keep arm 0's series in scope under the names the code below expects.
    steps = drawn[0][1].steps
    mean = drawn[0][1].mean
    low_data, high_data = min(lows), max(highs)

    reference = panel.spec.resolved_reference()
    if reference is not None:
        value, label = reference
        low_data = min(low_data, value)
        high_data = max(high_data, value)

    span = max(high_data - low_data, 1e-9)
    pad = 0.08 * span
    if reference is not None:
        value, _label = reference
        if value >= high_data - 1e-12:
            high_data = value + pad
        if value <= low_data + 1e-12:
            low_data = value - pad
    ax.set_ylim(low_data - pad, high_data + pad)

    if reference is not None:
        value, label = reference
        ax.axhline(value, color=REFERENCE_COLOR, linewidth=1.1, linestyle=(0, (4, 3)))
        # Which side of the line the label goes on has TWO constraints, and
        # honouring only the first is what put "chance 1/16" through the tick
        # row: it must not land on the curve, and it must fit inside the axes.
        #
        # A reference near the axis floor has no room below it however far the
        # curve is above -- so the preferred side is the one the curve is not
        # on, and it is overruled when that side has less clearance than the
        # label is tall.
        axis_low, axis_high = ax.get_ylim()
        axis_span = max(axis_high - axis_low, 1e-12)
        # The label's height as a fraction of the DATA range, via the axes'
        # physical height -- a fixed em guess does not know how tall this axis is.
        label_h = pub_measure_text_inches(
            plt_of(ax), label, style=typography.tick)[1]
        label_frac = label_h / max(_axes_height_inches(ax), 1e-6)
        room_below = (value - axis_low) / axis_span
        room_above = (axis_high - value) / axis_span

        below = float(mean[-1]) > value          # preferred: away from the curve
        if below and room_below < label_frac and room_above >= label_frac:
            below = False
        elif not below and room_above < label_frac and room_below >= label_frac:
            below = True

        ax.text(
            steps[-1],
            value,
            f"{label} ",
            fontsize=body_pt,
            color=REFERENCE_COLOR,
            ha="right",
            va="top" if below else "bottom",
        )

    if show_xlabel:
        ax.set_xlabel(
            "Environment steps", fontsize=body_pt, color=typography.axis_label.color
        )
    ax.set_ylabel(panel.spec.ylabel, fontsize=body_pt, color=typography.axis_label.color)
    ax.tick_params(axis="both", labelsize=body_pt, colors=typography.tick.color)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _format_steps(value)))

    # Format the y ticks from the AXIS range, not from the curve's own spread.
    # Using the curve's span printed "3.000" on an axis that runs 0 to 3.4,
    # because the MI curve itself only moves by 0.15.
    axis_low, axis_high = ax.get_ylim()
    axis_span = float(axis_high - axis_low)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: _format_value(value, max(axis_span, 1e-9)))
    )
    ax.margins(x=0.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=4)
    return handles


def finish(figure: MetricFigure, output_base: Path | str) -> list[Path]:
    """Assert the contract, then save PDF + PNG + SVG and the plot_data JSON."""
    fig = figure.fig
    build = figure.build

    assert_pub_type_scale(
        fig,
        build.body_pt,
        floor=build.floor,
        placement_in=build.placement_in,
        tier=build.tier_for_assert if build.tier_for_assert in PUB_TIERS else "paper",
    )
    assert_pub_uniform_text_size(fig, build.body_pt)

    # Matplotlib keeps spare tick artists parked far outside the axes; they are
    # never drawn but still report a window extent, so checking them raises a
    # spill that does not exist. Only ticks inside the axis limits are real.
    checked = []
    for ax in figure.axes:
        checked += [ax, ax.xaxis.label, ax.yaxis.label, ax.title]
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(float(value) for value in limits)
            for tick, label in zip(axis.get_ticklocs(), axis.get_ticklabels()):
                if low - 1e-9 <= float(tick) <= high + 1e-9 and label.get_text().strip():
                    checked.append(label)
    if figure.legend_artist is not None:
        checked += list(figure.legend_artist.get_texts())
    assert_pub_artists_inside_figure(fig, checked)

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = save_pub_figure(fig, output_base)

    data_path = output_base.with_name(output_base.name + "_plot_data.json")
    data_path.write_text(json.dumps(figure.plot_data, indent=2, sort_keys=True) + "\n")
    return [*paths, data_path]
