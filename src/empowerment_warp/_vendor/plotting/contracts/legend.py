"""Legend contracts for centered, measured pub-style legends."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

from empowerment_warp._vendor.plotting.contracts.core import PubBounds
from empowerment_warp._vendor.plotting.contracts.typography import pub_em_inches, pub_measure_text_inches


@dataclass(frozen=True)
class PubLegendEntry:
    """A serializable legend entry for table-like legend bands."""

    label: str
    kind: str = "line"
    color: object = "#111111"
    linestyle: str = "-"
    linewidth: float = 1.4
    marker: object = "o"
    markerfacecolor: object | None = None
    markeredgecolor: object = "#111111"
    markeredgewidth: float = 0.9
    markersize: float = 7.0

    def as_grid_entry(self) -> dict[str, object]:
        """Return the dictionary shape consumed by draw_pub_grid_legend."""

        entry = {
            "label": self.label,
            "kind": self.kind,
            "color": self.color,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "marker": self.marker,
            "markeredgecolor": self.markeredgecolor,
            "markeredgewidth": self.markeredgewidth,
            "markersize": self.markersize,
        }
        entry["markerfacecolor"] = self.color if self.markerfacecolor is None else self.markerfacecolor
        return entry


@dataclass(frozen=True)
class PubLegendSpec:
    """Legend style and spacing contract.

    Matplotlib's legend lengths and pads are in font-size units, so these
    values are intentionally kept relative to the legend font.
    """

    fontsize: float = 5.1
    boxed: bool = False
    box_style: str = "white"
    max_columns: int = 5
    handlelength: float = 1.65
    handletextpad: float = 0.56
    columnspacing: float = 1.05
    borderpad: float = 0.50
    labelspacing: float = 0.42
    borderaxespad: float = 0.0
    row_height_em: float = 1.35
    extra_width_em: float = 0.55
    extra_height_em: float = 0.80

    @classmethod
    def unboxed(cls, **kwargs) -> "PubLegendSpec":
        """Return the default external legend contract: centered, no frame."""

        kwargs.setdefault("boxed", False)
        return cls(**kwargs)

    @classmethod
    def boxed_white(cls, **kwargs) -> "PubLegendSpec":
        """Return a measured legend contract with the standard white frame."""

        kwargs.setdefault("boxed", True)
        kwargs.setdefault("box_style", "white")
        return cls(**kwargs)

    @classmethod
    def boxed_in_axes(cls, **kwargs) -> "PubLegendSpec":
        """Return a roomy boxed legend contract for legends inside plot axes."""

        kwargs.setdefault("boxed", True)
        kwargs.setdefault("box_style", "white")
        kwargs.setdefault("borderpad", 0.60)
        kwargs.setdefault("labelspacing", 0.50)
        kwargs.setdefault("handletextpad", 0.64)
        kwargs.setdefault("borderaxespad", 0.62)
        kwargs.setdefault("handlelength", 1.65)
        return cls(**kwargs)

    @property
    def resolved_borderpad(self) -> float:
        return float(self.borderpad if self.boxed else 0.0)


@dataclass(frozen=True)
class PubLegendGeometry:
    """Resolved legend rows, columns, and physical size in inches."""

    ncol: int
    rows: int
    required_width: float
    height: float
    spec: PubLegendSpec


def _entry_width_inches(plt, label: object, *, spec: PubLegendSpec) -> float:
    label_w, _ = pub_measure_text_inches(plt, label, fontsize=spec.fontsize)
    font_unit = pub_em_inches(spec.fontsize)
    return spec.handlelength * font_unit + spec.handletextpad * font_unit + label_w


def pub_resolve_legend_geometry(
    plt,
    labels: Iterable[object],
    *,
    target_width: float,
    spec: PubLegendSpec,
) -> PubLegendGeometry:
    """Choose legend columns and size from measured labels and target width."""

    labels = [str(label) for label in labels]
    if not labels:
        return PubLegendGeometry(ncol=1, rows=1, required_width=0.0, height=0.0, spec=spec)
    widths = [_entry_width_inches(plt, label, spec=spec) for label in labels]
    font_unit = pub_em_inches(spec.fontsize)
    best: tuple[int, int, float] | None = None
    for ncol in range(min(int(spec.max_columns), len(widths)), 0, -1):
        rows = int(ceil(len(widths) / float(ncol)))
        row_widths = []
        for row in range(rows):
            row_entries = widths[row * ncol : (row + 1) * ncol]
            row_width = sum(row_entries) + max(0, len(row_entries) - 1) * spec.columnspacing * font_unit
            row_widths.append(row_width)
        required_width = max(row_widths) + 2.0 * spec.resolved_borderpad * font_unit
        required_width += spec.extra_width_em * font_unit
        best = (ncol, rows, required_width)
        if required_width <= float(target_width):
            break
    if best is None:
        raise ValueError("Could not resolve legend geometry.")
    ncol, rows, required_width = best
    height = (
        rows * spec.row_height_em
        + max(0, rows - 1) * spec.labelspacing
        + 2.0 * spec.resolved_borderpad
    ) * font_unit
    height += spec.extra_height_em * font_unit
    return PubLegendGeometry(
        ncol=int(ncol),
        rows=int(rows),
        required_width=float(required_width),
        height=float(height),
        spec=spec,
    )


def draw_pub_external_legend(
    fig,
    handles,
    labels,
    *,
    bounds: PubBounds,
    fig_w: float,
    fig_h: float,
    spec: PubLegendSpec,
    geometry: PubLegendGeometry | None = None,
    loc: str = "lower center",
    handler_map=None,
):
    """Draw a centered figure legend in a reserved external bounds."""

    from empowerment_warp._vendor.plotting.plot_style import apply_pub_legend_alignment, apply_pub_legend_box, pub_legend_kwargs

    labels = list(labels)
    if geometry is None:
        geometry = PubLegendGeometry(
            ncol=min(len(labels), spec.max_columns),
            rows=1,
            required_width=bounds.w,
            height=bounds.h,
            spec=spec,
        )
    legend = fig.legend(
        handles,
        labels,
        handler_map=handler_map,
        **pub_legend_kwargs(
            loc=loc,
            bbox_to_anchor=bounds.as_mpl_rect(fig_w, fig_h),
            bbox_transform=fig.transFigure,
            ncol=geometry.ncol,
            frameon=bool(spec.boxed),
            fontsize=spec.fontsize,
            handlelength=spec.handlelength,
            columnspacing=spec.columnspacing,
            borderpad=spec.resolved_borderpad,
            handletextpad=spec.handletextpad,
            labelspacing=spec.labelspacing,
            borderaxespad=spec.borderaxespad,
        ),
    )
    legend.set_in_layout(False)
    if spec.boxed:
        apply_pub_legend_box(legend, style=spec.box_style)
    else:
        frame = legend.get_frame()
        frame.set_facecolor("none")
        frame.set_edgecolor("none")
        frame.set_linewidth(0.0)
        frame.set_alpha(0.0)
        for text in legend.get_texts():
            text.set_color("#111111")
    return apply_pub_legend_alignment(legend)


def draw_pub_in_axes_legend(
    ax,
    *,
    spec: PubLegendSpec | None = None,
    loc: str = "best",
    handles=None,
    labels=None,
    **kwargs,
):
    """Draw a roomy, font-relative legend inside one Matplotlib axes.

    Matplotlib legend padding and axes-border insets are measured in legend-font
    units. This helper keeps small in-axes legends from pinching text/handles or
    sitting flush against a plot corner while preserving native placement.
    """

    from empowerment_warp._vendor.plotting.plot_style import apply_pub_legend_alignment, apply_pub_legend_box, pub_legend_kwargs

    if spec is None:
        spec = PubLegendSpec.boxed_in_axes()
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    filtered = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label and not str(label).startswith("_")
    ]
    if not filtered:
        return None
    handles, labels = zip(*filtered)
    legend = ax.legend(
        handles,
        labels,
        **pub_legend_kwargs(
            loc=loc,
            frameon=bool(spec.boxed),
            fontsize=spec.fontsize,
            handlelength=spec.handlelength,
            columnspacing=spec.columnspacing,
            borderpad=spec.resolved_borderpad,
            handletextpad=spec.handletextpad,
            labelspacing=spec.labelspacing,
            borderaxespad=spec.borderaxespad,
            **kwargs,
        ),
    )
    if spec.boxed:
        apply_pub_legend_box(legend, style=spec.box_style)
    else:
        frame = legend.get_frame()
        frame.set_facecolor("none")
        frame.set_edgecolor("none")
        frame.set_linewidth(0.0)
        frame.set_alpha(0.0)
        for text in legend.get_texts():
            text.set_color("#111111")
    return apply_pub_legend_alignment(legend)


def draw_pub_legend_band(
    ax,
    entries: Iterable[PubLegendEntry | dict[str, object]],
    *,
    spec: PubLegendSpec,
    ncol: int,
    center_entries: bool = True,
) -> dict[str, object]:
    """Draw a table-like legend band with optional frame visibility."""

    from empowerment_warp._vendor.plotting.plot_style import draw_pub_grid_legend

    grid_entries = [entry.as_grid_entry() if isinstance(entry, PubLegendEntry) else dict(entry) for entry in entries]
    result = draw_pub_grid_legend(
        ax,
        grid_entries,
        ncol=ncol,
        style=spec.box_style,
        fontsize=spec.fontsize,
        center_entries=center_entries,
        frame_linewidth=None if spec.boxed else 0.0,
    )
    if not spec.boxed:
        frame = result["frame"]
        frame.set_facecolor("none")
        frame.set_edgecolor("none")
        frame.set_alpha(0.0)
        frame.set_linewidth(0.0)
    return result
