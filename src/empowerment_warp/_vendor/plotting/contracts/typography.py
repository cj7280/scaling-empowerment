"""Typography and text-measurement contracts for pub-style figures."""

from __future__ import annotations

from dataclasses import dataclass, field

from empowerment_warp._vendor.plotting.contracts.core import PubTextStyle, normalize_pub_markdown


def pub_em_inches(fontsize: float, em: float = 1.0) -> float:
    """Return an em-like spacing unit in inches for a point font size."""

    return float(fontsize) * float(em) / 72.0


@dataclass(frozen=True)
class PubTypographySpec:
    """Relative font-size contract for compact publication plots."""

    title: PubTextStyle = field(default_factory=lambda: PubTextStyle(fontsize=8.0, color="#111111"))
    axis_label: PubTextStyle = field(default_factory=lambda: PubTextStyle(fontsize=6.4, color="#111111"))
    tick: PubTextStyle = field(default_factory=lambda: PubTextStyle(fontsize=5.7, color="#111111"))
    legend: PubTextStyle = field(default_factory=lambda: PubTextStyle(fontsize=5.1, color="#111111", ha="left"))
    caption: PubTextStyle = field(default_factory=lambda: PubTextStyle(fontsize=5.5, color="#555555"))

    @classmethod
    def compact_metric(cls, *, axis_label_size: float = 6.4) -> "PubTypographySpec":
        """Return a metric-plot scale derived from the axis-label size."""

        base = float(axis_label_size)
        return cls(
            title=PubTextStyle(fontsize=1.25 * base, color="#111111", fontweight="normal"),
            axis_label=PubTextStyle(fontsize=base, color="#111111"),
            tick=PubTextStyle(fontsize=0.89 * base, color="#111111"),
            legend=PubTextStyle(fontsize=0.80 * base, color="#111111", ha="left"),
            caption=PubTextStyle(fontsize=0.86 * base, color="#555555"),
        )


def pub_measure_text_inches(
    plt,
    text: object,
    *,
    style: PubTextStyle | None = None,
    fontsize: float | None = None,
    rotation: float = 0.0,
    width: int | None = None,
) -> tuple[float, float]:
    """Measure normalized text in physical inches using Matplotlib's renderer."""

    if style is None:
        style = PubTextStyle(fontsize=9.0 if fontsize is None else float(fontsize))
    if fontsize is not None:
        style = PubTextStyle(
            fontsize=float(fontsize),
            color=style.color,
            fontweight=style.fontweight,
            ha=style.ha,
            va=style.va,
            linespacing=style.linespacing,
        )
    fig = plt.figure(figsize=(2.0, 1.0), dpi=200)
    artist = fig.text(
        0.5,
        0.5,
        normalize_pub_markdown(text, width=width),
        fontsize=float(style.fontsize),
        fontweight=style.fontweight,
        linespacing=style.linespacing,
        rotation=float(rotation),
    )
    fig.canvas.draw()
    bbox = artist.get_window_extent(fig.canvas.get_renderer())
    size = (float(bbox.width) / fig.dpi, float(bbox.height) / fig.dpi)
    plt.close(fig)
    return size


def pub_text_band_height_inches(
    plt,
    text: object,
    *,
    style: PubTextStyle,
    pad_em: float = 0.55,
    width: int | None = None,
) -> float:
    """Return the vertical slot height needed for a text band."""

    _, height = pub_measure_text_inches(plt, text, style=style, width=width)
    return float(height) + pub_em_inches(style.fontsize, pad_em)


def pub_axis_label_gutter_inches(
    plt,
    *,
    y_labels: tuple[object, ...] | list[object],
    y_tick_labels: tuple[object, ...] | list[object],
    typography: PubTypographySpec,
    min_gutter_em: float = 5.6,
    pad_em: float = 2.8,
) -> float:
    """Return a left gutter from measured y-label and y-tick text."""

    label_widths = [
        pub_measure_text_inches(plt, label, style=typography.axis_label, rotation=90.0)[0]
        for label in y_labels
    ]
    tick_widths = [
        pub_measure_text_inches(plt, label, style=typography.tick)[0]
        for label in y_tick_labels
    ]
    measured = (max(label_widths) if label_widths else 0.0) + (max(tick_widths) if tick_widths else 0.0)
    min_gutter = pub_em_inches(typography.axis_label.fontsize, min_gutter_em)
    return max(min_gutter, measured + pub_em_inches(typography.axis_label.fontsize, pad_em))
