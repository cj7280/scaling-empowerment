"""Small class-first contracts for reusable plotting geometry.

The shared plotting repo favors objects that name the geometry they own:
figure/page bounds, title bands, data panels, legends, and fixed text slots.
These helpers keep that contract lightweight so specialized plots can share
the same boundary and text-normalization rules without growing one-off layout
math in every example.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import textwrap
from typing import Iterable, Literal


_MATH_SPAN_RE = re.compile(r"(?<!\\)(\$[^$]*\$)")


@dataclass(frozen=True)
class PubBounds:
    """A rectangle in either figure inches or axes-fraction units."""

    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return float(self.x)

    @property
    def right(self) -> float:
        return float(self.x + self.w)

    @property
    def bottom(self) -> float:
        return float(self.y)

    @property
    def top(self) -> float:
        return float(self.y + self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (float(self.x + 0.5 * self.w), float(self.y + 0.5 * self.h))

    def inset(self, dx: float = 0.0, dy: float | None = None) -> "PubBounds":
        """Return a rectangle inset on all sides."""

        if dy is None:
            dy = dx
        dx = float(dx)
        dy = float(dy)
        return PubBounds(self.x + dx, self.y + dy, self.w - 2.0 * dx, self.h - 2.0 * dy)

    def as_mpl_rect(self, fig_w: float, fig_h: float) -> list[float]:
        """Return a Matplotlib add_axes rectangle from fixed-inch geometry."""

        fig_w = float(fig_w)
        fig_h = float(fig_h)
        if fig_w <= 0.0 or fig_h <= 0.0:
            raise ValueError("Figure width and height must be positive.")
        return [self.x / fig_w, self.y / fig_h, self.w / fig_w, self.h / fig_h]

    def split_cols(self, n: int, *, gap: float = 0.0) -> tuple["PubBounds", ...]:
        """Split this rectangle into fixed-width columns."""

        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive.")
        gap = float(gap)
        cell_w = (self.w - gap * (n - 1)) / n
        if cell_w <= 0.0:
            raise ValueError("Column gap leaves no usable width.")
        return tuple(PubBounds(self.x + i * (cell_w + gap), self.y, cell_w, self.h) for i in range(n))

    def split_rows(self, n: int, *, gap: float = 0.0, top_to_bottom: bool = False) -> tuple["PubBounds", ...]:
        """Split this rectangle into fixed-height rows."""

        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive.")
        gap = float(gap)
        cell_h = (self.h - gap * (n - 1)) / n
        if cell_h <= 0.0:
            raise ValueError("Row gap leaves no usable height.")
        rows = tuple(PubBounds(self.x, self.y + i * (cell_h + gap), self.w, cell_h) for i in range(n))
        return tuple(reversed(rows)) if top_to_bottom else rows


@dataclass(frozen=True)
class PubTextStyle:
    """Text style for a fixed plotting slot."""

    fontsize: float = 9.0
    color: str = "#111111"
    fontweight: str = "normal"
    ha: Literal["left", "center", "right"] = "center"
    va: Literal["bottom", "center", "top"] = "center"
    linespacing: float = 1.0


def normalize_pub_markdown(text: object, *, width: int | None = None) -> str:
    """Normalize display text before handing it to Matplotlib or Manim.

    This is intentionally not a Markdown renderer. It strips lightweight
    emphasis/code markers from prose, preserves ``$...$`` math spans, collapses
    prose whitespace, and optionally wraps non-math lines. Plotting code should
    pass explicit math snippets instead of expecting raw Markdown to render.
    """

    raw = "" if text is None else str(text)
    normalized_parts: list[str] = []
    for part in _MATH_SPAN_RE.split(raw):
        if not part:
            continue
        if part.startswith("$") and part.endswith("$"):
            normalized_parts.append(part)
            continue
        cleaned = part.replace("`", "")
        cleaned = cleaned.replace("**", "").replace("__", "")
        cleaned = re.sub(r"(?<!\*)\*(?!\*)", "", cleaned)
        lines = [" ".join(line.split()) for line in cleaned.splitlines()]
        if width is not None:
            wrapped = []
            for line in lines:
                wrapped.extend(textwrap.wrap(line, width=width) if line else [""])
            lines = wrapped
        normalized_parts.append("\n".join(lines))
    normalized = "".join(normalized_parts).strip()
    return re.sub(r"([A-Za-z0-9,;:%])(?=\$)", r"\1 ", normalized)


def pub_text_position(bounds: PubBounds, *, ha: str = "center", va: str = "center") -> tuple[float, float]:
    """Return the anchor point for text inside a fixed slot."""

    x = {"left": bounds.left, "center": bounds.center[0], "right": bounds.right}.get(str(ha), bounds.center[0])
    y = {"bottom": bounds.bottom, "center": bounds.center[1], "top": bounds.top}.get(str(va), bounds.center[1])
    return float(x), float(y)


def add_pub_slot_text(
    ax,
    bounds: PubBounds,
    text: object,
    *,
    style: PubTextStyle | None = None,
    width: int | None = None,
    transform=None,
    clip: bool = True,
    **kwargs,
):
    """Add normalized text anchored inside a fixed rectangle.

    ``bounds`` are interpreted in the coordinate system of ``transform``. The
    default is ``ax.transAxes`` so callers can reserve stable title, label, and
    counter slots inside a panel or storyboard frame.
    """

    if style is None:
        style = PubTextStyle()
    if transform is None:
        transform = ax.transAxes
    x, y = pub_text_position(bounds, ha=style.ha, va=style.va)
    artist = ax.text(
        x,
        y,
        normalize_pub_markdown(text, width=width),
        transform=transform,
        ha=style.ha,
        va=style.va,
        fontsize=style.fontsize,
        color=style.color,
        fontweight=style.fontweight,
        linespacing=style.linespacing,
        clip_on=clip,
        **kwargs,
    )
    artist._pub_slot_bounds = bounds
    return artist


def pub_measure_text_axes(
    ax,
    text: object,
    *,
    style: PubTextStyle | None = None,
    width: int | None = None,
) -> tuple[float, float]:
    """Measure normalized text as axes-fraction width and height.

    This uses the active Matplotlib renderer and the same font settings as the
    eventual text artist. Use it before placing fixed slots when label length
    and font size should determine the gutter instead of a guessed constant.
    """

    if style is None:
        style = PubTextStyle()
    fig = ax.figure
    probe = ax.text(
        0.0,
        0.0,
        normalize_pub_markdown(text, width=width),
        transform=ax.transAxes,
        ha=style.ha,
        va=style.va,
        fontsize=style.fontsize,
        color=style.color,
        fontweight=style.fontweight,
        linespacing=style.linespacing,
        alpha=0.0,
        clip_on=False,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_box = probe.get_window_extent(renderer=renderer)
    axes_box = ax.get_window_extent(renderer=renderer)
    probe.remove()
    return (
        float(text_box.width) / max(float(axes_box.width), 1.0),
        float(text_box.height) / max(float(axes_box.height), 1.0),
    )


def pub_required_text_width_axes(
    ax,
    texts: Iterable[object],
    *,
    style: PubTextStyle | None = None,
    width: int | None = None,
    pad: float = 0.0,
    min_width: float = 0.0,
    max_width: float = 1.0,
) -> float:
    """Return the axes-fraction width needed for the widest text block."""

    measured = [pub_measure_text_axes(ax, text, style=style, width=width)[0] for text in texts]
    needed = (max(measured) if measured else 0.0) + float(pad)
    return min(max(float(needed), float(min_width)), float(max_width))


def pub_required_text_height_axes(
    ax,
    texts: Iterable[object],
    *,
    style: PubTextStyle | None = None,
    width: int | None = None,
    pad: float = 0.0,
    min_height: float = 0.0,
    max_height: float = 1.0,
) -> float:
    """Return the axes-fraction height needed for the tallest text block."""

    measured = [pub_measure_text_axes(ax, text, style=style, width=width)[1] for text in texts]
    needed = (max(measured) if measured else 0.0) + float(pad)
    return min(max(float(needed), float(min_height)), float(max_height))


def assert_pub_bounds_inside_unit(bounds: Iterable[PubBounds], *, tol: float = 1e-9) -> None:
    """Raise if axes-fraction bounds leave the unit square."""

    for idx, bound in enumerate(bounds):
        if (
            bound.left < -tol
            or bound.bottom < -tol
            or bound.right > 1.0 + tol
            or bound.top > 1.0 + tol
        ):
            raise AssertionError(f"PubBounds #{idx} leaves the unit square: {bound!r}.")
