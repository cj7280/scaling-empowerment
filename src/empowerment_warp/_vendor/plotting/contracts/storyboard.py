"""Lightweight Manim-style storyboard contracts.

The actual Manim package is optional and often expensive to smoke on cluster
nodes. This module captures the reusable parts of Manim-like diagrams that
should be stable before an external render runs: fixed scene slots, normalized
display text, color/style choices, and a manifest-friendly frame spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from empowerment_warp._vendor.plotting.contracts.core import PubBounds, PubTextStyle, add_pub_slot_text, assert_pub_bounds_inside_unit


@dataclass(frozen=True)
class PubManimFrame:
    """One semantic frame in a Manim-style storyboard."""

    label: str
    detail: str
    value: float
    color: object


@dataclass(frozen=True)
class PubManimStoryboardGeometry:
    """Stable scene slots for a storyboard preview."""

    bounds: PubBounds = field(default_factory=lambda: PubBounds(0.05, 0.10, 0.90, 0.72))
    title_slot: PubBounds = field(default_factory=lambda: PubBounds(0.05, 0.86, 0.90, 0.10))
    footer_slot: PubBounds = field(default_factory=lambda: PubBounds(0.08, 0.02, 0.84, 0.06))
    frame_gap: float = 0.018
    label_h: float = 0.18
    axis_h: float = 0.10


class PubManimStoryboard:
    """A static contract preview for a Manim-like sequence."""

    def __init__(
        self,
        *,
        title: str,
        footer: str,
        frames: list[PubManimFrame] | tuple[PubManimFrame, ...],
        geometry: PubManimStoryboardGeometry | None = None,
    ) -> None:
        if not frames:
            raise ValueError("A storyboard needs at least one frame.")
        self.title = str(title)
        self.footer = str(footer)
        self.frames = tuple(frames)
        self.geometry = geometry or PubManimStoryboardGeometry()
        assert_pub_bounds_inside_unit((self.geometry.bounds, self.geometry.title_slot, self.geometry.footer_slot))

    def frame_slots(self) -> tuple[PubBounds, ...]:
        """Return fixed slots, left-to-right, for every semantic frame."""

        return self.geometry.bounds.split_cols(len(self.frames), gap=self.geometry.frame_gap)

    def draw(self, ax) -> dict[str, object]:
        """Draw a Matplotlib storyboard preview and return created artists."""

        import numpy as np
        from matplotlib.patches import Rectangle

        ax.set_axis_off()
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        title_style = PubTextStyle(fontsize=10.4, color="#111111", fontweight="normal", ha="center", va="center")
        footer_style = PubTextStyle(fontsize=7.1, color="#555555", ha="center", va="center")
        artists: dict[str, object] = {"frames": [], "labels": []}
        artists["title"] = add_pub_slot_text(ax, self.geometry.title_slot, self.title, style=title_style, width=96)
        artists["footer"] = add_pub_slot_text(ax, self.geometry.footer_slot, self.footer, style=footer_style, width=112)

        for slot, frame in zip(self.frame_slots(), self.frames):
            label_slot = PubBounds(slot.x + 0.025 * slot.w, slot.top - self.geometry.label_h, 0.95 * slot.w, self.geometry.label_h)
            plot_slot = PubBounds(slot.x + 0.06 * slot.w, slot.y + self.geometry.axis_h, 0.88 * slot.w, slot.h - self.geometry.label_h - self.geometry.axis_h)
            axis_y = slot.y + 0.38 * self.geometry.axis_h
            label_style = PubTextStyle(fontsize=6.6, color="#111111", ha="center", va="center", linespacing=0.94)
            artists["labels"].append(
                add_pub_slot_text(
                    ax,
                    label_slot,
                    f"{frame.label}\n{frame.detail}",
                    style=label_style,
                    width=max(12, int(0.14 * slot.w * 100)),
                )
            )
            ax.add_patch(
                Rectangle(
                    (slot.x, slot.y),
                    slot.w,
                    slot.h,
                    transform=ax.transAxes,
                    facecolor="#ffffff",
                    edgecolor="#d7dbe2",
                    linewidth=0.65,
                    zorder=0,
                )
            )
            ax.plot(
                [plot_slot.left, plot_slot.right],
                [axis_y, axis_y],
                transform=ax.transAxes,
                color="#8b8f97",
                lw=0.75,
                zorder=2,
            )
            xs = np.linspace(plot_slot.left, plot_slot.right, 160)
            center = plot_slot.left + plot_slot.w * (0.25 + 0.50 * float(frame.value))
            spread = max(0.045, plot_slot.w * (0.30 - 0.20 * float(frame.value)))
            ys = np.exp(-0.5 * ((xs - center) / spread) ** 2)
            ys = axis_y + ys / max(float(ys.max()), 1e-9) * plot_slot.h * (0.42 + 0.36 * float(frame.value))
            poly_x = np.concatenate([[plot_slot.left], xs, [plot_slot.right]])
            poly_y = np.concatenate([[axis_y], ys, [axis_y]])
            fill = ax.fill(
                poly_x,
                poly_y,
                transform=ax.transAxes,
                color=frame.color,
                alpha=0.72,
                linewidth=0.0,
                zorder=3,
            )[0]
            line = ax.plot(xs, ys, transform=ax.transAxes, color=frame.color, lw=1.0, zorder=4)[0]
            artists["frames"].append((fill, line))
        return artists

    def manifest(self) -> dict:
        """Return a small machine-readable contract summary."""

        return {
            "kind": "pub_manim_storyboard",
            "title": self.title,
            "frame_count": len(self.frames),
            "geometry": {
                "bounds": self.geometry.bounds.__dict__,
                "title_slot": self.geometry.title_slot.__dict__,
                "footer_slot": self.geometry.footer_slot.__dict__,
                "frame_gap": self.geometry.frame_gap,
            },
            "frames": [
                {"label": frame.label, "detail": frame.detail, "value": float(frame.value)}
                for frame in self.frames
            ],
        }
