"""Measured single-field layout with a separate colorbar and uniform type."""

from __future__ import annotations

from dataclasses import dataclass

from empowerment_warp._vendor.plotting.contracts.core import PubBounds
from empowerment_warp._vendor.plotting.contracts.layout import PubNamedLayout
from empowerment_warp._vendor.plotting.contracts.typography import pub_em_inches, pub_measure_text_inches
from empowerment_warp._vendor.plotting.plot_style import pub_uniform_typography_for_width


@dataclass(frozen=True)
class PubScalarFieldLayoutSpec:
    """Fixed reading width; text determines gutters, never the font size."""

    width_in: float = 6.5
    tier: str = "board"
    outer_pad_em: float = 0.6
    label_pad_em: float = 0.45
    tick_pad_em: float = 0.25
    title_gap_em: float = 0.6
    cbar_gap_em: float = 0.65
    cbar_width_em: float = 0.6

    def resolve(self, plt, *, title, x_label, y_label, x_ticks, y_ticks, cbar_ticks):
        typography = pub_uniform_typography_for_width(self.width_in, tier=self.tier)
        font = typography.axis_label.fontsize
        em = pub_em_inches(font)

        def measure(text):
            return pub_measure_text_inches(plt, text, style=typography.axis_label)

        def tick_extent(labels, dimension):
            return max((measure(label)[dimension] for label in labels), default=0.0)

        title_w, title_h = measure(title)
        x_label_w, x_label_h = measure(x_label)
        y_label_w, y_label_h = measure(y_label)
        margin = self.outer_pad_em * em
        tick_pad = self.tick_pad_em * em
        label_pad = self.label_pad_em * em
        # Include tick length (Matplotlib's shared style) in the owning gutter.
        tick_length = float(plt.rcParams['xtick.major.size']) / 72.0
        tick_gap = tick_pad + tick_length
        left = margin + y_label_h + label_pad + tick_extent(y_ticks, 0) + tick_gap
        right = margin + tick_extent(cbar_ticks, 0) + tick_gap
        bottom = margin + x_label_h + label_pad + tick_extent(x_ticks, 1) + tick_gap
        cbar_width = self.cbar_width_em * em
        gap = self.cbar_gap_em * em
        panel = self.width_in - left - gap - cbar_width - right
        if panel <= 0 or max(title_w, x_label_w, y_label_w) > panel:
            raise ValueError('Scalar-field labels exceed the fixed panel; shorten or wrap the labels.')
        title_gap = max(self.title_gap_em * em, tick_extent(cbar_ticks, 1) / 2)
        title_y = bottom + panel + title_gap
        height = title_y + title_h + margin
        return PubNamedLayout(
            fig_w=self.width_in,
            fig_h=height,
            bounds={
                'field': PubBounds(left, bottom, panel, panel),
                'cbar': PubBounds(left + panel + gap, bottom, cbar_width, panel),
                'title': PubBounds(left, title_y, panel, title_h),
            },
            metadata={
                'tier': self.tier, 'text_pt': font,
                'label_pad_pt': self.label_pad_em * font,
                'tick_pad_pt': self.tick_pad_em * font,
            },
        )
