"""Reusable fixed-inch layout contracts for publication figures."""

from __future__ import annotations

from dataclasses import dataclass, field

from empowerment_warp._vendor.plotting.contracts.core import PubBounds
from empowerment_warp._vendor.plotting.contracts.legend import PubLegendGeometry, PubLegendSpec, pub_resolve_legend_geometry
from empowerment_warp._vendor.plotting.contracts.typography import (
    pub_measure_text_inches,
    PubTypographySpec,
    pub_axis_label_gutter_inches,
    pub_em_inches,
    pub_text_band_height_inches,
)


@dataclass(frozen=True)
class PubInsets:
    """Fixed page margins in inches."""

    left: float
    right: float
    bottom: float
    top: float


@dataclass(frozen=True)
class PubNamedLayout:
    """Resolved named bounds plus figure size."""

    fig_w: float
    fig_h: float
    bounds: dict[str, PubBounds]
    metadata: dict[str, object] = field(default_factory=dict)

    def __getitem__(self, name: str) -> PubBounds:
        return self.bounds[name]


@dataclass(frozen=True)
class PubFixedPanelCbarRowSpec:
    """Fixed equal-panel row, optionally with a colorbar after panel zero."""

    panel_names: tuple[str, ...] = ("heat", "vector")
    panel_size: float = 3.05
    insets: PubInsets = field(default_factory=lambda: PubInsets(left=0.48, right=0.38, bottom=0.625, top=0.495))
    cbar_name: str = "cbar"
    cbar_width: float = 0.17
    cbar_height_fraction: float = 0.78
    heat_cbar_gap: float = 0.12
    post_cbar_gap: float = 1.58
    remaining_panel_gaps: tuple[float, ...] = ()
    bottom_band_name: str | None = None
    bottom_band_height: float = 0.0
    bottom_band_gap: float = 0.0
    title_band_height: float = 0.0
    title_band_gap: float = 0.0
    title_pad_points: float = 8.0

    @classmethod
    def publication_pair(cls) -> "PubFixedPanelCbarRowSpec":
        """Reference heatmap + vector-panel layout."""

        return cls()

    @classmethod
    def stress_triplet(cls) -> "PubFixedPanelCbarRowSpec":
        """Compact three-panel stress layout with legend and title bands."""

        return cls(
            panel_names=("heat", "board", "vector"),
            panel_size=1.86,
            insets=PubInsets(left=0.337, right=0.203, bottom=0.173, top=0.087),
            cbar_width=0.13,
            cbar_height_fraction=0.74,
            heat_cbar_gap=0.10,
            post_cbar_gap=0.62,
            remaining_panel_gaps=(0.68,),
            bottom_band_name="legend",
            bottom_band_height=0.48,
            bottom_band_gap=0.34,
            title_band_height=0.38,
            title_band_gap=0.05,
            title_pad_points=5.8,
        )

    def resolve(self) -> PubNamedLayout:
        if not self.panel_names:
            raise ValueError("panel_names must not be empty.")
        cbar_h = self.cbar_height_fraction * self.panel_size
        content_y = self.insets.bottom + self.bottom_band_height + self.bottom_band_gap
        bounds: dict[str, PubBounds] = {}
        x = self.insets.left
        bounds[self.panel_names[0]] = PubBounds(x, content_y, self.panel_size, self.panel_size)
        x += self.panel_size + self.heat_cbar_gap
        bounds[self.cbar_name] = PubBounds(
            x,
            content_y + 0.5 * (self.panel_size - cbar_h),
            self.cbar_width,
            cbar_h,
        )
        x += self.cbar_width + self.post_cbar_gap
        for idx, name in enumerate(self.panel_names[1:]):
            bounds[name] = PubBounds(x, content_y, self.panel_size, self.panel_size)
            if idx < len(self.remaining_panel_gaps):
                x += self.panel_size + self.remaining_panel_gaps[idx]
            else:
                x += self.panel_size
        content_right = max(bound.right for name, bound in bounds.items() if name != self.cbar_name)
        fig_w = content_right + self.insets.right
        fig_h = content_y + self.panel_size + self.title_band_gap + self.title_band_height + self.insets.top
        if self.bottom_band_name is not None and self.bottom_band_height > 0.0:
            bounds[self.bottom_band_name] = PubBounds(
                self.insets.left,
                self.insets.bottom,
                fig_w - self.insets.left - self.insets.right,
                self.bottom_band_height,
            )
        if self.title_band_height > 0.0:
            title_y = content_y + self.panel_size + self.title_band_gap
            for name in self.panel_names:
                panel = bounds[name]
                bounds[f"{name}_title"] = PubBounds(panel.x, title_y, panel.w, self.title_band_height)
        return PubNamedLayout(
            fig_w=float(fig_w),
            fig_h=float(fig_h),
            bounds=bounds,
            metadata={"title_pad_points": self.title_pad_points, "panel_size": self.panel_size},
        )


@dataclass(frozen=True)
class PubGridMainLegendLayoutSpec:
    """Side grid plus main square panel with title and legend bands."""

    side_rows: int = 3
    side_cols: int = 1
    small_panel: float = 1.55
    panel_gap: float = 0.07
    x_gap: float = 0.16
    legend_height: float = 0.48
    legend_gap: float = 0.18
    title_height: float = 0.34
    title_gap: float = 0.05
    insets: PubInsets = field(default_factory=lambda: PubInsets(left=0.30, right=0.25, bottom=0.18, top=0.06))

    @classmethod
    def rollout_reference(cls) -> "PubGridMainLegendLayoutSpec":
        return cls()

    @classmethod
    def stress_grid(cls) -> "PubGridMainLegendLayoutSpec":
        return cls(
            side_rows=2,
            side_cols=2,
            small_panel=1.14,
            panel_gap=0.065,
            x_gap=0.18,
            legend_height=0.68,
            legend_gap=0.20,
            title_height=0.37,
            title_gap=0.05,
            insets=PubInsets(left=0.28, right=0.25, bottom=0.17, top=0.08),
        )

    @property
    def main_panel(self) -> float:
        return self.side_cols * self.small_panel + (self.side_cols - 1) * self.panel_gap

    def resolve(self) -> PubNamedLayout:
        grid_w = self.main_panel
        grid_h = self.side_rows * self.small_panel + (self.side_rows - 1) * self.panel_gap
        main_panel = max(grid_w, grid_h)
        content_y = self.insets.bottom + self.legend_height + self.legend_gap
        main_x = self.insets.left + grid_w + self.x_gap
        fig_w = main_x + main_panel + self.insets.right
        fig_h = content_y + main_panel + self.title_gap + self.title_height + self.insets.top
        bounds: dict[str, PubBounds] = {
            "main": PubBounds(main_x, content_y, main_panel, main_panel),
            "legend": PubBounds(self.insets.left, self.insets.bottom, grid_w + self.x_gap + main_panel, self.legend_height),
            "side_title": PubBounds(self.insets.left, content_y + main_panel + self.title_gap, grid_w, self.title_height),
            "main_title": PubBounds(main_x, content_y + main_panel + self.title_gap, main_panel, self.title_height),
        }
        for row in range(self.side_rows):
            for col in range(self.side_cols):
                idx = row * self.side_cols + col
                x = self.insets.left + col * (self.small_panel + self.panel_gap)
                y = content_y + (self.side_rows - 1 - row) * (self.small_panel + self.panel_gap)
                bounds[f"side_{idx}"] = PubBounds(x, y, self.small_panel, self.small_panel)
        return PubNamedLayout(
            fig_w=float(fig_w),
            fig_h=float(fig_h),
            bounds=bounds,
            metadata={"small_panel": self.small_panel, "main_panel": main_panel, "panel_gap": self.panel_gap},
        )


@dataclass(frozen=True)
class PubMetricTimelineLayoutSpec:
    """Two metric panels plus event, legend, and title bands."""

    fig_width: float = 4.92
    insets: PubInsets = field(default_factory=lambda: PubInsets(left=0.80, right=0.18, bottom=0.44, top=0.08))
    panel_height: float = 0.95
    panel_gap_em: float = 4.5
    icon_height: float = 0.80
    icon_gap_em: float = 1.8
    legend_height: float | None = None
    legend_gap_em: float = 1.55
    title_height: float | None = None
    title_height_em: float = 1.38
    title_pad_em: float = 0.16
    title_gap_em: float = 0.32

    def resolve(
        self,
        *,
        typography: PubTypographySpec,
        legend_height: float | None = None,
        title: object | None = None,
        plt=None,
    ) -> PubNamedLayout:
        axis_em = pub_em_inches(typography.axis_label.fontsize)
        title_em = pub_em_inches(typography.title.fontsize)
        plot_w = self.fig_width - self.insets.left - self.insets.right
        legend_h = self.legend_height if legend_height is None else float(legend_height)
        if legend_h is None:
            legend_h = pub_em_inches(typography.legend.fontsize, 2.05)
        if self.title_height is not None:
            title_h = float(self.title_height)
        elif title is not None and plt is not None:
            title_h = max(
                pub_text_band_height_inches(plt, title, style=typography.title, pad_em=self.title_pad_em),
                pub_em_inches(typography.title.fontsize, self.title_height_em),
            )
        else:
            title_h = pub_em_inches(typography.title.fontsize, self.title_height_em)
        bottom_y = self.insets.bottom
        top_y = bottom_y + self.panel_height + self.panel_gap_em * axis_em
        icon_y = top_y + self.panel_height + self.icon_gap_em * axis_em
        legend_y = icon_y + self.icon_height + self.legend_gap_em * axis_em
        title_y = legend_y + legend_h + self.title_gap_em * title_em
        fig_h = title_y + title_h + self.insets.top
        bounds = {
            "bottom": PubBounds(self.insets.left, bottom_y, plot_w, self.panel_height),
            "top": PubBounds(self.insets.left, top_y, plot_w, self.panel_height),
            "icons": PubBounds(self.insets.left, icon_y, plot_w, self.icon_height),
            "legend": PubBounds(self.insets.left, legend_y, plot_w, legend_h),
            "title": PubBounds(self.insets.left, title_y, plot_w, title_h),
        }
        return PubNamedLayout(fig_w=float(self.fig_width), fig_h=float(fig_h), bounds=bounds)


@dataclass(frozen=True)
class PubMediaRowsLayoutSpec:
    """Compact rows with label, frame, and path columns plus title/legend."""

    row_count: int = 2
    label_width: float = 0.95
    label_gap: float = 0.08
    frame_width: float = 0.72
    frame_height: float = 0.72
    path_width: float = 1.95
    path_height: float = 0.72
    col_gap: float = 0.34
    row_gap: float = 0.28
    col_label_height: float = 0.13
    legend_width: float = 2.52
    legend_height: float = 0.15
    legend_gap: float = 0.20
    title_height: float = 0.30
    title_gap: float = 0.04
    insets: PubInsets = field(default_factory=lambda: PubInsets(left=0.24, right=0.20, bottom=0.12, top=0.08))

    @classmethod
    def compact_two_row(cls) -> "PubMediaRowsLayoutSpec":
        return cls()

    def resolve(self) -> PubNamedLayout:
        content_w = self.label_width + self.label_gap + self.frame_width + self.col_gap + self.path_width
        rows_h = self.row_count * self.frame_height + (self.row_count - 1) * self.row_gap
        fig_w = self.insets.left + content_w + self.insets.right
        fig_h = (
            self.insets.bottom
            + self.legend_height
            + self.legend_gap
            + rows_h
            + self.col_label_height
            + self.title_gap
            + self.title_height
            + self.insets.top
        )
        rows_bottom = self.insets.bottom + self.legend_height + self.legend_gap
        frame_x = self.insets.left + self.label_width + self.label_gap
        path_x = frame_x + self.frame_width + self.col_gap
        bounds = {
            "legend": PubBounds(0.5 * (fig_w - self.legend_width), self.insets.bottom, self.legend_width, self.legend_height),
            "title": PubBounds(0.0, rows_bottom + rows_h + self.col_label_height + self.title_gap, fig_w, self.title_height),
            "label_header": PubBounds(self.insets.left, rows_bottom + rows_h + 0.02, self.label_width, self.col_label_height),
            "frame_header": PubBounds(frame_x, rows_bottom + rows_h + 0.02, self.frame_width, self.col_label_height),
            "path_header": PubBounds(path_x, rows_bottom + rows_h + 0.02, self.path_width, self.col_label_height),
        }
        for row_idx in range(self.row_count):
            y = rows_bottom + (self.row_count - 1 - row_idx) * (self.frame_height + self.row_gap)
            bounds[f"label_{row_idx}"] = PubBounds(self.insets.left, y, self.label_width, self.frame_height)
            bounds[f"frame_{row_idx}"] = PubBounds(frame_x, y, self.frame_width, self.frame_height)
            bounds[f"path_{row_idx}"] = PubBounds(path_x, y, self.path_width, self.path_height)
        return PubNamedLayout(
            fig_w=float(fig_w),
            fig_h=float(fig_h),
            bounds=bounds,
            metadata={"row_gap": self.row_gap, "frame_height": self.frame_height},
        )


@dataclass(frozen=True)
class PubContractStressLayoutSpec:
    """Named page regions for contract stress fixtures."""

    kind: str

    @classmethod
    def sankey(cls) -> "PubContractStressLayoutSpec":
        return cls(kind="sankey")

    @classmethod
    def storyboard(cls) -> "PubContractStressLayoutSpec":
        return cls(kind="storyboard")

    def resolve(self) -> PubNamedLayout:
        if self.kind == "sankey":
            return PubNamedLayout(
                fig_w=6.6,
                fig_h=3.85,
                bounds={
                    "title": PubBounds(0.28, 3.48, 6.04, 0.28),
                    "body": PubBounds(0.24, 0.72, 6.12, 2.66),
                    "legend": PubBounds(0.78, 0.18, 5.04, 0.36),
                },
                metadata={"left_label_pad": 0.040, "right_label_pad": 0.045},
            )
        if self.kind == "storyboard":
            return PubNamedLayout(
                fig_w=6.6,
                fig_h=3.25,
                bounds={"body": PubBounds(0.18, 0.12, 6.24, 3.00)},
            )
        raise ValueError(f"Unknown contract stress layout kind: {self.kind!r}")


@dataclass(frozen=True)
class PubMetricStackLayout:
    """Resolved two-or-more-panel metric layout in figure inches."""

    fig_w: float
    fig_h: float
    panels: tuple[PubBounds, ...]
    title: PubBounds
    legend: PubBounds
    legend_geometry: PubLegendGeometry

    @property
    def panels_bottom_to_top(self) -> tuple[PubBounds, ...]:
        return self.panels

    @property
    def panels_top_to_bottom(self) -> tuple[PubBounds, ...]:
        return tuple(reversed(self.panels))


@dataclass(frozen=True)
class PubMetricStackLayoutSpec:
    """Intent-level contract for compact stacked metric plots."""

    panel_count: int = 2
    min_panel_width: float = 3.05
    panel_height: float = 1.08
    panel_gap_em: float = 5.1
    title_gap_em: float = 1.25
    legend_gap_em: float = 2.45
    bottom_margin_em: float = 2.0
    top_margin_em: float = 0.85
    right_margin_em: float = 1.35
    content_pad_em: float = 1.10
    title_pad_em: float = 0.45

    def resolve(
        self,
        plt,
        *,
        title: object,
        legend_labels: tuple[object, ...] | list[object],
        y_labels: tuple[object, ...] | list[object],
        y_tick_labels: tuple[object, ...] | list[object],
        typography: PubTypographySpec,
        legend_spec: PubLegendSpec,
    ) -> PubMetricStackLayout:
        """Resolve figure, panel, title, and legend bounds from measured text."""

        if int(self.panel_count) <= 0:
            raise ValueError("panel_count must be positive.")
        panel_count = int(self.panel_count)
        axis_em = pub_em_inches(typography.axis_label.fontsize)
        legend_em = pub_em_inches(legend_spec.fontsize)
        title_em = pub_em_inches(typography.title.fontsize)

        left_gutter = pub_axis_label_gutter_inches(
            plt,
            y_labels=y_labels,
            y_tick_labels=y_tick_labels,
            typography=typography,
        )
        plot_w = float(self.min_panel_width)
        legend_geometry = pub_resolve_legend_geometry(
            plt,
            legend_labels,
            target_width=plot_w,
            spec=legend_spec,
        )
        plot_w = max(plot_w, legend_geometry.required_width + self.content_pad_em * legend_em)
        legend_geometry = pub_resolve_legend_geometry(
            plt,
            legend_labels,
            target_width=plot_w,
            spec=legend_spec,
        )
        plot_w = max(plot_w, legend_geometry.required_width + self.content_pad_em * legend_em)

        # The panel must also be wide enough for the TITLE, not just the legend.
        # Solving on the title's height alone lets a long title at a large type size
        # overflow the page horizontally. Same treatment the legend gets.
        # The figure page is left_gutter + plot_w + margin_r, so the title has to fit
        # inside ALL of it, not just the panel.
        _margin_r = self.right_margin_em * axis_em
        title_w, _ = pub_measure_text_inches(plt, title, style=typography.title)
        # The title is anchored at the panel's LEFT edge, so the page must hold
        # left_gutter + title_w; only the right margin can be borrowed back.
        plot_w = max(plot_w, title_w + self.content_pad_em * title_em - _margin_r)

        title_h = pub_text_band_height_inches(
            plt,
            title,
            style=typography.title,
            pad_em=self.title_pad_em,
        )
        panel_gap = self.panel_gap_em * axis_em
        title_gap = self.title_gap_em * title_em
        legend_gap = self.legend_gap_em * axis_em
        margin_b = self.bottom_margin_em * axis_em
        margin_t = self.top_margin_em * title_em
        margin_r = self.right_margin_em * axis_em

        legend_bounds = PubBounds(left_gutter, margin_b, plot_w, legend_geometry.height)
        first_panel_y = legend_bounds.top + legend_gap
        panels = tuple(
            PubBounds(
                left_gutter,
                first_panel_y + idx * (float(self.panel_height) + panel_gap),
                plot_w,
                float(self.panel_height),
            )
            for idx in range(panel_count)
        )
        title_bounds = PubBounds(left_gutter, panels[-1].top + title_gap, plot_w, title_h)
        fig_w = left_gutter + plot_w + margin_r
        fig_h = title_bounds.top + margin_t
        return PubMetricStackLayout(
            fig_w=float(fig_w),
            fig_h=float(fig_h),
            panels=panels,
            title=title_bounds,
            legend=legend_bounds,
            legend_geometry=legend_geometry,
        )
