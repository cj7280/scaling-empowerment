"""Class-based Sankey layout and rendering helpers.

This module is deliberately data-agnostic. Callers provide nodes and flows;
``PubSankeyDiagram`` computes node extents, proportional ports, x-monotone
ribbons, labels, and validation before anything is drawn. Project scripts can
still own data loading, but they should not own ribbon geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from empowerment_warp._vendor.plotting.contracts.core import PubBounds, PubTextStyle, add_pub_slot_text, normalize_pub_markdown


@dataclass(frozen=True)
class SankeyNode:
    """A Sankey node with a semantic column and conserved value."""

    key: str
    label: str
    column: int
    value: float
    order: float = 0.0
    color: object = "#c9cdd3"
    edgecolor: object | None = None
    label_side: Literal["left", "right", "none"] | None = None


@dataclass(frozen=True)
class SankeyFlow:
    """A conserved flow from one node to another."""

    source: str
    target: str
    value: float
    label: str = ""
    color: object | None = None
    alpha: float = 0.56
    source_rank: float = 0.0
    target_rank: float = 0.0


@dataclass(frozen=True)
class PubSankeyGeometry:
    """Geometry knobs for a whole Sankey diagram.

    All lengths are axes-fraction units inside ``bounds``. The renderer computes
    all node/ribbon positions from these values; callers should change the
    object, not nudge individual nodes.
    """

    bounds: PubBounds = field(default_factory=lambda: PubBounds(0.08, 0.16, 0.84, 0.68))
    label_bounds: PubBounds | None = None
    node_width: float = 0.032
    node_gap: float = 0.024
    min_node_height: float = 0.012
    curve_fraction: float = 0.34
    label_pad: float = 0.014
    label_min_height: float = 0.052
    label_gap: float = 0.010
    label_fontsize: float = 7.2
    label_linespacing: float = 1.08
    label_bbox_facecolor: object | None = None
    label_bbox_alpha: float = 0.70
    label_bbox_pad: float = 0.06
    label_side: Literal["outside", "right", "left", "none"] = "outside"
    node_edgecolor: object = "none"
    node_linewidth: float = 0.0
    node_joinstyle: Literal["miter", "round", "bevel"] = "miter"


@dataclass(frozen=True)
class SankeyNodeLayout:
    """Resolved node geometry."""

    node: SankeyNode
    x: float
    y: float
    h: float
    w: float


@dataclass(frozen=True)
class SankeyFlowLayout:
    """Resolved flow geometry."""

    flow: SankeyFlow
    x0: float
    y0: float
    x1: float
    y1: float
    h: float


@dataclass(frozen=True)
class SankeyLayout:
    """Resolved diagram layout."""

    nodes: dict[str, SankeyNodeLayout]
    flows: tuple[SankeyFlowLayout, ...]
    unit: float


class PubSankeyDiagram:
    """Compute and draw a proportional, x-monotone Sankey diagram."""

    def __init__(
        self,
        nodes: list[SankeyNode] | tuple[SankeyNode, ...],
        flows: list[SankeyFlow] | tuple[SankeyFlow, ...],
        *,
        geometry: PubSankeyGeometry | None = None,
    ) -> None:
        self.nodes = tuple(nodes)
        self.flows = tuple(flows)
        self.geometry = geometry or PubSankeyGeometry()
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        keys = [node.key for node in self.nodes]
        if len(set(keys)) != len(keys):
            raise ValueError("Sankey node keys must be unique.")
        known = set(keys)
        for node in self.nodes:
            if float(node.value) < 0.0:
                raise ValueError(f"Sankey node {node.key!r} has a negative value.")
        for flow in self.flows:
            if flow.source not in known or flow.target not in known:
                raise ValueError(f"Sankey flow {flow.source!r}->{flow.target!r} references an unknown node.")
            if float(flow.value) < 0.0:
                raise ValueError(f"Sankey flow {flow.source!r}->{flow.target!r} has a negative value.")

    def _column_x(self, columns: list[int], column: int) -> float:
        bounds = self.geometry.bounds
        if len(columns) == 1:
            return bounds.center[0]
        rank = columns.index(column)
        return bounds.left + bounds.w * rank / float(len(columns) - 1)

    def _global_unit(self) -> float:
        by_col: dict[int, float] = {}
        for node in self.nodes:
            by_col[int(node.column)] = by_col.get(int(node.column), 0.0) + max(float(node.value), 0.0)
        if not by_col:
            raise ValueError("Sankey diagrams need at least one node.")
        candidates = []
        for column, total in by_col.items():
            count = sum(1 for node in self.nodes if int(node.column) == column)
            usable = self.geometry.bounds.h - self.geometry.node_gap * max(count - 1, 0)
            if usable <= 0.0:
                raise ValueError(f"Sankey column {column} has no usable vertical space.")
            if total > 0.0:
                candidates.append(usable / total)
        if not candidates:
            raise ValueError("Sankey diagrams need positive node mass.")
        return min(candidates)

    def _check_mass_conservation(self, *, tol: float = 1e-8) -> None:
        incoming = {node.key: 0.0 for node in self.nodes}
        outgoing = {node.key: 0.0 for node in self.nodes}
        for flow in self.flows:
            outgoing[flow.source] += float(flow.value)
            incoming[flow.target] += float(flow.value)
        for node in self.nodes:
            expected = float(node.value)
            for direction, observed in (("incoming", incoming[node.key]), ("outgoing", outgoing[node.key])):
                if observed <= 0.0:
                    continue
                if abs(observed - expected) > max(tol, abs(expected) * tol):
                    raise ValueError(
                        f"Sankey node {node.key!r} violates {direction} mass conservation: "
                        f"observed={observed:.12g}, expected={expected:.12g}."
                    )

    @staticmethod
    def _ports(
        node_layout: SankeyNodeLayout,
        flows: list[tuple[int, SankeyFlow]],
        *,
        unit: float,
        side: Literal["source", "target"],
    ) -> dict[int, float]:
        if side == "source":
            ordered = sorted(flows, key=lambda item: (item[1].source_rank, item[0]))
        else:
            ordered = sorted(flows, key=lambda item: (item[1].target_rank, item[0]))
        y_top = node_layout.y + 0.5 * node_layout.h
        ports: dict[int, float] = {}
        for idx, flow in ordered:
            h = max(float(flow.value) * unit, 0.0)
            ports[idx] = y_top - 0.5 * h
            y_top -= h
        return ports

    def layout(self) -> SankeyLayout:
        """Return the resolved layout, raising on invalid geometry."""

        self._check_mass_conservation()
        unit = self._global_unit()
        columns = sorted({int(node.column) for node in self.nodes})
        resolved: dict[str, SankeyNodeLayout] = {}
        for column in columns:
            column_nodes = sorted(
                [node for node in self.nodes if int(node.column) == column],
                key=lambda node: (float(node.order), node.label, node.key),
            )
            heights = [max(float(node.value) * unit, self.geometry.min_node_height) for node in column_nodes]
            total_h = sum(heights) + self.geometry.node_gap * max(len(heights) - 1, 0)
            y_top = self.geometry.bounds.center[1] + 0.5 * total_h
            x = self._column_x(columns, column)
            for node, h in zip(column_nodes, heights):
                resolved[node.key] = SankeyNodeLayout(
                    node=node,
                    x=float(x),
                    y=float(y_top - 0.5 * h),
                    h=float(h),
                    w=float(self.geometry.node_width),
                )
                y_top -= h + self.geometry.node_gap

        incoming: dict[str, list[tuple[int, SankeyFlow]]] = {node.key: [] for node in self.nodes}
        outgoing: dict[str, list[tuple[int, SankeyFlow]]] = {node.key: [] for node in self.nodes}
        for idx, flow in enumerate(self.flows):
            outgoing[flow.source].append((idx, flow))
            incoming[flow.target].append((idx, flow))
        source_ports = {
            key: self._ports(resolved[key], vals, unit=unit, side="source")
            for key, vals in outgoing.items()
        }
        target_ports = {
            key: self._ports(resolved[key], vals, unit=unit, side="target")
            for key, vals in incoming.items()
        }
        flow_layouts = []
        for idx, flow in enumerate(self.flows):
            src = resolved[flow.source]
            dst = resolved[flow.target]
            x0 = src.x + 0.5 * src.w
            x1 = dst.x - 0.5 * dst.w
            if x1 < x0:
                raise ValueError(
                    f"Sankey flow {flow.source!r}->{flow.target!r} doubles back in x: "
                    f"x0={x0:.6f}, x1={x1:.6f}."
                )
            flow_layouts.append(
                SankeyFlowLayout(
                    flow=flow,
                    x0=float(x0),
                    y0=float(source_ports[flow.source][idx]),
                    x1=float(x1),
                    y1=float(target_ports[flow.target][idx]),
                    h=float(max(flow.value * unit, 0.0)),
                )
            )
        return SankeyLayout(nodes=resolved, flows=tuple(flow_layouts), unit=float(unit))

    def _draw_flow(self, ax, flow_layout: SankeyFlowLayout, layout: SankeyLayout) -> object:
        from matplotlib.path import Path as MplPath
        from matplotlib.patches import PathPatch

        flow = flow_layout.flow
        color = flow.color
        if color is None:
            color = layout.nodes[flow.source].node.color
        curve = self.geometry.curve_fraction * (flow_layout.x1 - flow_layout.x0)
        c0 = flow_layout.x0 + curve
        c1 = flow_layout.x1 - curve
        if c0 > c1 + 1e-9:
            raise ValueError(f"Sankey flow {flow.source!r}->{flow.target!r} has invalid Bezier controls.")
        h = 0.5 * flow_layout.h
        verts = [
            (flow_layout.x0, flow_layout.y0 + h),
            (c0, flow_layout.y0 + h),
            (c1, flow_layout.y1 + h),
            (flow_layout.x1, flow_layout.y1 + h),
            (flow_layout.x1, flow_layout.y1 - h),
            (c1, flow_layout.y1 - h),
            (c0, flow_layout.y0 - h),
            (flow_layout.x0, flow_layout.y0 - h),
            (flow_layout.x0, flow_layout.y0 + h),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        patch = PathPatch(
            MplPath(verts, codes),
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            alpha=float(flow.alpha),
            zorder=1,
        )
        ax.add_patch(patch)
        return patch

    def _draw_node(self, ax, node_layout: SankeyNodeLayout) -> object:
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (node_layout.x - 0.5 * node_layout.w, node_layout.y - 0.5 * node_layout.h),
            node_layout.w,
            node_layout.h,
            transform=ax.transAxes,
            facecolor=node_layout.node.color,
            edgecolor=self.geometry.node_edgecolor if node_layout.node.edgecolor is None else node_layout.node.edgecolor,
            linewidth=float(self.geometry.node_linewidth),
            joinstyle=self.geometry.node_joinstyle,
            antialiased=False,
            zorder=3,
        )
        ax.add_patch(rect)
        return rect

    def _label_slot(self, node_layout: SankeyNodeLayout) -> PubBounds | None:
        side = self.geometry.label_side if node_layout.node.label_side is None else node_layout.node.label_side
        if side == "none":
            return None
        label_bounds = self.geometry.label_bounds or self.geometry.bounds
        if side == "outside":
            node_left = node_layout.x - 0.5 * node_layout.w
            node_right = node_layout.x + 0.5 * node_layout.w
            left_space = node_left - label_bounds.left
            right_space = label_bounds.right - node_right
            side = "left" if left_space > right_space else "right"
        pad = self.geometry.label_pad
        if side == "left":
            x0 = label_bounds.left
            x1 = node_layout.x - 0.5 * node_layout.w - pad
            return PubBounds(x0, node_layout.y - 0.5 * node_layout.h, max(x1 - x0, 0.01), node_layout.h)
        x0 = node_layout.x + 0.5 * node_layout.w + pad
        x1 = label_bounds.right
        return PubBounds(x0, node_layout.y - 0.5 * node_layout.h, max(x1 - x0, 0.01), node_layout.h)

    def _resolved_label_slots(self, layout: SankeyLayout) -> list[tuple[SankeyNodeLayout, PubBounds]]:
        """Return label slots with a separate collision pass from node heights."""

        raw: list[tuple[SankeyNodeLayout, PubBounds]] = []
        for node_layout in layout.nodes.values():
            slot = self._label_slot(node_layout)
            if slot is not None:
                raw.append((node_layout, slot))

        grouped: dict[tuple[str, float, float], list[tuple[SankeyNodeLayout, PubBounds]]] = {}
        for node_layout, slot in raw:
            side = "left" if slot.right <= node_layout.x else "right"
            key = (side, round(slot.x, 4), round(slot.right, 4))
            grouped.setdefault(key, []).append((node_layout, slot))

        label_bounds = self.geometry.label_bounds or self.geometry.bounds
        resolved: list[tuple[SankeyNodeLayout, PubBounds]] = []
        for items in grouped.values():
            ordered = sorted(items, key=lambda item: item[0].y, reverse=True)
            placed: list[tuple[SankeyNodeLayout, PubBounds]] = []
            cursor_top = label_bounds.top
            for node_layout, slot in ordered:
                h = max(float(slot.h), float(self.geometry.label_min_height))
                center_y = min(float(slot.center[1]), cursor_top - 0.5 * h)
                y = center_y - 0.5 * h
                placed.append((node_layout, PubBounds(slot.x, y, slot.w, h)))
                cursor_top = y - float(self.geometry.label_gap)
            if placed:
                min_bottom = min(slot.bottom for _, slot in placed)
                max_top = max(slot.top for _, slot in placed)
                min_shift = label_bounds.bottom - min_bottom
                max_shift = label_bounds.top - max_top
                if min_shift <= max_shift:
                    shift = min(max(0.0, min_shift), max_shift)
                else:
                    shift = 0.5 * (min_shift + max_shift)
                if abs(shift) > 1e-12:
                    placed = [
                        (node_layout, PubBounds(slot.x, slot.y + shift, slot.w, slot.h))
                        for node_layout, slot in placed
                    ]
            resolved.extend(placed)
        return resolved

    def draw(self, ax, *, label_width: int = 24) -> dict[str, object]:
        """Draw the diagram on ``ax`` and return the created artists."""

        layout = self.layout()
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_axis_off()

        artists: dict[str, object] = {"flows": [], "nodes": [], "labels": []}
        for flow_layout in layout.flows:
            artists["flows"].append(self._draw_flow(ax, flow_layout, layout))
        for node_layout in layout.nodes.values():
            artists["nodes"].append(self._draw_node(ax, node_layout))
        for node_layout, slot in self._resolved_label_slots(layout):
            ha = "right" if slot.right <= node_layout.x else "left"
            style = PubTextStyle(
                fontsize=self.geometry.label_fontsize,
                color="#111111",
                ha=ha,
                va="center",
                linespacing=self.geometry.label_linespacing,
            )
            label = normalize_pub_markdown(node_layout.node.label, width=label_width)
            kwargs = {}
            if self.geometry.label_bbox_facecolor is not None:
                kwargs["bbox"] = {
                    "boxstyle": f"square,pad={float(self.geometry.label_bbox_pad):.3f}",
                    "facecolor": self.geometry.label_bbox_facecolor,
                    "edgecolor": "none",
                    "alpha": float(self.geometry.label_bbox_alpha),
                }
            artists["labels"].append(add_pub_slot_text(ax, slot, label, style=style, width=None, **kwargs))
        artists["layout"] = layout
        return artists
