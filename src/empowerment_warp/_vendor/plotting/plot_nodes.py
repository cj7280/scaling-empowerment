"""Draw a small TikZ-like graph using bundled SVG markers."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from empowerment_warp._vendor.plotting.plot_style import (
    DEFAULT_FIGURE_FORMATS,
    apply_pub_style,
    configure_rendering_environment,
    fontawesome_icon_path,
    high_contrast_rgb01,
    normalize_figure_formats,
    save_pub_figure,
    sharp_geometric_icon_path,
)


DEFAULT_OUTPUT_BASE = Path("icons/node_graph")
DEFAULT_FORMATS = (*DEFAULT_FIGURE_FORMATS, "svg")


def _svg_path_data(svg_path: str | Path) -> str:
    root = ET.parse(svg_path).getroot()
    path_el = root.find("{http://www.w3.org/2000/svg}path")
    if path_el is None:
        path_el = root.find("path")
    if path_el is None or "d" not in path_el.attrib:
        raise ValueError(f"No SVG path data found in {svg_path}")
    return path_el.attrib["d"]


def svg_marker_path(svg_path: str | Path):
    """Convert a simple SVG path into a centered Matplotlib marker path."""
    from fontTools.pens.recordingPen import RecordingPen
    from fontTools.svgLib.path import parse_path
    from matplotlib.path import Path as MplPath

    pen = RecordingPen()
    parse_path(_svg_path_data(svg_path), pen)

    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    subpath_start = (0.0, 0.0)
    for command, points in pen.value:
        if command == "moveTo":
            subpath_start = points[0]
            vertices.append(points[0])
            codes.append(MplPath.MOVETO)
        elif command == "lineTo":
            vertices.append(points[0])
            codes.append(MplPath.LINETO)
        elif command == "curveTo":
            vertices.extend(points)
            codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        elif command == "qCurveTo":
            vertices.extend(points)
            codes.extend([MplPath.CURVE3] * len(points))
        elif command == "closePath":
            vertices.append(subpath_start)
            codes.append(MplPath.CLOSEPOLY)
        else:
            raise ValueError(f"Unsupported SVG path command: {command}")

    path = MplPath(np.asarray(vertices, dtype=float), np.asarray(codes, dtype=np.uint8))
    drawable = path.vertices[path.codes != MplPath.CLOSEPOLY]
    mins = drawable.min(axis=0)
    maxs = drawable.max(axis=0)
    center = 0.5 * (mins + maxs)
    scale = float(np.max(maxs - mins))

    normalized = path.vertices.copy()
    normalized[:, 0] = (normalized[:, 0] - center[0]) / scale
    normalized[:, 1] = -(normalized[:, 1] - center[1]) / scale
    return MplPath(normalized, path.codes)


def fontawesome_marker_path(name: str, *, style: str = "solid"):
    """Convert a bundled Font Awesome SVG into a Matplotlib marker path."""
    return svg_marker_path(fontawesome_icon_path(name, style=style))


def sharp_geometric_marker_path(name: str):
    """Convert a centered sharp geometric SVG into a Matplotlib marker path."""
    return svg_marker_path(sharp_geometric_icon_path(name))


def _mix_rgb01(color, target, amount: float) -> tuple[float, float, float]:
    color = np.asarray(color, dtype=float)
    target = np.asarray(target, dtype=float)
    mixed = (1.0 - float(amount)) * color + float(amount) * target
    return tuple(float(x) for x in np.clip(mixed, 0.0, 1.0))


def node_color_palette() -> dict[str, dict[str, tuple[float, float, float]]]:
    """Return node colors derived from the shared Okabe-Ito palette."""
    robot_edge, star_edge, diamond_edge = high_contrast_rgb01(3)
    return {
        "robot": {
            "face": _mix_rgb01(robot_edge, (1.0, 1.0, 1.0), 0.88),
            "edge": tuple(float(x) for x in robot_edge),
            "icon": _mix_rgb01(robot_edge, (0.0, 0.0, 0.0), 0.12),
        },
        "star": {
            "face": _mix_rgb01(star_edge, (1.0, 1.0, 1.0), 0.88),
            "edge": tuple(float(x) for x in star_edge),
            "icon": _mix_rgb01(star_edge, (0.0, 0.0, 0.0), 0.12),
        },
        "diamond": {
            "face": _mix_rgb01(diamond_edge, (1.0, 1.0, 1.0), 0.90),
            "edge": tuple(float(x) for x in diamond_edge),
            "icon": _mix_rgb01(diamond_edge, (0.0, 0.0, 0.0), 0.12),
        },
    }


def build_node_graph_figure():
    configure_rendering_environment()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    apply_pub_style(plt, compact=True)

    robot_marker = fontawesome_marker_path("robot")
    star_marker = sharp_geometric_marker_path("star")
    diamond_marker = sharp_geometric_marker_path("diamond-border")

    nodes = {
        "collect": {"xy": (-1.85, 0.44), "kind": "robot", "label": "collector"},
        "embed": {"xy": (-0.66, 0.82), "kind": "star", "label": "skills"},
        "plan": {"xy": (0.56, 0.50), "kind": "robot", "label": "planner"},
        "reward": {"xy": (-0.34, -0.38), "kind": "star", "label": "reward"},
        "policy": {"xy": (1.46, -0.30), "kind": "robot", "label": "policy"},
        "eval": {"xy": (0.14, -1.12), "kind": "diamond", "label": "eval"},
    }
    edges = [
        ("collect", "embed"),
        ("embed", "plan"),
        ("collect", "reward"),
        ("reward", "policy"),
        ("plan", "policy"),
        ("policy", "eval"),
        ("eval", "reward"),
    ]

    fig, ax = plt.subplots(figsize=(5.8, 3.7), constrained_layout=True)
    ax.set_facecolor("#fbfbf8")
    fig.patch.set_facecolor("#fbfbf8")

    for src, dst in edges:
        x0, y0 = nodes[src]["xy"]
        x1, y1 = nodes[dst]["xy"]
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=9,
            shrinkA=21,
            shrinkB=21,
            linewidth=1.15,
            color="#777777",
            alpha=0.82,
            connectionstyle="arc3,rad=0.08",
            joinstyle="miter",
            capstyle="butt",
            zorder=1,
        )
        ax.add_patch(arrow)

    colors = node_color_palette()

    for spec in nodes.values():
        x, y = spec["xy"]
        style = colors[spec["kind"]]
        markers = {"robot": robot_marker, "star": star_marker, "diamond": diamond_marker}
        marker = markers[spec["kind"]]

        ax.scatter(
            [x],
            [y],
            s=1480,
            marker="o",
            facecolor=style["face"],
            edgecolor=style["edge"],
            linewidth=1.35,
            zorder=2,
        )
        ax.scatter(
            [x],
            [y],
            s=470,
            marker=marker,
            facecolor=style["icon"],
            edgecolor="none",
            zorder=3,
        )
        ax.text(
            x,
            y - 0.34,
            spec["label"],
            ha="center",
            va="center",
            fontsize=8.2,
            color="#202020",
            zorder=4,
        )

    ax.text(
        -2.1,
        1.46,
        "Icon Nodes",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#202020",
    )
    ax.text(
        -2.1,
        1.25,
        "Font Awesome robots, centered geometric SVG markers",
        ha="left",
        va="center",
        fontsize=8.6,
        color="#575757",
    )

    ax.set_xlim(-2.25, 1.9)
    ax.set_ylim(-1.55, 1.62)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig


def draw_node_graph(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = build_node_graph_figure()
    fig.savefig(output_path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return output_path


def save_node_graph_outputs(
    output_base: str | Path = DEFAULT_OUTPUT_BASE,
    *,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> list[Path]:
    fig = build_node_graph_figure()
    return save_pub_figure(fig, output_base, formats=formats, close=True)


def _parse_formats(raw_formats: str) -> tuple[str, ...]:
    return normalize_figure_formats(raw_formats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="Save one exact output path.",
    )
    parser.add_argument(
        "--output-base",
        default=str(DEFAULT_OUTPUT_BASE),
        help="Base path used for multi-save mode.",
    )
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="Comma-separated formats for multi-save mode. PDF is canonical; PNG is wandb-friendly.",
    )
    args = parser.parse_args()
    if args.output:
        print(draw_node_graph(args.output))
    else:
        for output_path in save_node_graph_outputs(args.output_base, formats=_parse_formats(args.formats)):
            print(output_path)


if __name__ == "__main__":
    main()
