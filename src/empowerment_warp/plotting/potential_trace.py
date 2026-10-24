"""The potential along a rollout and the skill paths drawn from a saved branch trace.

`render_branch_skill_paths` draws the x-y skill branches stored in a video trace archive,
colored by skill direction, with the shared potential prefix drawn once. Layout and type come
from the shared publication contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from empowerment_warp._vendor.plotting.contracts.typography import pub_em_inches, pub_measure_text_inches
from empowerment_warp._vendor.plotting.plot_style import DEFAULT_CYCLIC_CMAP, add_pub_fixed_axes, apply_pub_style, assert_pub_artists_inside_figure, assert_pub_type_scale, assert_pub_uniform_text_size, configure_rendering_environment, pub_uniform_typography_for_width, save_pub_figure, skill_to_rgb01


def _tick_labels(values: np.ndarray, step: float = 5.0) -> tuple[np.ndarray, list[str]]:
    """Round ticks inside the grid (0, 5, 10, 15, 20 for the arena), never the padded edges."""

    lo, hi = float(values[0]), float(values[-1])
    ticks = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step) + 0.0
    ticks[np.abs(ticks) < 1e-9] = 0.0  # never print "-0"
    return ticks, [f"{t:.0f}" for t in ticks]


# ---------------------------------------------------------------------------------------------
# The potential ALONG a rollout (preferred over the x-y field: the value is a function of the pose
# the policy actually has at each point, which a teleported grid cannot reproduce).
# ---------------------------------------------------------------------------------------------

TRACE_TITLE = r"Empowerment $\mathcal{E}(s_t)$"   # the title names the quantity, not the estimator
TRACE_DEFINITION = r"$\mathcal{E}(s_t) = \mathrm{E}_z[V(s_t,\ s_0{=}s_t,\ z)]$"  # recorded in the plot data
TRACE_YLABEL = r"$\mathcal{E}$ (nats)"
PREFIX_COLOR = "#8a8f98"
LANE_COLOR = "#9aa5b1"
MEAN_COLOR = "#0072b2"
HANDOFF_COLOR = "#d55e00"


def _trace_inputs(npz_path, plt, example_lanes):
    npz_path = Path(npz_path).expanduser().resolve()
    arrays = dict(np.load(npz_path))
    manifest = json.loads(npz_path.with_suffix(".json").read_text())
    pot = np.maximum(np.asarray(arrays["potential"], dtype=np.float64), 0.0)          # (T, N)
    xy = np.asarray(arrays["agent_xy"], dtype=np.float64)            # (T, N, 2)
    ball = np.asarray(arrays["ball_xy"], dtype=np.float64)           # (T, N, 2)
    T, N = pot.shape
    handoff = int(manifest.get("potential_steps", 0))
    # Example lanes SWEEP the circle of skills (lane i carries library skill i % K, so the first
    # six lanes would be a 72-degree arc of a 30-direction library and read as collapsed).
    lane_skills = np.asarray(arrays["lane_skills"], dtype=np.float64)
    K = int(manifest.get("num_skills", N))
    picked = np.unique(np.round(np.linspace(0, K, min(example_lanes, K), endpoint=False)).astype(int))
    lanes = [int(np.argmax(np.arange(N) % K == k)) for k in picked]
    angle = np.degrees(np.arctan2(lane_skills[:, 1], lane_skills[:, 0])) % 360.0
    cmap = plt.get_cmap(DEFAULT_CYCLIC_CMAP)
    lane_color = {lane: cmap(angle[lane] / 360.0) for lane in lanes}
    return npz_path, arrays, manifest, pot, xy, ball, T, N, K, handoff, lanes, angle, cmap, lane_color


def _legend_band_height(plt, typography, rows: int, em: float) -> float:
    return rows * pub_measure_text_inches(plt, "Soccer ball", style=typography.legend)[1] + (rows + 0.6) * 0.5 * em


def _render_skill_paths(xy, ball, handoff, lanes, lane_color, output_base, *, title=None,
                        width_in=6.5, tier="board"):
    """Shared path layout; callers provide the saved trajectories and their colors."""
    configure_rendering_environment()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerBase
    from empowerment_warp._vendor.plotting.contracts.core import PubBounds
    from empowerment_warp._vendor.plotting.plot_nodes import fontawesome_marker_path

    apply_pub_style(plt, compact=True, use_tex=False)
    ball_marker = fontawesome_marker_path("futbol")
    typography = pub_uniform_typography_for_width(width_in, tier=tier)
    pt = float(typography.axis_label.fontsize)
    em = pub_em_inches(pt)
    tick_len = float(plt.rcParams["xtick.major.size"]) / 72.0
    measure = lambda text: pub_measure_text_inches(plt, text, style=typography.axis_label)
    margin = 0.6 * em
    T = xy.shape[0]
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    produced = []
    # ---------------- figure 2: the x-y paths, colored by skill ----------------
    x_ticks, x_labels = _tick_labels(np.array([-1.8, 21.8]))
    left = margin + measure("Y")[1] + 0.45 * em + max(measure(l)[0] for l in x_labels) + 0.25 * em + tick_len
    right = margin + 0.5 * max(measure(label)[0] for label in x_labels)
    panel_w = width_in - left - right
    x_gutter = measure("X")[1] + 0.45 * em + max(measure(l)[1] for l in x_labels) + 0.25 * em + tick_len
    legend_labels = (["Potential"] if handoff > 0 else []) + ["Effective policy"]
    legend_columns = len(legend_labels)
    legend_h = _legend_band_height(plt, typography, 1, em)
    xy_b = PubBounds(left, margin + legend_h + 0.6 * em + x_gutter, panel_w, panel_w)
    title_y = xy_b.y + xy_b.h + 0.6 * em
    path_title = "Skill paths" if title is None else f"Skill paths: {title}"
    fig_h = title_y + measure(path_title)[1] + margin
    fig = plt.figure(figsize=(width_in, fig_h))
    ax_xy = add_pub_fixed_axes(fig, width_in, fig_h, xy_b.x, xy_b.y, xy_b.w, xy_b.h, label="x-y paths")
    lo, hi = -1.8, 21.8
    ax_xy.set_xlim(lo, hi); ax_xy.set_ylim(lo, hi); ax_xy.set_aspect("equal")
    h = max(min(handoff, T - 1), 0)
    # The skill lanes BRANCH from lane 0's state at the onset (collect_locomotion_branch_trace
    # broadcasts lane 0 via gather_env_state_lanes), so lane 0's potential prefix is the one every skill
    # segment continues from. Draw that prefix ONCE; the other lanes' own pre-onset rollouts are
    # discarded by the branch and drawing them showed spurious "splits" in the potential phase.
    branch_lane = 0
    if h > 0:
        ax_xy.plot(xy[: h + 1, branch_lane, 0], xy[: h + 1, branch_lane, 1], color=PREFIX_COLOR, lw=1.4, alpha=0.9, zorder=3)
    for lane in lanes:
        # Plain colored segments, no dark outline stroke.
        ax_xy.plot(xy[h:, lane, 0], xy[h:, lane, 1], color=lane_color[lane], lw=2.4, zorder=4)
        ax_xy.plot(xy[-1, lane, 0], xy[-1, lane, 1], "o", ms=pt * 0.4, mfc=lane_color[lane], mec="#111111", mew=0.6, zorder=6)
        if 0 < handoff < T:
            ax_xy.plot(xy[handoff, lane, 0], xy[handoff, lane, 1], "o", ms=pt * 0.6, mfc="#ffffff", mec="#111111", mew=1.0, zorder=5)
    ball0 = ball[0, lanes[0]]
    if np.all(np.isfinite(ball0)) and np.any(ball0 != 0):
        ax_xy.plot(ball0[0], ball0[1], marker=ball_marker, ms=pt * 1.1, mfc="#111111", mec="none", ls="none", zorder=6)
    ax_xy.set_xticks(x_ticks); ax_xy.set_xticklabels(x_labels, fontsize=pt)
    ax_xy.set_yticks(x_ticks); ax_xy.set_yticklabels(x_labels, fontsize=pt)
    ax_xy.tick_params(labelsize=pt)
    ax_xy.set_xlabel("X", fontsize=pt); ax_xy.set_ylabel("Y", fontsize=pt)
    fig.text((xy_b.x + xy_b.w / 2.0) / width_in, title_y / fig_h, path_title, ha="center", va="bottom", fontsize=pt, color="#111111")
    class StackedLines:
        def __init__(self, colors):
            self.colors = tuple(colors)

    class StackedLinesHandler(HandlerBase):
        def create_artists(
            self, legend, original, xdescent, ydescent, width, height, fontsize, transform
        ):
            # One flush BLOCK of stripes: contiguous filled rectangles that tile the handle box
            # top to bottom, no gaps; separate Line2D stripes would read as floating lines.
            # Drawn in the legend's own order.
            from matplotlib.patches import Rectangle

            # Center the block where HandlerLine2D centers a line handle, y = (height - ydescent) / 2,
            # NOT at ydescent + height / 2: tiling [ydescent, ydescent + height] puts the block one
            # descent above the "Potential" line and the label.
            count = max(len(original.colors), 1)
            stripe = height / count
            center = (height - ydescent) / 2.0
            bottom = center - height / 2.0
            return [
                Rectangle(
                    (xdescent, bottom + index * stripe), width, stripe,
                    facecolor=color, edgecolor="none", linewidth=0, transform=transform,
                )
                for index, color in enumerate(reversed(original.colors))
            ]

    skill_handle = StackedLines([lane_color[lane] for lane in lanes])
    handles = ([Line2D([], [], color=PREFIX_COLOR, lw=1.4, label="Potential")]
               if handoff > 0 else []) + [skill_handle]
    legend = fig.legend(handles=handles, loc="lower center", ncol=legend_columns, fontsize=pt, bbox_to_anchor=(0.5, margin / fig_h),
                        labels=legend_labels, handler_map={StackedLines: StackedLinesHandler()},
                        bbox_transform=fig.transFigure, handletextpad=0.5, columnspacing=1.0, handlelength=1.6, borderaxespad=0.0, frameon=False)
    assert_pub_type_scale(fig, pt, tier=tier); assert_pub_uniform_text_size(fig, pt)
    assert_pub_artists_inside_figure(fig, [ax_xy, ax_xy.xaxis.label, ax_xy.yaxis.label, legend, *legend.get_texts()])
    path_base = output_base.with_name(output_base.name + "_path")
    produced += save_pub_figure(fig, path_base, close=True)

    return produced


def render_branch_skill_paths(trace_path, output_base, *, width_in=6.5, tier="board"):
    """Draw precisely the skill branches in the video, without another rollout."""
    source = Path(trace_path).resolve()
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        skills = np.asarray(archive["skills"])
        indices = np.asarray(archive["skill_indices"])
        xy = np.asarray(archive["agent_positions"]).swapaxes(0, 1)
        ball = (np.asarray(archive["ball_positions"]).swapaxes(0, 1)
                if "ball_positions" in archive else np.zeros_like(xy))
    lanes = np.flatnonzero(indices >= 0).tolist()
    colors = {lane: tuple(skill_to_rgb01(skills[lane])) for lane in lanes}
    handoff = int(metadata["activation_index"])
    produced = _render_skill_paths(xy, ball, handoff, lanes, colors, output_base,
                                   width_in=width_in, tier=tier)
    data = {
        "source_trace": str(source), "source_metadata": metadata,
        "example_lanes": lanes, "handoff_step": handoff,
        "example_lane_skill_vectors": {lane: skills[lane].tolist() for lane in lanes},
        "example_lane_colors_rgb": colors,
        "agent_xy": xy[:, lanes].tolist(), "ball_xy": ball[:, lanes].tolist(),
        "potential_prefix_xy": xy[:handoff + 1, 0].tolist(),
        "selection": "all video skill branches; excludes the potential continuation",
    }
    path = Path(output_base).with_name(Path(output_base).name + "_path_plot_data.json")
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return [*produced, path]
