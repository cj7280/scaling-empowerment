"""Lightweight pub-style video render helpers.

These helpers are intentionally independent of MuJoCo. They exercise the same
palette contracts used by the real renderers: Okabe-Ito for categorical trails,
viridis for scalar fields, and twilight for cyclic phase colors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from empowerment_warp._vendor.plotting.contracts.media import (
    PUB_ENV_ANIMATION_FPS,
    PubEnvFrameBundle,
    PubEnvVideoSpec,
    assert_pub_env_symmetry_contract,
    assert_pub_env_video_contract,
    pub_env_ease,
    pub_env_lerp,
    pub_env_motion_path,
    pub_env_phase_at,
    pub_env_progress_values,
    pub_env_snapshot_frames,
    pub_env_timing,
    save_pub_env_base_animation_bundle,
    save_pub_env_frame_bundle,
    save_pub_frame_bundle,
)
from empowerment_warp._vendor.plotting.plot_style import (
    DEFAULT_CYCLIC_CMAP,
    DEFAULT_SCALAR_CMAP,
    PUB_LEGEND_LAYOUT,
    configure_rendering_environment,
    high_contrast_rgb01,
    materialize_pub_font,
    prepare_pub_style,
    pub_render_legend_position,
    pub_legend_box_style,
    pub_rcparams,
    skill_to_rgb01,
)




def _canvas_unit(width: int, height: int) -> float:
    return float(max(1, min(int(width), int(height))))


def _rel_px(width: int, height: int, fraction: float, *, min_px: int = 1) -> int:
    return max(int(min_px), int(round(_canvas_unit(width, height) * float(fraction))))


def _rgb01_to_rgb255(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    return tuple(int(round(255 * float(x))) for x in np.clip(rgb, 0.0, 1.0))


def _load_font(size_px: float, bold: bool = False):
    font_path = materialize_pub_font(bold=bold)
    return ImageFont.truetype(font_path, size=max(1, int(round(float(size_px)))))


def _draw_dashed_line(draw, xy, *, fill, width: int, dash: float | None = None, gap: float | None = None):
    x0, y0, x1, y1 = xy
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length <= 0:
        return
    dash = max(1.0, 2.3 * float(width)) if dash is None else float(dash)
    gap = max(1.0, 1.45 * float(width)) if gap is None else float(gap)
    direction = np.array([(x1 - x0) / length, (y1 - y0) / length], dtype=np.float32)
    start = np.array([x0, y0], dtype=np.float32)
    distance = 0.0
    while distance < length:
        seg_start = start + direction * distance
        seg_end = start + direction * min(distance + dash, length)
        draw.line((tuple(seg_start), tuple(seg_end)), fill=fill, width=width)
        distance += dash + gap


def _diamond_points(center, radius: float):
    cx, cy = center
    return [(cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)]


def _draw_diamond(draw, center, radius: int, *, fill, backing):
    outer = _diamond_points(center, float(radius) * 1.28)
    inner = _diamond_points(center, max(1.0, float(radius) * 0.74))
    draw.polygon(outer, fill=backing)
    draw.polygon(inner, fill=fill)


def pub_text_bbox(draw, text: str, font) -> tuple[float, float, float, float]:
    """Return a text bbox measured at the origin as floats."""
    return tuple(float(value) for value in draw.textbbox((0, 0), str(text), font=font))


def pub_text_y_centered_on(
    draw,
    text: str,
    font,
    center_y: float,
    *,
    max_error_px: float = 1.0,
    role: str = "Text",
) -> float:
    """Return the y coordinate that visually centers text on a row center."""
    label = str(text)
    bbox = pub_text_bbox(draw, label, font)
    text_height = float(bbox[3] - bbox[1])
    text_y = float(center_y) - text_height / 2.0 - float(bbox[1])
    placed_bbox = draw.textbbox((0, text_y), label, font=font)
    placed_center_y = 0.5 * (float(placed_bbox[1]) + float(placed_bbox[3]))
    if abs(placed_center_y - float(center_y)) > float(max_error_px):
        raise AssertionError(
            f"{role} {label!r} is not vertically centered: "
            f"target={center_y:.3f}, actual={placed_center_y:.3f}"
        )
    return text_y


def _text_y_centered_on(draw, label: str, font, center_y: float) -> float:
    return pub_text_y_centered_on(draw, label, font, center_y, role="Legend label")


def pub_ellipsize_text(draw, text: str, font, max_width: float, *, ellipsis: str = "...") -> str:
    """Shorten text to fit a measured pixel width without shifting its slot."""
    text = str(text)
    max_width = float(max_width)
    if max_width <= 0:
        return ""
    bbox = pub_text_bbox(draw, text, font)
    if bbox[2] - bbox[0] <= max_width:
        return text
    ellipsis_bbox = pub_text_bbox(draw, ellipsis, font)
    ellipsis_width = ellipsis_bbox[2] - ellipsis_bbox[0]
    if ellipsis_width > max_width:
        return ""

    lo = 0
    hi = len(text)
    best = ellipsis
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + ellipsis
        bbox = pub_text_bbox(draw, candidate, font)
        if bbox[2] - bbox[0] <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def draw_pub_text_in_slot(
    draw,
    text: str,
    slot,
    font,
    *,
    fill,
    align: str = "center",
    valign: str = "center",
    ellipsize: bool = True,
    max_width: float | None = None,
    max_center_error_px: float = 1.0,
) -> dict[str, Any]:
    """Draw text inside a fixed slot and return measured layout metadata.

    Use this for animation headers, counters, panel titles, and PIL overlays
    whose position must remain stable frame-to-frame as values change.
    """
    x0, y0, x1, y1 = (float(value) for value in slot)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    available_width = max(0.0, x1 - x0)
    if max_width is not None:
        available_width = min(available_width, max(0.0, float(max_width)))
    display_text = pub_ellipsize_text(draw, text, font, available_width) if ellipsize else str(text)
    bbox = pub_text_bbox(draw, display_text, font)
    text_width = float(bbox[2] - bbox[0])
    text_height = float(bbox[3] - bbox[1])
    if text_width > available_width + 1.0:
        raise AssertionError(
            f"Text {display_text!r} is wider than its slot: "
            f"text_width={text_width:.3f}, slot_width={available_width:.3f}."
        )

    align = str(align)
    if align == "left":
        text_x = x0 - float(bbox[0])
    elif align == "center":
        text_x = 0.5 * (x0 + x1) - 0.5 * text_width - float(bbox[0])
    elif align == "right":
        text_x = x1 - text_width - float(bbox[0])
    else:
        raise ValueError(f"Unsupported text slot align: {align!r}")

    valign = str(valign)
    if valign == "top":
        text_y = y0 - float(bbox[1])
    elif valign == "center":
        text_y = pub_text_y_centered_on(
            draw,
            display_text,
            font,
            0.5 * (y0 + y1),
            max_error_px=max_center_error_px,
            role="Slot text",
        )
    elif valign == "bottom":
        text_y = y1 - text_height - float(bbox[1])
    else:
        raise ValueError(f"Unsupported text slot valign: {valign!r}")

    draw.text((text_x, text_y), display_text, fill=fill, font=font)
    placed_bbox = tuple(float(value) for value in draw.textbbox((text_x, text_y), display_text, font=font))
    tolerance = 1.0
    if (
        placed_bbox[0] < x0 - tolerance
        or placed_bbox[1] < y0 - tolerance
        or placed_bbox[2] > x1 + tolerance
        or placed_bbox[3] > y1 + tolerance
    ):
        raise AssertionError(
            f"Text {display_text!r} overflowed its slot: "
            f"bbox={placed_bbox}, slot={(x0, y0, x1, y1)}."
        )
    return {
        "text": display_text,
        "slot": (x0, y0, x1, y1),
        "bbox": placed_bbox,
        "align": align,
        "valign": valign,
        "center": (0.5 * (placed_bbox[0] + placed_bbox[2]), 0.5 * (placed_bbox[1] + placed_bbox[3])),
    }


def _legend_row_half_height(row, text_height: float) -> float:
    marker_half_height = {
        "circle": 0.35,
        "diamond": 0.42,
        "line": 0.16,
        "dashed": 0.16,
        "stacked_lines": 0.45,  # a 4-stripe block is taller than a line; keep rows from overlapping
    }.get(row["kind"], 0.16) * float(text_height)
    return max(0.5 * float(text_height), marker_half_height)


def _legend_layout(
    draw,
    rows,
    font,
    *,
    x0: int,
    y0: int,
    text_x_offset: int,
    font_px: float,
    row_gap_units: float = PUB_LEGEND_LAYOUT["video_row_gap_units"],
):
    text_bboxes = [draw.textbbox((0, 0), row["label"], font=font) for row in rows]
    text_widths = [bbox[2] - bbox[0] for bbox in text_bboxes]
    text_heights = [bbox[3] - bbox[1] for bbox in text_bboxes]
    row_half_heights = [
        _legend_row_half_height(row, text_height)
        for row, text_height in zip(rows, text_heights)
    ]
    row_unit = max([2.0 * half_height for half_height in row_half_heights] + [float(font_px)])
    pad_y = max(12.0, PUB_LEGEND_LAYOUT["video_pad_y_units"] * row_unit)
    row_gap = max(8.0, float(row_gap_units) * row_unit)

    centers = []
    cursor = float(y0) + pad_y
    last_bottom = cursor
    for half_height in row_half_heights:
        center_y = cursor + half_height
        centers.append(center_y)
        last_bottom = center_y + half_height
        cursor = last_bottom + row_gap

    pad_x = max(12.0, PUB_LEGEND_LAYOUT["video_pad_x_units"] * row_unit)
    box_w = int(round(text_x_offset + max(text_widths, default=0) + pad_x))
    box_h = int(round(last_bottom + pad_y - float(y0)))
    box = (x0, y0, x0 + box_w, y0 + box_h)
    return box, centers


def _scalar_field_rgb(width: int, height: int, phase: float):
    from matplotlib import colormaps

    xs = np.linspace(-2.8, 2.8, int(width), dtype=np.float32)
    ys = np.linspace(-1.9, 1.9, int(height), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    field = np.exp(-0.22 * (xx**2 + yy**2)) * np.cos(1.8 * xx - phase)
    field += 0.23 * np.sin(2.2 * yy + 0.7 * phase)
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    rgb = colormaps[DEFAULT_SCALAR_CMAP](field)[..., :3]
    rgb = (255.0 * rgb).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _builderbench_scene_rgb(width: int, height: int, phase: float):
    """Draw a lightweight cube-pick-arm scene for BuilderBench legend smoke tests."""
    unit = _canvas_unit(width, height)
    image = Image.new("RGBA", (int(width), int(height)), (54, 57, 62, 255))
    draw = ImageDraw.Draw(image, "RGBA")

    grid_y = max(12, int(round(0.040 * unit)))
    grid_x = max(16, int(round(0.053 * unit)))
    for row in range(0, int(height), grid_y):
        alpha = int(20 + 28 * row / max(int(height), 1))
        draw.line((0, row, width, row), fill=(84, 88, 95, alpha), width=1)
    for col in range(0, int(width), grid_x):
        draw.line((col, int(height * 0.44), col, height), fill=(84, 88, 95, 34), width=1)

    workspace = (
        int(width * 0.26),
        int(height * 0.30),
        int(width * 0.78),
        int(height * 0.82),
    )
    draw.rounded_rectangle(
        workspace,
        radius=_rel_px(width, height, 0.013, min_px=4),
        fill=(67, 70, 76, 235),
        outline=(115, 120, 128, 180),
        width=_rel_px(width, height, 0.003, min_px=1),
    )

    base = (int(width * 0.52), int(height * 0.75))
    elbow = (
        int(width * (0.47 + 0.035 * np.sin(phase))),
        int(height * (0.55 + 0.025 * np.cos(phase))),
    )
    wrist = (
        int(width * (0.60 + 0.040 * np.cos(phase))),
        int(height * (0.42 + 0.035 * np.sin(phase))),
    )
    gripper_span = _rel_px(width, height, 0.037, min_px=12)
    gripper_drop = _rel_px(width, height, 0.033, min_px=10)
    gripper_finger = _rel_px(width, height, 0.047, min_px=14)
    gripper_left = (wrist[0] - gripper_span, wrist[1] + gripper_drop)
    gripper_right = (wrist[0] + gripper_span, wrist[1] + gripper_drop)
    arm_shadow = (21, 23, 26, 95)
    arm_color = (187, 191, 198, 235)
    joint_color = (222, 224, 229, 255)

    link_width = _rel_px(width, height, 0.040, min_px=10)
    forearm_width = _rel_px(width, height, 0.033, min_px=8)
    for shift in (_rel_px(width, height, 0.013, min_px=3), _rel_px(width, height, 0.007, min_px=2)):
        draw.line((base[0] + shift, base[1] + shift, elbow[0] + shift, elbow[1] + shift), fill=arm_shadow, width=link_width)
        draw.line((elbow[0] + shift, elbow[1] + shift, wrist[0] + shift, wrist[1] + shift), fill=arm_shadow, width=forearm_width)
    draw.line((base, elbow), fill=arm_color, width=link_width)
    draw.line((elbow, wrist), fill=arm_color, width=forearm_width)
    joint_specs = (
        (base, _rel_px(width, height, 0.030, min_px=8)),
        (elbow, _rel_px(width, height, 0.027, min_px=7)),
        (wrist, _rel_px(width, height, 0.022, min_px=6)),
    )
    for center, radius in joint_specs:
        draw.ellipse(
            (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
            fill=joint_color,
            outline=(112, 118, 128, 255),
            width=_rel_px(width, height, 0.003, min_px=1),
        )

    gripper_width = _rel_px(width, height, 0.013, min_px=4)
    draw.line((wrist, gripper_left), fill=(229, 231, 235, 255), width=gripper_width)
    draw.line((wrist, gripper_right), fill=(229, 231, 235, 255), width=gripper_width)
    draw.line((gripper_left[0], gripper_left[1], gripper_left[0] - gripper_drop, gripper_left[1] + gripper_finger), fill=(229, 231, 235, 255), width=max(2, gripper_width - 1))
    draw.line((gripper_right[0], gripper_right[1], gripper_right[0] + gripper_drop, gripper_right[1] + gripper_finger), fill=(229, 231, 235, 255), width=max(2, gripper_width - 1))

    cube_center = (
        int(width * (0.62 + 0.035 * np.sin(phase + 0.5))),
        int(height * (0.68 + 0.020 * np.cos(phase))),
    )
    cube_size = _rel_px(width, height, 0.063, min_px=18)
    cube = (
        cube_center[0] - cube_size,
        cube_center[1] - cube_size,
        cube_center[0] + cube_size,
        cube_center[1] + cube_size,
    )
    draw.rounded_rectangle(
        cube,
        radius=_rel_px(width, height, 0.008, min_px=3),
        fill=(214, 92, 18, 245),
        outline=(255, 244, 235, 220),
        width=_rel_px(width, height, 0.003, min_px=1),
    )
    cube_inset = _rel_px(width, height, 0.015, min_px=4)
    draw.line(
        (cube[0] + cube_inset, cube[1] + cube_inset, cube[2] - cube_inset, cube[1] + cube_inset),
        fill=(252, 166, 89, 125),
        width=_rel_px(width, height, 0.005, min_px=1),
    )

    return image.convert("RGB")


def _world_to_pixel(points, width: int, height: int, margin: int = 36):
    points = np.asarray(points, dtype=np.float32)
    x = margin + (points[..., 0] + 2.4) / 4.8 * (width - 2 * margin)
    y = margin + (1.0 - (points[..., 1] + 1.45) / 2.9) * (height - 2 * margin)
    return np.stack([x, y], axis=-1)


def _trajectory_points(frame_idx: int, trace_idx: int, frame_count: int, trail_steps: int):
    phases = np.linspace(
        max(0, frame_idx - trail_steps + 1),
        frame_idx,
        num=min(frame_idx + 1, trail_steps),
        dtype=np.float32,
    )
    t = 2.0 * np.pi * (phases / max(frame_count - 1, 1))
    offset = 2.0 * np.pi * trace_idx / 3.0
    radius = 1.0 + 0.18 * trace_idx
    x = radius * np.cos(t + offset) + 0.16 * np.sin(2.0 * t)
    y = 0.72 * radius * np.sin(t + offset) + 0.12 * np.cos(3.0 * t)
    return np.stack([x, y], axis=-1)


def _legend_rows(legend_style: str, okabe_rgbs, phase_rgb, *, include_potential_policy: bool = True):
    """Legend rows for a video overlay.

    ``include_potential_policy`` exists because the "Potential policy" row
    describes a trajectory segment that only exists when the run has a potential
    phase (``potential_steps > 0``). A run with ``potential_steps=0`` has the skill on
    from step 0, draws no Potential segment, and advertising the row anyway tells
    the viewer the video contains something it does not.
    """
    style_name = pub_legend_box_style(legend_style)["name"]
    if style_name == "builderbench_gray":
        rows = []
        if include_potential_policy:
            rows.append(
                {
                    "label": "Potential policy",
                    "color": (244, 244, 244),
                    "kind": "line",
                }
            )
        rows += [
            {
                "label": "Effective policy",
                "color": _rgb01_to_rgb255(okabe_rgbs[0]),
                "colors": [_rgb01_to_rgb255(rgb) for rgb in okabe_rgbs],
                "kind": "stacked_lines",
            },
            {
                "label": "Skill onset",
                "color": (244, 244, 244),
                "kind": "circle",
            },
        ]
        return rows

    return [
        {
            "label": "Okabe-Ito trails",
            "color": _rgb01_to_rgb255(okabe_rgbs[0]),
            "kind": "line",
        },
        {
            "label": "Viridis field",
            "color": (68, 1, 84),
            "kind": "line",
        },
        {
            "label": "Twilight phase",
            "color": _rgb01_to_rgb255(phase_rgb),
            "kind": "circle",
        },
    ]


def add_pub_overlay_legend(
    frames, colors, *, font_px, legend_style="builderbench_gray", include_potential_policy=True,
):
    """Overlay one measured publication legend without changing frame dimensions."""
    values = np.asarray(frames, dtype=np.uint8)
    if values.ndim != 4 or values.shape[-1] != 3 or not len(values):
        raise ValueError("Expected nonempty [frames,height,width,3] RGB frames.")
    colors = np.asarray(colors)
    if colors.ndim != 2 or not len(colors):
        raise ValueError("A publication legend needs at least one color.")
    width = values.shape[2]
    font = _load_font(font_px, bold=False)
    rows = _legend_rows(legend_style, colors, colors[0], include_potential_policy=include_potential_policy)
    scratch = ImageDraw.Draw(Image.new("RGB", (width, 1)))
    box, _ = _legend_layout(
        scratch, rows, font, x0=0, y0=0,
        text_x_offset=round(PUB_LEGEND_LAYOUT["video_text_x_offset_units"] * font_px),
        font_px=font_px,
    )
    pad = int(round(font_px))
    if box[2] + 2 * pad > width:
        raise ValueError("Legend does not fit at the requested publication type scale.")
    if box[3] + 2 * pad > values.shape[1]:
        raise ValueError("Legend is taller than the rendered scene.")
    output = np.empty_like(values)
    for index, frame in enumerate(values):
        canvas = Image.fromarray(frame).convert('RGBA')
        overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
        _draw_legend(ImageDraw.Draw(overlay), width, colors, colors[0], height=canvas.height,
                     legend_style=legend_style, font_px=font_px,
                     include_potential_policy=include_potential_policy, anchor="upper_left")
        output[index] = np.asarray(Image.alpha_composite(canvas, overlay).convert('RGB'))
    return output


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    width: int,
    okabe_rgbs,
    phase_rgb,
    *,
    height: int | None = None,
    legend_style: str = "white",
    anchor: str = "upper_left",
    anchor_rect=None,
    border_rects=(),
    include_potential_policy: bool = True,
    font_px: float | None = None,
):
    box_style = pub_legend_box_style(legend_style)
    if font_px is not None:
        legend_px = float(font_px)
    else:
        # Without `font_px` the size is a CONSTANT -- 8.5 pt at 200 dpi = 23.6 px --
        # with no knowledge of how wide the frame is, so the legend occupies a
        # different fraction of a 640 px BuilderBench frame, a 960 px AntSoccer
        # frame and a 1164 px Craftax frame, and cannot follow a publication
        # tier at all. Pass `font_px` from `pub_overlay_px_for_frame` instead.
        style = pub_rcparams(compact=True)
        legend_px = float(style["legend.fontsize"]) * float(style["figure.dpi"]) / 72.0
    text_font = _load_font(legend_px, bold=False)
    rows = _legend_rows(box_style["name"], okabe_rgbs, phase_rgb,
                        include_potential_policy=include_potential_policy)
    text_x_offset = int(round(PUB_LEGEND_LAYOUT["video_text_x_offset_units"] * legend_px))
    height = int(height if height is not None else width)
    probe_box, _ = _legend_layout(
        draw,
        rows,
        text_font,
        x0=0,
        y0=0,
        text_x_offset=text_x_offset,
        font_px=legend_px,
    )
    x0, y0 = pub_render_legend_position(
        int(width),
        int(height),
        probe_box[2] - probe_box[0],
        probe_box[3] - probe_box[1],
        anchor=anchor,
        anchor_rect=anchor_rect,
        border_rects=border_rects,
        font_px=legend_px,
    )
    box, centers = _legend_layout(
        draw,
        rows,
        text_font,
        x0=x0,
        y0=y0,
        text_x_offset=text_x_offset,
        font_px=legend_px,
    )
    draw.rounded_rectangle(
        box,
        radius=4,
        fill=box_style["fill_rgba"],
        outline=box_style["outline_rgba"],
        width=max(1, int(round(float(box_style["outline_width"])))),
    )

    for row, center_y in zip(rows, centers):
        label = row["label"]
        color = row["color"]
        rgba = color + (255,)
        handle_x0 = x0 + PUB_LEGEND_LAYOUT["video_handle_start_units"] * legend_px
        handle_x1 = x0 + PUB_LEGEND_LAYOUT["video_handle_end_units"] * legend_px
        marker_cx = x0 + PUB_LEGEND_LAYOUT["video_marker_center_units"] * legend_px
        line_width = max(2, int(round(PUB_LEGEND_LAYOUT["video_line_width_units"] * legend_px)))
        diamond_radius = max(3, int(round(PUB_LEGEND_LAYOUT["video_diamond_radius_units"] * legend_px)))
        if row["kind"] == "dashed":
            _draw_dashed_line(draw, (handle_x0, center_y, handle_x1, center_y), fill=rgba, width=line_width)
        elif row["kind"] == "stacked_lines":
            # One solid BLOCK of contiguous stripes (no gaps), centered on the row like the
            # single-line handle -- not separated thin lines floating above the text baseline.
            stack_colors = [tuple(color) for color in row.get("colors", (color,))]
            stack_colors = stack_colors if stack_colors else [color]
            stack_count = len(stack_colors)
            stripe_h = max(line_width, int(round(0.18 * legend_px)))
            block_h = stripe_h * stack_count
            top = int(round(center_y - 0.5 * block_h))
            for index, line_color in enumerate(stack_colors):
                y_top = top + index * stripe_h
                draw.rectangle(
                    (handle_x0, y_top, handle_x1, y_top + stripe_h - 1),
                    fill=tuple(line_color) + (255,),
                )
        elif row["kind"] == "line":
            draw.line((handle_x0, center_y, handle_x1, center_y), fill=rgba, width=line_width)
        if row["kind"] == "circle":
            radius = max(3, int(round(0.30 * legend_px)))
            draw.ellipse((marker_cx - radius, center_y - radius, marker_cx + radius, center_y + radius), fill=rgba)
        elif row["kind"] == "diamond":
            _draw_diamond(
                draw,
                (marker_cx, center_y),
                diamond_radius,
                fill=rgba,
                backing=row.get("backing", (255, 255, 255, 255)),
            )
        text_y = _text_y_centered_on(draw, label, text_font, center_y)
        draw.text((x0 + text_x_offset, text_y), label, font=text_font, fill=box_style["text_rgba"])


def format_pub_metric_value(value: int | float, width: int) -> str:
    """Format integer-like video metrics with stable zero padding."""
    value = int(round(float(value)))
    width = max(1, int(width))
    if value < 0:
        return "-" + f"{abs(value):0{max(1, width - 1)}d}"
    return f"{value:0{width}d}"


def draw_pub_metric_badge(
    image,
    rows,
    *,
    anchor: str = "top_right",
    margin_px: int | None = None,
    font_px: int | None = None,
    pad_x: int | None = None,
    pad_y: int | None = None,
    col_gap: int | None = None,
    row_gap: int | None = None,
    style: str = "white",
    bold: bool = False,
):
    """Draw a compact metric badge with right-aligned values on a PIL image."""
    width, height = image.size
    unit = _canvas_unit(width, height)
    font_px = _rel_px(width, height, 0.025, min_px=10) if font_px is None else int(font_px)
    margin_px = _rel_px(width, height, 0.018, min_px=6) if margin_px is None else int(margin_px)
    pad_x = _rel_px(width, height, 0.014, min_px=6) if pad_x is None else int(pad_x)
    pad_y = _rel_px(width, height, 0.010, min_px=5) if pad_y is None else int(pad_y)
    col_gap = _rel_px(width, height, 0.020, min_px=9) if col_gap is None else int(col_gap)
    row_gap = _rel_px(width, height, 0.006, min_px=2) if row_gap is None else int(row_gap)
    radius = max(4, int(round(0.018 * unit)))

    normalized_rows = []
    for row in rows:
        if isinstance(row, dict):
            label = row["label"]
            value = row["value"]
        else:
            label, value = row
        normalized_rows.append((str(label), str(value)))
    if not normalized_rows:
        return None

    draw = ImageDraw.Draw(image, "RGBA")
    font = _load_font(font_px, bold=bold)
    label_boxes = [draw.textbbox((0, 0), label, font=font) for label, _ in normalized_rows]
    value_boxes = [draw.textbbox((0, 0), value, font=font) for _, value in normalized_rows]
    label_w = max(box[2] - box[0] for box in label_boxes)
    value_w = max(box[2] - box[0] for box in value_boxes)
    row_h = max(
        max(box[3] - box[1] for box in label_boxes),
        max(box[3] - box[1] for box in value_boxes),
    )
    text_w = label_w + col_gap + value_w
    text_h = len(normalized_rows) * row_h + (len(normalized_rows) - 1) * row_gap
    box_w = text_w + 2 * pad_x
    box_h = text_h + 2 * pad_y

    anchor = str(anchor)
    if anchor == "top_right":
        x0 = int(width) - int(margin_px) - int(box_w)
        y0 = int(margin_px)
    elif anchor == "top_left":
        x0 = int(margin_px)
        y0 = int(margin_px)
    elif anchor == "bottom_right":
        x0 = int(width) - int(margin_px) - int(box_w)
        y0 = int(height) - int(margin_px) - int(box_h)
    elif anchor == "bottom_left":
        x0 = int(margin_px)
        y0 = int(height) - int(margin_px) - int(box_h)
    else:
        raise ValueError(f"Unsupported metric badge anchor: {anchor!r}")
    x1 = x0 + int(box_w)
    y1 = y0 + int(box_h)

    box_style = pub_legend_box_style(style)
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=radius,
        fill=box_style["fill_rgba"],
        outline=box_style["outline_rgba"],
        width=max(1, int(round(float(box_style["outline_width"])))),
    )
    first_center_y = y0 + pad_y + 0.5 * row_h
    value_right = x1 - pad_x
    for row_idx, ((label, value), value_box) in enumerate(zip(normalized_rows, value_boxes, strict=True)):
        center_y = first_center_y + row_idx * (row_h + row_gap)
        label_text_y = _text_y_centered_on(draw, label, font, center_y)
        value_text_y = _text_y_centered_on(draw, value, font, center_y)
        draw.text((x0 + pad_x, label_text_y), label, fill=box_style["text_rgba"], font=font)
        value_x = value_right - (value_box[2] - value_box[0])
        draw.text((value_x, value_text_y), value, fill=box_style["text_rgba"], font=font)
    return (x0, y0, x1, y1)


def create_pub_video_render_frames(
    *,
    frame_count: int = 48,
    width: int = 960,
    height: int = 600,
    trail_steps: int = 24,
    legend_style: str = "white",
):
    """Create synthetic pub-style RGB frames for smoke testing video output."""
    configure_rendering_environment()
    prepare_pub_style()

    okabe_rgbs = high_contrast_rgb01(3)
    legend_style_name = pub_legend_box_style(legend_style)["name"]
    frames = []
    for frame_idx in range(int(frame_count)):
        phase = 2.0 * np.pi * frame_idx / max(int(frame_count) - 1, 1)
        if legend_style_name == "builderbench_gray":
            frame = _builderbench_scene_rgb(width, height, phase).convert("RGBA")
        else:
            frame = _scalar_field_rgb(width, height, phase).convert("RGBA")
        overlay = Image.new("RGBA", frame.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        trace_frame_idx = frame_idx + int(trail_steps)
        trace_frame_count = int(frame_count) + int(trail_steps)
        for trace_idx, rgb in enumerate(okabe_rgbs):
            points = _trajectory_points(trace_frame_idx, trace_idx, trace_frame_count, int(trail_steps))
            pixels = _world_to_pixel(points, width, height)
            color = _rgb01_to_rgb255(rgb)
            if len(pixels) >= 2:
                for seg_idx in range(len(pixels) - 1):
                    alpha = int(75 + 165 * (seg_idx + 1) / max(len(pixels) - 1, 1))
                    draw.line(
                        [tuple(pixels[seg_idx]), tuple(pixels[seg_idx + 1])],
                        fill=color + (alpha,),
                        width=4,
                    )
            cx, cy = pixels[-1]
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=color + (255,), outline=(255, 255, 255, 255))

        phase_rgb = skill_to_rgb01(np.array([np.cos(phase), np.sin(phase)], dtype=np.float32))
        phase_color = _rgb01_to_rgb255(phase_rgb)
        draw.ellipse(
            (width - 58, 24, width - 22, 60),
            fill=phase_color + (245,),
            outline=(255, 255, 255, 255),
            width=2,
        )
        _draw_legend(draw, width, okabe_rgbs, phase_rgb, height=height, legend_style=legend_style)
        draw_pub_metric_badge(
            overlay,
            (
                ("Frame", f"{frame_idx:03d}/{int(frame_count) - 1:03d}"),
                ("Score", format_pub_metric_value(25 * frame_idx, 6)),
            ),
            anchor="bottom_right",
            style="white",
            bold=False,
        )

        frame = Image.alpha_composite(frame, overlay).convert("RGB")
        frames.append(frame)
    return frames


def save_pub_video_render_demo(output_dir: str | Path, *, fps: int = 12):
    """Save a GIF and representative preview PNG into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    white_paths = save_pub_frame_bundle(
        output_dir,
        "pub_video_render",
        create_pub_video_render_frames(legend_style="white"),
        fps=fps,
    )
    gray_paths = save_pub_frame_bundle(
        output_dir,
        "pub_video_render_builderbench_legend",
        create_pub_video_render_frames(legend_style="builderbench_gray"),
        fps=fps,
    )
    return {**white_paths, "white": white_paths, "builderbench_gray": gray_paths}
