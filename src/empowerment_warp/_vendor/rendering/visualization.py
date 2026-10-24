"""Provide lightweight visualization helpers for empowerment rollout renders.

This module maps skills to colors, overlays labels, and builds simple trajectory summary images.
"""

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from empowerment_warp._vendor.plotting.plot_style import materialize_pub_font, skill_to_rgb255

from empowerment_warp._vendor.plotting.plot_style import high_contrast_rgb01

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the active smoke env.
    cv2 = None



SKILL_GRID_BG = (248, 248, 244)

SKILL_GRID_EMPTY_LABEL = (32, 32, 32)

SKILL_GRID_LABEL_PX = 26

SKILL_GRID_LABEL_FONT_SCALE = 0.82

ROLLOUT_LABEL_BG = (248, 248, 244)

ROLLOUT_LABEL_TEXT = (32, 32, 32)

ROLLOUT_LABEL_BAND_MIN_PX = 26

ROLLOUT_LABEL_BAND_FRACTION = 0.075

ROLLOUT_LABEL_FONT_MIN_PX = 14

ROLLOUT_LABEL_FONT_SCALE = 0.62

def _load_overlay_font(size_px: float, bold: bool = False):
    font_path = materialize_pub_font(bold=bold)
    return ImageFont.truetype(font_path, size=max(1, int(round(float(size_px)))))

def _format_rollout_skill_label(skill=None, skill_index=None, max_values=3):
    if skill_index is not None:
        return f'Skill {int(skill_index)}'
    skill_arr = np.asarray(() if skill is None else skill, dtype=np.float32).reshape(-1)
    if skill_arr.size == 0 or np.linalg.norm(skill_arr) <= 1e-6:
        return 'Skill off'
    shown = [f'{float(x):+.2f}' for x in skill_arr[:max_values]]
    if skill_arr.size > max_values:
        shown.append('...')
    return 'Skill [' + ', '.join(shown) + ']'

def _rollout_policy_label(frame_idx, trace, *, frame_stride=1, has_reset_frame=False):
    trace = {} if trace is None else dict(trace)
    mode = str(trace.get('mode', '') or '')
    if mode == 'pe_only':
        return 'Potential'
    if mode == 'skill_only':
        return 'Skill'

    activation_index = trace.get('activation_index', trace.get('activation_record', None))
    if activation_index is None:
        skill = np.asarray(trace.get('skill', ()), dtype=np.float32).reshape(-1)
        return 'Skill' if skill.size > 0 and np.linalg.norm(skill) > 1e-6 else 'Potential'

    raw_step = int(frame_idx) if has_reset_frame else (int(frame_idx) + 1) * max(1, int(frame_stride))
    return 'Potential' if raw_step <= int(activation_index) else 'Skill'

def _rollout_label_text(frame_idx, trace, *, skill_index=None, frame_stride=1, has_reset_frame=False):
    trace = {} if trace is None else dict(trace)
    skill_label = _format_rollout_skill_label(
        trace.get('skill', None),
        skill_index=trace.get('skill_index', skill_index),
    )
    policy_label = _rollout_policy_label(
        frame_idx,
        trace,
        frame_stride=frame_stride,
        has_reset_frame=has_reset_frame,
    )
    return f'Policy: {policy_label} | Target: {skill_label}'

def add_video_label_band(
    frames,
    label_fn,
    *,
    band_px=None,
    bg=ROLLOUT_LABEL_BG,
    text_rgb=ROLLOUT_LABEL_TEXT,
    font_scale=None,
    font_min_px=None,
    bold=False,
):
    """Add a fixed top label band to every video frame."""
    from empowerment_warp._vendor.plotting.video_render import draw_pub_text_in_slot

    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[0] == 0:
        return frames
    frame_count, height, width = frames.shape[:3]
    band_px = (
        max(ROLLOUT_LABEL_BAND_MIN_PX, int(round(ROLLOUT_LABEL_BAND_FRACTION * min(height, width))))
        if band_px is None
        else max(1, int(band_px))
    )
    font_scale = ROLLOUT_LABEL_FONT_SCALE if font_scale is None else float(font_scale)
    font_min_px = ROLLOUT_LABEL_FONT_MIN_PX if font_min_px is None else int(font_min_px)
    font = _load_overlay_font(max(font_min_px, int(round(font_scale * band_px))), bold=bold)
    out = np.empty((frame_count, height + band_px, width, 3), dtype=np.uint8)
    out[:, :band_px, :, :] = np.asarray(bg, dtype=np.uint8)
    out[:, band_px:, :, :] = frames[..., :3]
    for frame_idx in range(frame_count):
        image = Image.fromarray(out[frame_idx], mode='RGB')
        draw = ImageDraw.Draw(image)
        draw_pub_text_in_slot(
            draw,
            str(label_fn(frame_idx)),
            (8, 0, width - 8, band_px),
            font,
            fill=tuple(int(v) for v in text_rgb),
            align='center',
            valign='center',
        )
        out[frame_idx] = np.asarray(image, dtype=np.uint8)
    return out

def add_rollout_label_band(
    frames,
    *,
    trace=None,
    skill_index=None,
    frame_stride=1,
    has_reset_frame=False,
):
    """Label a rollout video with the active policy and target skill."""
    return add_video_label_band(
        frames,
        lambda frame_idx: _rollout_label_text(
            frame_idx,
            trace,
            skill_index=skill_index,
            frame_stride=frame_stride,
            has_reset_frame=has_reset_frame,
        ),
    )
