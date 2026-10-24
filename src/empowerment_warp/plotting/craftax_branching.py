"""Craftax shared-prefix branching eval media: stills, stats figure, skill-grid video.

Writes ``craftax_branching_stills.{png,pdf}`` (+metadata/plot_data),
``craftax_branching_stats`` (+metadata/plot_data) and ``eval_branching_skill_grid.mp4``
(+preview/manifest) from the rollout dicts built by ``training/pqn/branch_protocol.py``.
Publication helpers (plot_style, achievement_icons, video_render) are bundled under
``empowerment_warp._vendor.plotting``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from empowerment_warp._vendor.plotting.plot_style import add_pub_fixed_axes, pub_rcparams as _pub_rcparams, save_pub_figure
from empowerment_warp._vendor.plotting.achievement_icons import draw_craftax_achievement_icon_band  # noqa: E402
from empowerment_warp._vendor.plotting.plot_style import high_contrast_rgb01, materialize_pub_font  # noqa: E402
from empowerment_warp._vendor.plotting.video_render import PUB_ENV_ANIMATION_FPS, pub_env_timing, save_pub_env_frame_bundle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402



# ---- constants (reference, transitively referenced) ----

CRAFTAX_MEDIA_DPI = 200

_INK = '#181818'

_SKILL_GRID_FPS_SCALE = 0.8

_MIN_ENV_VIDEO_FPS = 24

_BRANCH_GRID_FRAME_REPEAT = 4

_BRANCH_GRID_RENDER_SCALE = 3

_BRANCH_GRID_LABEL_BASE_PX = 14

_BRANCH_GRID_PAD_BASE_PX = 4

_BRANCH_GRID_PHASE_BAND_BASE_PX = 24

_BRANCH_GRID_PHASE_FONT_SCALE = 0.72

_BRANCH_POTENTIAL_LABEL_COLOR = (42, 42, 42)

_STRIKE_SKILL_INDEX = -2

_BRANCH_STRIKE_LABEL_COLOR = (197, 27, 138)

_EP_SKILL_INDEX = -3

_BRANCH_EP_LABEL_COLOR = (0, 128, 128)

_BRANCH_STILL_TILE_IN = 0.85

_BRANCH_STILL_LABEL_IN = 0.13

_BRANCH_STILL_GAP_IN = 0.08

_BRANCH_STILL_MAX_COLS = 4

_BRANCH_GRID_REFERENCE_BLOCK_PX = 7

_CRAFTAX_BLOCK_PX = {'agent': 7, 'img': 16, 'human': 64}



SKILL_GRID_BG = (248, 248, 244)

SKILL_GRID_LABEL_PX = 26

SKILL_GRID_LABEL_FONT_SCALE = 0.82

ROLLOUT_LABEL_BG = (248, 248, 244)

ROLLOUT_LABEL_TEXT = (32, 32, 32)

ROLLOUT_LABEL_BAND_MIN_PX = 26

ROLLOUT_LABEL_BAND_FRACTION = 0.075

ROLLOUT_LABEL_FONT_MIN_PX = 14

ROLLOUT_LABEL_FONT_SCALE = 0.62


# ---- rendering functions ----

def _branch_grid_chrome_scale(config, render_scale):
    """Effective tile scale: 3.0 for agent+3x, 9.14 for human+1x."""
    config = _config_dict(config)
    block_px = _CRAFTAX_BLOCK_PX.get(
        str(config.get('craftax_render_block_pixel_size', 'agent')).lower(),
        _BRANCH_GRID_REFERENCE_BLOCK_PX,
    )
    return (float(block_px) * max(1, int(render_scale))) / float(_BRANCH_GRID_REFERENCE_BLOCK_PX)

def _craftax_media_title(figure_title, default, *, disallow_terms=()):
    text = '' if figure_title is None else str(figure_title).strip()
    if not text:
        return default
    lower_text = text.lower()
    looks_like_run_id = (
        '_' in text
        or '/' in text
        or lower_text.startswith('craftax_')
        or lower_text.startswith('sd')
        or any(str(term).lower() in lower_text for term in disallow_terms)
    )
    return default if looks_like_run_id else text

def _config_dict(config):
    if config is None:
        return {}
    if hasattr(config, 'to_dict'):
        return config.to_dict()
    return dict(config)

def _resolved_skill_grid_fps(config, requested_fps):
    if requested_fps is not None:
        return max(_MIN_ENV_VIDEO_FPS, int(requested_fps))
    config = _config_dict(config)
    configured = int(config.get('eval_skill_grid_video_fps', 0) or config.get('skill_grid_video_fps', 0) or 0)
    if configured > 0:
        return max(_MIN_ENV_VIDEO_FPS, configured)
    return max(_MIN_ENV_VIDEO_FPS, int(round(float(PUB_ENV_ANIMATION_FPS) * _SKILL_GRID_FPS_SCALE)))

def _resolved_branch_grid_frame_repeat(config):
    config = _config_dict(config)
    value = config.get(
        'eval_branching_skill_grid_frame_repeat',
        config.get('branching_skill_grid_frame_repeat', _BRANCH_GRID_FRAME_REPEAT),
    )
    return max(1, int(value))

def _resolved_branch_grid_render_scale(config):
    config = _config_dict(config)
    explicit = config.get(
        'eval_branching_skill_grid_render_scale',
        config.get('branching_skill_grid_render_scale', None),
    )
    if explicit is not None:
        return max(1, int(explicit))
    # The 3x upscale exists only because the `agent` render is 7 px/block -- a 63 px
    # tile is unreadable without it. The `human` render is 64 px/block (576 px tiles),
    # already publication size, so upscaling it again would just triple the file for
    # no detail. Default the scale to the render mode instead of to a constant.
    if str(config.get('craftax_render_block_pixel_size', 'agent')).lower() == 'human':
        return 1
    return max(1, int(_BRANCH_GRID_RENDER_SCALE))

def _resolved_branch_grid_label_px(config, render_scale):
    config = _config_dict(config)
    value = config.get(
        'eval_branching_skill_grid_label_px',
        config.get('branching_skill_grid_label_px', None),
    )
    if value is not None:
        return max(1, int(value))
    chrome = _branch_grid_chrome_scale(config, render_scale)
    return max(26, int(round(_BRANCH_GRID_LABEL_BASE_PX * chrome)))

def _resolved_branch_grid_pad_px(config, render_scale):
    config = _config_dict(config)
    value = config.get(
        'eval_branching_skill_grid_pad_px',
        config.get('branching_skill_grid_pad_px', None),
    )
    if value is not None:
        return max(0, int(value))
    chrome = _branch_grid_chrome_scale(config, render_scale)
    return max(4, int(round(_BRANCH_GRID_PAD_BASE_PX * chrome)))

def _resolved_branch_grid_phase_band_px(config, render_scale, tile_label_px):
    config = _config_dict(config)
    value = config.get(
        'eval_branching_skill_grid_phase_band_px',
        config.get('branching_skill_grid_phase_band_px', None),
    )
    if value is not None:
        return max(1, int(value))
    chrome = _branch_grid_chrome_scale(config, render_scale)
    return max(
        int(tile_label_px) + max(8, int(round(4 * chrome))),
        int(round(_BRANCH_GRID_PHASE_BAND_BASE_PX * chrome)),
    )

def _scale_frame_stack_nearest(frames, scale):
    scale = max(1, int(scale))
    frames = np.asarray(frames, dtype=np.uint8)
    if scale <= 1:
        return frames
    return np.repeat(np.repeat(frames, scale, axis=1), scale, axis=2)

def _branch_prefix_frame_report(branch_rollouts):
    # strike rollouts have no shared potential prefix (fresh z=ones episode), so they
    # are excluded from the shared-prefix consistency check.
    rollouts = [
        rollout for rollout in (branch_rollouts or [])
        if rollout.get('frames') is not None
        and not _is_strike_rollout(rollout)
        and not _is_ep_rollout(rollout)
    ]
    if len(rollouts) <= 1:
        return {'checked': False, 'valid': True, 'max_abs_delta': 0.0, 'prefix_frame_count': 0}

    activation_frame = min(int(rollout.get('activation_frame', 0) or 0) for rollout in rollouts)
    prefix_frame_count = max(1, activation_frame + 1)
    reference = np.asarray(rollouts[0]['frames'], dtype=np.int16)
    max_abs_delta = 0.0
    valid = reference.shape[0] >= prefix_frame_count
    for rollout in rollouts[1:]:
        frames = np.asarray(rollout['frames'], dtype=np.int16)
        if frames.shape[0] < prefix_frame_count or frames.shape[1:] != reference.shape[1:]:
            valid = False
            max_abs_delta = float('inf')
            continue
        delta = np.abs(frames[:prefix_frame_count] - reference[:prefix_frame_count])
        max_abs_delta = max(max_abs_delta, float(np.max(delta)) if delta.size else 0.0)
    valid = bool(valid and max_abs_delta == 0.0)
    return {
        'checked': True,
        'valid': valid,
        'max_abs_delta': max_abs_delta,
        'prefix_frame_count': int(prefix_frame_count),
    }

def _branch_label_text(frame_idx, skill_idx, _skill_vec, activation_frame):
    skill_idx = int(skill_idx)
    if skill_idx == _EP_SKILL_INDEX:
        return 'EP'
    if skill_idx == _STRIKE_SKILL_INDEX:
        return 'Strike'
    if skill_idx < 0 or int(frame_idx) <= int(activation_frame):
        return 'Potential'
    return f'Skill {skill_idx}'

def _branch_label_color(frame_idx, skill_idx, _skill_vec, activation_frame):
    if int(skill_idx) == _EP_SKILL_INDEX:
        return _BRANCH_EP_LABEL_COLOR
    if int(skill_idx) == _STRIKE_SKILL_INDEX:
        return _BRANCH_STRIKE_LABEL_COLOR
    if int(skill_idx) < 0 or int(frame_idx) <= int(activation_frame):
        return _BRANCH_POTENTIAL_LABEL_COLOR
    return skill_index_to_color(skill_idx)

def _skill_count_label(count):
    count = int(count)
    return 'No Skills' if count <= 0 else ('1 Skill' if count == 1 else f'{count} Skills')

def _branch_video_label(frame_idx, flip_after_frame, skill_count):
    del skill_count
    return 'Potential' if int(frame_idx) <= int(flip_after_frame) else 'Branch'

def _figure_to_rgb(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[..., :3].copy()

def _record_saved_paths(bundle, output_dir, name, paths):
    for path in paths:
        path = Path(path)
        artifact_name = name if path.suffix.lower() == '.png' else f'{name}_{path.suffix.lower().lstrip(".")}'
        bundle['saved_paths'][f'{artifact_name}:{output_dir}'] = str(path)

def _save_metadata(output_dir, name, metadata, bundle):
    path = Path(output_dir) / f'{name}_metadata.json'
    path.write_text(json.dumps(_json_ready(metadata), indent=2, sort_keys=True) + '\n')
    bundle['saved_paths'][f'{name}_metadata:{output_dir}'] = str(path)

def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value

def _save_plot_data(output_dir, name, plot_data, bundle):
    path = Path(output_dir) / f'{name}_plot_data.json'
    path.write_text(json.dumps(_json_ready(plot_data), indent=2, sort_keys=True) + '\n')
    bundle['saved_paths'][f'{name}_plot_data:{output_dir}'] = str(path)

def _craftax_render_stack(render):
    """Coerce a per-episode render into a (T,H,W,3) uint8 stack, or None."""
    arr = np.asarray(render)
    if arr.ndim == 4 and arr.shape[0] > 0 and arr.shape[-1] in (3, 4):
        return np.asarray(arr[..., :3], dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):  # single frame -> length-1 stack.
        return np.asarray(arr[None, ..., :3], dtype=np.uint8)
    return None

def _is_ep_rollout(rollout):
    return bool(
        bool(rollout.get('is_ep', False)) or int(rollout.get('skill_index', 0)) == _EP_SKILL_INDEX
    )

def _is_strike_rollout(rollout):
    return bool(rollout) and (
        bool(rollout.get('is_strike', False)) or int(rollout.get('skill_index', 0)) == _STRIKE_SKILL_INDEX
    )

def _is_potential_continuation_rollout(rollout):
    if rollout is None:
        return False
    if _is_strike_rollout(rollout) or _is_ep_rollout(rollout):
        return False  # strike/ep are their own categories (own line + tile)
    return bool(rollout.get('is_potential_continuation', False)) or int(rollout.get('skill_index', 0)) < 0

def _craftax_branch_entries(branch_rollouts):
    entries = []
    for rollout in branch_rollouts or []:
        frames = _craftax_render_stack(rollout.get('frames'))
        if frames is None:
            continue
        entries.append(
            {
                'rollout': rollout,
                'frames': frames,
                'skill_index': int(rollout.get('skill_index', len(entries))),
                'skill': rollout.get('skill'),
                'is_potential_continuation': _is_potential_continuation_rollout(rollout),
                'is_strike': _is_strike_rollout(rollout),
                'is_ep': _is_ep_rollout(rollout),
            }
        )

    potential_entries = [entry for entry in entries if entry['is_potential_continuation']]
    strike_entries = [entry for entry in entries if entry['is_strike']]
    ep_entries = [entry for entry in entries if entry.get('is_ep')]
    skill_entries = [
        entry for entry in entries
        if not entry['is_potential_continuation'] and not entry['is_strike']
        and not entry.get('is_ep')
    ]
    # order: potential first, numbered skills, strike last (its own tile / line).
    ordered_entries = potential_entries[:1] + skill_entries + strike_entries + ep_entries
    return ordered_entries, potential_entries, skill_entries

def _frame_at(frames, frame_idx):
    frame_idx = int(frame_idx)
    if frame_idx < 0:
        frame_idx = int(frames.shape[0]) + frame_idx
    frame_idx = int(np.clip(frame_idx, 0, max(0, int(frames.shape[0]) - 1)))
    return np.asarray(frames[frame_idx], dtype=np.uint8), frame_idx

def _branch_still_tiles(ordered_entries, potential_entries, skill_entries):
    if not ordered_entries:
        return []

    reference = ordered_entries[0]
    activation_frame = int(reference['rollout'].get('activation_frame', 0) or 0)
    start_frame, start_idx = _frame_at(reference['frames'], 0)
    branch_frame, branch_idx = _frame_at(reference['frames'], activation_frame)
    tiles = [
        {
            'label': 'Start',
            'role': 'start',
            'frame': start_frame,
            'frame_index': start_idx,
            'skill_index': None,
            'color': _BRANCH_POTENTIAL_LABEL_COLOR,
        },
        {
            'label': 'Branch Point',
            'role': 'branch_point',
            'frame': branch_frame,
            'frame_index': branch_idx,
            'skill_index': None,
            'color': _BRANCH_POTENTIAL_LABEL_COLOR,
        },
    ]

    if potential_entries:
        potential_frame, potential_idx = _frame_at(potential_entries[0]['frames'], -1)
        tiles.append(
            {
                'label': 'Potential End',
                'role': 'potential_end',
                'frame': potential_frame,
                'frame_index': potential_idx,
                'skill_index': int(potential_entries[0]['skill_index']),
                'color': _BRANCH_POTENTIAL_LABEL_COLOR,
            }
        )

    for entry in skill_entries:
        frame, frame_idx = _frame_at(entry['frames'], -1)
        color = skill_index_to_color(entry['skill_index'])
        tiles.append(
            {
                'label': f'Skill {int(entry["skill_index"])} End',
                'role': 'skill_end',
                'frame': frame,
                'frame_index': frame_idx,
                'skill_index': int(entry['skill_index']),
                'color': color,
            }
        )
    return tiles

def _make_branching_stills_figure(tiles, metadata, *, figure_title=None, max_cols=None):
    tile_count = len(tiles)
    if tile_count <= 0:
        return None

    max_cols = _BRANCH_STILL_MAX_COLS if max_cols is None else max(1, int(max_cols))
    cols = min(max_cols, tile_count)
    rows = (tile_count + cols - 1) // cols
    tile_w = _BRANCH_STILL_TILE_IN
    tile_h = _BRANCH_STILL_TILE_IN
    label_h = _BRANCH_STILL_LABEL_IN
    gap = _BRANCH_STILL_GAP_IN
    margin_l = 0.16
    margin_r = 0.16
    margin_b = 0.10
    margin_t = 0.08
    title_h = 0.20 if figure_title else 0.0
    stats_h = 0.18
    title_gap = 0.04 if figure_title else 0.0
    row_h = tile_h + label_h
    content_w = cols * tile_w + max(0, cols - 1) * gap
    content_h = rows * row_h + max(0, rows - 1) * gap
    fig_w = margin_l + content_w + margin_r
    fig_h = margin_b + content_h + gap + stats_h + title_gap + title_h + margin_t
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=CRAFTAX_MEDIA_DPI, constrained_layout=False)

    y_top = margin_b + content_h
    for tile_idx, tile in enumerate(tiles):
        row = tile_idx // cols
        col = tile_idx % cols
        x = margin_l + col * (tile_w + gap)
        y = margin_b + (rows - 1 - row) * (row_h + gap)
        image_ax = add_pub_fixed_axes(fig, fig_w, fig_h, x, y + label_h, tile_w, tile_h)
        image_ax.imshow(tile['frame'], interpolation='nearest', resample=False)
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        border_color = np.asarray(tile.get('color', _BRANCH_POTENTIAL_LABEL_COLOR), dtype=np.float32)
        if border_color.max(initial=1.0) > 1.0:
            border_color = border_color / 255.0
        for spine in image_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.65)
            spine.set_edgecolor(border_color)

        label_ax = add_pub_fixed_axes(fig, fig_w, fig_h, x, y, tile_w, label_h)
        label_ax.axis('off')
        label_ax.text(
            0.5,
            0.54,
            tile['label'],
            ha='center',
            va='center',
            fontsize=5.0,
            color=border_color,
        )

    stats_text = (
        f"Shared prefix {int(metadata.get('branch_prefix_steps', 0))} steps, "
        f"{_skill_count_label(metadata.get('skill_count', 0))}"
    )
    stats_ax = add_pub_fixed_axes(fig, fig_w, fig_h, margin_l, y_top + gap, content_w, stats_h)
    stats_ax.axis('off')
    stats_ax.text(0.5, 0.50, stats_text, ha='center', va='center', fontsize=6.2, color=_INK)

    if figure_title:
        title_y = y_top + gap + stats_h + title_gap
        title_ax = add_pub_fixed_axes(fig, fig_w, fig_h, margin_l, title_y, content_w, title_h)
        title_ax.axis('off')
        title_ax.text(
            0.5,
            0.50,
            figure_title,
            ha='center',
            va='center',
            fontsize=7.4,
            fontweight='normal',
            color=_INK,
        )

    return fig

def _color01(color):
    arr = np.asarray(color, dtype=np.float32)
    if arr.max(initial=1.0) > 1.0:
        arr = arr / 255.0
    return tuple(float(x) for x in np.clip(arr[:3], 0.0, 1.0))

def _branch_series_label(entry):
    if entry.get('is_ep'):
        return 'EP'
    if entry.get('is_strike'):
        return 'Strike'
    if entry['is_potential_continuation'] or int(entry['skill_index']) < 0:
        return 'Potential'
    return f'Skill {int(entry["skill_index"])}'

def _branch_series_color(entry):
    if entry.get('is_ep'):
        return _color01(_BRANCH_EP_LABEL_COLOR)
    if entry.get('is_strike'):
        return _color01(_BRANCH_STRIKE_LABEL_COLOR)
    if entry['is_potential_continuation'] or int(entry['skill_index']) < 0:
        return _color01(_BRANCH_POTENTIAL_LABEL_COLOR)
    return _color01(skill_index_to_color(entry['skill_index']))

def _branch_stat_series(ordered_entries):
    series = []
    for entry in ordered_entries:
        records = list(entry['rollout'].get('records') or [])
        if not records:
            continue
        steps = []
        achievements = []
        scores = []
        rewards = []
        for record in records:
            steps.append(int(record.get('step', len(steps))))
            achievements.append(float(record.get('cumulative_achievements', 0.0)))
            scores.append(float(record.get('total_achievement_score', record.get('achievement_score', 0.0))))
            rewards.append(float(record.get('total_reward', 0.0)))
        series.append(
            {
                'label': _branch_series_label(entry),
                'color': _branch_series_color(entry),
                'steps': np.asarray(steps, dtype=np.float32),
                'cumulative_achievements': np.asarray(achievements, dtype=np.float32),
                'achievement_score': np.asarray(scores, dtype=np.float32),
                'total_reward': np.asarray(rewards, dtype=np.float32),
                'skill_index': int(entry['skill_index']),
                'is_potential_continuation': bool(entry['is_potential_continuation']),
                'is_strike': bool(entry.get('is_strike', False)),
                'is_ep': bool(entry.get('is_ep', False)),
            }
        )
    return series

def _record_achievement_values(record):
    values = record.get('achievement_values')
    if values is not None:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size:
            names = [str(name) for name in record.get('achievement_names', ())]
            if len(names) < values.size:
                names.extend(f'achievement_{idx}' for idx in range(len(names), int(values.size)))
            return names[: int(values.size)], values

    pairs = []
    for key, value in (record or {}).items():
        if not str(key).startswith('Achievements/'):
            continue
        try:
            scalar = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        pairs.append((str(key).split('/', 1)[1], scalar))
    if pairs:
        names, scalar_values = zip(*sorted(pairs))
        return list(names), np.asarray(scalar_values, dtype=np.float32)
    return [], None

def _branch_achievement_events(ordered_entries):
    events = []
    for entry in ordered_entries:
        row = _branch_series_label(entry)
        records = list(entry['rollout'].get('records') or [])
        previous_values = None
        previous_unlocked = set()
        for record in records:
            step = int(record.get('step', 0))
            names, values = _record_achievement_values(record)
            if values is not None:
                if previous_values is None or previous_values.shape != values.shape:
                    previous_values = np.zeros_like(values)
                newly_unlocked = np.flatnonzero((values > 0.0) & (previous_values <= 0.0))
                for idx in newly_unlocked:
                    events.append({'step': step, 'achievement': names[int(idx)], 'row': row})
                previous_values = values.copy()
                continue

            unlocked = {str(name) for name in record.get('unlocked_achievements', ())}
            for name in sorted(unlocked - previous_unlocked):
                events.append({'step': step, 'achievement': name, 'row': row})
            previous_unlocked |= unlocked
    return events

def _branch_stats_series_plot_data(series):
    return [
        {
            'label': str(item['label']),
            'color': [float(v) for v in item['color']],
            'skill_index': int(item['skill_index']),
            'is_potential_continuation': bool(item['is_potential_continuation']),
            'steps': item['steps'].astype(float).tolist(),
            'cumulative_achievements': item['cumulative_achievements'].astype(float).tolist(),
            'achievement_score': item['achievement_score'].astype(float).tolist(),
            'total_reward': item['total_reward'].astype(float).tolist(),
        }
        for item in series
    ]

def _branch_stats_metadata(series, prefix_report, *, skill_onset_step=None):
    finals = [item for item in series if item['steps'].size > 0]
    final_achievements = [float(item['cumulative_achievements'][-1]) for item in finals]
    final_scores = [float(item['achievement_score'][-1]) for item in finals]
    final_rewards = [float(item['total_reward'][-1]) for item in finals]
    skill_series = [item for item in finals if not item['is_potential_continuation']]
    onset_step = None
    if skill_onset_step is not None:
        onset_value = float(skill_onset_step)
        if np.isfinite(onset_value):
            onset_step = int(onset_value)
    return {
        'series_count': int(len(finals)),
        'skill_series_count': int(len(skill_series)),
        'skill_onset_step': onset_step,
        'metric_family': 'craftax_branching_achievements',
        'step_source': 'branch rollout records[].step',
        'record_fields': {
            'cumulative_achievements': 'records[].cumulative_achievements',
            'achievement_score': 'records[].total_achievement_score with records[].achievement_score fallback',
            'total_reward': 'records[].total_reward',
            'achievement_names': 'records[].achievement_names',
            'achievement_values': 'records[].achievement_values',
            'unlocked_achievements': 'records[].unlocked_achievements',
        },
        'related_logged_metric_keys': [
            'evaluation/num_achievements',
            'exploration/num_achievements',
            'evaluation/achievement_score',
            'exploration/achievement_score',
        ],
        'final_cumulative_achievements_mean': float(np.mean(final_achievements)) if final_achievements else 0.0,
        'final_cumulative_achievements_max': float(np.max(final_achievements)) if final_achievements else 0.0,
        'final_achievement_score_mean': float(np.mean(final_scores)) if final_scores else 0.0,
        'final_achievement_score_max': float(np.max(final_scores)) if final_scores else 0.0,
        'final_total_reward_mean': float(np.mean(final_rewards)) if final_rewards else 0.0,
        'final_total_reward_max': float(np.max(final_rewards)) if final_rewards else 0.0,
        'branch_prefix_invariant': prefix_report,
        'series': [
            {
                'label': str(item['label']),
                'skill_index': int(item['skill_index']),
                'is_potential_continuation': bool(item['is_potential_continuation']),
                'frame_count': int(item['steps'].size),
                'final_step': int(item['steps'][-1]) if item['steps'].size else 0,
                'final_cumulative_achievements': float(item['cumulative_achievements'][-1])
                if item['steps'].size
                else 0.0,
                'final_achievement_score': float(item['achievement_score'][-1]) if item['steps'].size else 0.0,
                'final_total_reward': float(item['total_reward'][-1]) if item['steps'].size else 0.0,
            }
            for item in finals
        ],
    }

def _make_branching_stats_figure(series, *, figure_title=None, skill_onset_step=None, achievement_events=None):
    if not series:
        return None, {}
    onset_step = None
    if skill_onset_step is not None:
        onset_value = float(skill_onset_step)
        if np.isfinite(onset_value):
            onset_step = onset_value

    row_order = [str(item['label']) for item in series]
    row_colors = {str(item['label']): item['color'] for item in series}
    achievement_events = list(achievement_events or [])

    x_values = [item['steps'] for item in series if item['steps'].size]
    if onset_step is not None:
        x_values.append(np.asarray([onset_step], dtype=np.float32))
    x_values = np.concatenate(x_values) if x_values else np.asarray([0.0, 1.0], dtype=np.float32)
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    x_span = max(1.0, x_max - x_min)
    x_min -= max(0.5, 0.02 * x_span)
    x_max += max(0.5, 0.02 * x_span)

    fig_w = 4.85
    margin_l = 0.86
    margin_r = 0.18
    margin_b = 0.52
    margin_t = 0.08
    plot_w = fig_w - margin_l - margin_r
    panel_h = 1.12
    icon_h = max(0.52, min(1.55, 0.17 * max(1, len(row_order)) + 0.18))
    icon_gap = 0.16
    title_h = 0.34 if figure_title else 0.0
    title_gap = 0.025 if figure_title else 0.0

    plot_y = margin_b
    icon_y = plot_y + panel_h + icon_gap
    title_y = icon_y + icon_h + title_gap
    fig_h = title_y + title_h + margin_t
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=CRAFTAX_MEDIA_DPI, constrained_layout=False)

    ax_main = add_pub_fixed_axes(fig, fig_w, fig_h, margin_l, plot_y, plot_w, panel_h)
    ax_icons = add_pub_fixed_axes(fig, fig_w, fig_h, margin_l, icon_y, plot_w, icon_h)
    ax_title = None
    if figure_title:
        ax_title = add_pub_fixed_axes(fig, fig_w, fig_h, margin_l, title_y, plot_w, title_h)

    for item in series:
        label = item['label']
        color = item['color']
        ax_main.plot(
            item['steps'],
            item['cumulative_achievements'],
            color=color,
            linewidth=1.0,
            alpha=0.92,
            label=label,
        )

    if onset_step is not None:
        ax_main.axvline(
            onset_step,
            color=_color01(_BRANCH_POTENTIAL_LABEL_COLOR),
            linestyle='--',
            linewidth=0.85,
            alpha=0.82,
            label='_nolegend_',
            zorder=3.0,
        )
        ax_main.plot(
            [onset_step],
            [1.0],
            transform=ax_main.get_xaxis_transform(),
            marker='o',
            markersize=3.5,
            markerfacecolor='white',
            markeredgecolor=_color01(_BRANCH_POTENTIAL_LABEL_COLOR),
            markeredgewidth=0.75,
            linestyle='None',
            clip_on=False,
            zorder=4.0,
        )

    ax_icons.set_xlim(x_min, x_max)
    icon_metadata = draw_craftax_achievement_icon_band(
        ax_icons,
        achievement_events,
        row_order=row_order,
        row_colors=row_colors,
        branch_step=onset_step,
        source_row='Potential',
        icon_px=22,
        icon_zoom=0.46,
        label_fontsize=plt.rcParams['ytick.labelsize'],
    )

    # font sizes inherit the active pub compact rcParams (see pub_rcparams(compact=True))
    ax_main.set_ylabel('num achievements')
    ax_main.set_xlabel('environment step')
    ax_main.set_xlim(x_min, x_max)
    ax_main.tick_params(axis='both', pad=1.0, length=2.0)
    ax_main.grid(True, alpha=0.22, linewidth=0.55)
    ax_main.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))

    if figure_title:
        ax_title.axis('off')
        # manual text band reserves space so the title never overlaps the icon band;
        # size/weight track the pub figure-title contract rather than a magic number
        ax_title.text(
            0.5,
            0.50,
            figure_title,
            ha='center',
            va='center',
            fontsize=plt.rcParams['figure.titlesize'],
            fontweight=plt.rcParams['figure.titleweight'],
            color=_INK,
        )

    return fig, icon_metadata

def create_craftax_branching_stills_bundle(
    *,
    branch_rollouts=None,
    output_dirs=None,
    filename_prefix='craftax_branching_stills',
    figure_title=None,
    config=None,
):
    """Render compact stills and metadata for a shared-prefix Craftax branch eval."""
    output_dirs = [] if output_dirs is None else [d for d in output_dirs if d]
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

    bundle = {
        'craftax_branching_stills_image': None,
        'craftax_branching_stills_metadata': None,
        'craftax_branching_stats_image': None,
        'craftax_branching_stats_metadata': None,
        'saved_paths': {},
    }
    ordered_entries, potential_entries, skill_entries = _craftax_branch_entries(branch_rollouts)
    if not ordered_entries:
        return bundle

    prefix_report = _branch_prefix_frame_report(branch_rollouts)
    if prefix_report['checked'] and not prefix_report['valid']:
        raise AssertionError(
            'Craftax branching stills prefix frames differ before skill activation.'
        )

    activation_frame = int(ordered_entries[0]['rollout'].get('activation_frame', 0) or 0)
    prefix_steps = [
        int(entry['rollout'].get('branch_prefix_steps', 0))
        for entry in ordered_entries
    ]
    frame_counts = [int(entry['frames'].shape[0]) for entry in ordered_entries]
    tiles = _branch_still_tiles(ordered_entries, potential_entries, skill_entries)
    if not tiles:
        return bundle

    max_cols = int(_config_dict(config).get('craftax_branching_stills_cols', _BRANCH_STILL_MAX_COLS))
    stills_title = _craftax_media_title(
        figure_title,
        'Craftax potential branch stills',
        disallow_terms=('achievements', 'stats'),
    )
    stats_title = _craftax_media_title(
        figure_title,
        'Craftax potential branch achievements',
        disallow_terms=('stills',),
    )
    metadata = {
        'skill_count': int(len(skill_entries)),
        'skill_indices': [int(entry['skill_index']) for entry in skill_entries],
        'panel_count': int(len(tiles)),
        'includes_potential_continuation': bool(potential_entries),
        'branch_prefix_steps': int(prefix_steps[0]) if prefix_steps else 0,
        'branch_activation_frame': int(activation_frame),
        'branch_prefix_invariant': prefix_report,
        'source_frame_count_min': int(min(frame_counts)) if frame_counts else 0,
        'source_frame_count_max': int(max(frame_counts)) if frame_counts else 0,
        'render_kind': 'snapshot',
        'source': (
            'craftax branching stills '
            '(shared potential prefix, potential continuation, then skill branch endpoints)'
        ),
        'shown_stills': [
            {
                'label': str(tile['label']),
                'role': str(tile['role']),
                'frame_index': int(tile['frame_index']),
                'skill_index': None if tile['skill_index'] is None else int(tile['skill_index']),
            }
            for tile in tiles
        ],
    }
    stills_plot_data = {
        'artifact': 'craftax_branching_stills',
        'figure_title': stills_title,
        'metadata': metadata,
        'tiles': [
            {
                'label': str(tile['label']),
                'role': str(tile['role']),
                'frame_index': int(tile['frame_index']),
                'skill_index': None if tile['skill_index'] is None else int(tile['skill_index']),
                'color': _json_ready(tile.get('color', _BRANCH_POTENTIAL_LABEL_COLOR)),
            }
            for tile in tiles
        ],
        'note': 'Tile images come from branch rollout frames; this JSON stores selected frame indices, labels, and colors.',
    }
    metadata['figure_title'] = stills_title
    with plt.rc_context(_pub_rcparams(compact=True)):
        fig = _make_branching_stills_figure(
            tiles,
            metadata,
            figure_title=stills_title,
            max_cols=max_cols,
        )
        if fig is None:
            return bundle

        bundle['craftax_branching_stills_image'] = _figure_to_rgb(fig)
        bundle['craftax_branching_stills_metadata'] = metadata
        for output_dir in output_dirs:
            paths = save_pub_figure(
                fig,
                os.path.join(output_dir, filename_prefix),
                formats=('pdf', 'png'),
                bbox_inches=None,
                dpi=CRAFTAX_MEDIA_DPI,
            )
            _record_saved_paths(bundle, output_dir, filename_prefix, paths)
            _save_metadata(output_dir, filename_prefix, metadata, bundle)
            _save_plot_data(output_dir, filename_prefix, stills_plot_data, bundle)
        plt.close(fig)

        series = _branch_stat_series(ordered_entries)
        skill_onset_step = int(prefix_steps[0]) if prefix_steps else None
        achievement_events = _branch_achievement_events(ordered_entries)
        stats_metadata = (
            _branch_stats_metadata(series, prefix_report, skill_onset_step=skill_onset_step)
            if series
            else None
        )
        if stats_metadata is not None:
            stats_metadata['figure_title'] = stats_title
        stats_fig, icon_metadata = _make_branching_stats_figure(
            series,
            figure_title=stats_title,
            skill_onset_step=skill_onset_step,
            achievement_events=achievement_events,
        )
        if stats_fig is not None and stats_metadata is not None:
            stats_prefix = 'craftax_branching_stats'
            stats_metadata['achievement_icon_timeline'] = {
                'event_count': int(len(achievement_events)),
                **icon_metadata,
            }
            stats_plot_data = {
                'artifact': stats_prefix,
                'figure_title': stats_title,
                'series': _branch_stats_series_plot_data(series),
                'achievement_events': achievement_events,
                'achievement_icon_timeline': stats_metadata['achievement_icon_timeline'],
                'skill_onset_step': skill_onset_step,
                'branch_prefix_invariant': prefix_report,
                'record_fields': stats_metadata.get('record_fields', {}),
                'plotting_helper': 'plotting/achievement_icons.py',
            }
            bundle['craftax_branching_stats_image'] = _figure_to_rgb(stats_fig)
            bundle['craftax_branching_stats_metadata'] = stats_metadata
            for output_dir in output_dirs:
                paths = save_pub_figure(
                    stats_fig,
                    os.path.join(output_dir, stats_prefix),
                    formats=('pdf', 'png'),
                    bbox_inches=None,
                    dpi=CRAFTAX_MEDIA_DPI,
                )
                _record_saved_paths(bundle, output_dir, stats_prefix, paths)
                _save_metadata(output_dir, stats_prefix, stats_metadata, bundle)
                _save_plot_data(output_dir, stats_prefix, stats_plot_data, bundle)
            plt.close(stats_fig)

    return bundle

def create_craftax_branching_skill_grid_video_bundle(
    *,
    branch_rollouts=None,
    output_dirs=None,
    filename_prefix='eval_branching_skill_grid',
    fps=None,
    config=None,
):
    """Tile rollouts that share a potential-policy prefix before splitting by skill."""
    output_dirs = [] if output_dirs is None else [d for d in output_dirs if d]
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

    resolved_fps = _resolved_skill_grid_fps(config, fps)
    bundle = {
        'craftax_branching_skill_grid_video_frames': None,
        'craftax_branching_skill_grid_video_fps': resolved_fps,
        'craftax_branching_skill_grid_video_metadata': None,
        'saved_paths': {},
    }

    potential_rollouts = [
        rollout
        for rollout in branch_rollouts or []
        if _is_potential_continuation_rollout(rollout)
    ]
    skill_branch_rollouts = [
        rollout
        for rollout in branch_rollouts or []
        if not _is_potential_continuation_rollout(rollout)
    ]
    ordered_rollouts = potential_rollouts[:1] + skill_branch_rollouts

    per_skill = []
    for rollout in ordered_rollouts:
        frames = _craftax_render_stack(rollout.get('frames'))
        if frames is None:
            continue
        per_skill.append((
            int(rollout.get('skill_index', len(per_skill))),
            rollout.get('skill'),
            frames,
        ))
    if not per_skill:
        return bundle

    prefix_report = _branch_prefix_frame_report(branch_rollouts)
    if prefix_report['checked'] and not prefix_report['valid']:
        raise AssertionError(
            'Craftax branching skill grid prefix frames differ before skill activation.'
        )

    render_scale = _resolved_branch_grid_render_scale(config)
    tile_label_px = _resolved_branch_grid_label_px(config, render_scale)
    grid_pad_px = _resolved_branch_grid_pad_px(config, render_scale)
    phase_band_px = _resolved_branch_grid_phase_band_px(config, render_scale, tile_label_px)
    per_skill = [
        (skill_idx, skill_vec, _scale_frame_stack_nearest(frames, render_scale))
        for skill_idx, skill_vec, frames in per_skill
    ]
    activation_source = (skill_branch_rollouts or potential_rollouts or [{}])[0]
    activation_frame = int(activation_source.get('activation_frame', 0) or 0)
    grid_frames = make_skill_grid_frames(
        per_skill,
        label_px=tile_label_px,
        pad=grid_pad_px,
        label_text_fn=lambda frame_idx, skill_idx, skill_vec: _branch_label_text(
            frame_idx, skill_idx, skill_vec, activation_frame
        ),
        label_color_fn=lambda frame_idx, skill_idx, skill_vec: _branch_label_color(
            frame_idx, skill_idx, skill_vec, activation_frame
        ),
    )
    if grid_frames is None or grid_frames.shape[0] == 0:
        return bundle

    source_frame_count = int(grid_frames.shape[0])
    frame_repeat = _resolved_branch_grid_frame_repeat(config)
    grid_frames = repeat_frame_stack(grid_frames, frame_repeat)

    min_frames = int(pub_env_timing().get('invalid_swap_total', 15) or 15)
    if grid_frames.shape[0] < min_frames:
        tail = np.repeat(grid_frames[-1:], min_frames - grid_frames.shape[0], axis=0)
        grid_frames = np.concatenate([grid_frames, tail], axis=0)
    flip_after_video_frame = int((activation_frame + 1) * frame_repeat - 1)
    grid_frames = add_video_label_band(
        grid_frames,
        lambda frame_idx: _branch_video_label(frame_idx, flip_after_video_frame, len(skill_branch_rollouts)),
        band_px=phase_band_px,
        font_scale=_BRANCH_GRID_PHASE_FONT_SCALE,
        font_min_px=max(16, int(round(8 * render_scale))),
    )

    prefix_steps = [
        int(rollout.get('branch_prefix_steps', 0))
        for rollout in ordered_rollouts
        if rollout.get('frames') is not None
    ]
    metadata = {
        'skill_count': len(skill_branch_rollouts),
        'skill_indices': [int(rollout.get('skill_index', -1)) for rollout in skill_branch_rollouts],
        'panel_count': len(per_skill),
        'includes_potential_continuation': bool(potential_rollouts),
        'held_frame_count': int(grid_frames.shape[0]),
        'source_frame_count': int(source_frame_count),
        'branch_grid_frame_repeat': int(frame_repeat),
        'branch_grid_layout': {
            'render_scale': int(render_scale),
            'tile_label_px': int(tile_label_px),
            'phase_band_px': int(phase_band_px),
            'pad_px': int(grid_pad_px),
            'phase_font_scale': float(_BRANCH_GRID_PHASE_FONT_SCALE),
            'output_height': int(grid_frames.shape[1]),
            'output_width': int(grid_frames.shape[2]),
        },
        'fps': resolved_fps,
        'render_kind': 'animated',
        'source': 'craftax branching skill grid (shared potential prefix, then skill continuation)',
        'branch_prefix_steps': int(prefix_steps[0]) if prefix_steps else 0,
        'branch_activation_frame': activation_frame,
        'rollout_label': 'Policy: Potential -> Branch, Targets: ' + _skill_count_label(len(skill_branch_rollouts)),
        'branch_label_flip': {
            'potential_label': 'Potential',
            'skill_label_template': 'Skill {skill_index}',
            'legend_policy': 'top phase and per-tile labels only; no gray skill-vector legend band',
            'flip_after_frame': int(activation_frame),
            'flip_after_video_frame': flip_after_video_frame,
        },
        'branch_prefix_invariant': prefix_report,
    }
    bundle['craftax_branching_skill_grid_video_frames'] = grid_frames
    bundle['craftax_branching_skill_grid_video_fps'] = resolved_fps
    bundle['craftax_branching_skill_grid_video_metadata'] = metadata

    frames = list(grid_frames)
    for output_dir in output_dirs:
        paths = save_pub_env_frame_bundle(
            output_dir,
            filename_prefix,
            frames,
            fps=resolved_fps,
            render_kind='animated',
            require_motion=False,
            save_gif=False,
        )
        saved = {key: str(value) for key, value in paths.items() if key != 'contract'}
        manifest = {
            'name': filename_prefix,
            'artifact_role': 'craftax_eval_branching_skill_grid',
            'metadata': metadata,
            'pub_env_contract': dict(paths.get('contract') or {}),
            'outputs': saved,
        }
        manifest_path = Path(output_dir) / f'{filename_prefix}_manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        for key, path in saved.items():
            bundle['saved_paths'][f'{filename_prefix}_{key}:{output_dir}'] = path
        bundle['saved_paths'][f'{filename_prefix}_manifest:{output_dir}'] = str(manifest_path)

    return bundle

def skill_index_to_color(skill_index, *, count=None):
    """Convert a categorical skill index to an Okabe-Ito RGB tuple."""
    skill_index = int(skill_index)
    palette_count = count
    if palette_count is None:
        palette_count = max(1, skill_index + 1)
    palette = high_contrast_rgb01(max(1, int(palette_count)))
    rgb01 = palette[skill_index % len(palette)]
    return tuple(int(round(255 * float(v))) for v in rgb01)

def _load_overlay_font(size_px: float, bold: bool = False):
    font_path = materialize_pub_font(bold=bold)
    return ImageFont.truetype(font_path, size=max(1, int(round(float(size_px)))))

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

def _render_skill_grid_label_layer(per_skill, *, grid_shape, cell_geom, label_px, pad, bg, labels=None, colors=None):
    """Draw per-cell `skill N` captions onto a label layer using the publication contract.

    Uses the shared pub font (`_load_overlay_font`), `draw_pub_text_in_slot` for stable
    centered placement, and Okabe-Ito categorical colors by skill index. Pass explicit
    colors only when a continuous skill-vector color has semantic meaning.
    """
    from empowerment_warp._vendor.plotting.video_render import draw_pub_text_in_slot

    rows, cols = grid_shape
    cell_h, width, height = cell_geom
    grid_h = rows * cell_h + (rows + 1) * pad
    grid_w = cols * width + (cols + 1) * pad
    image = Image.new('RGB', (grid_w, grid_h), tuple(int(v) for v in bg))
    draw = ImageDraw.Draw(image)
    font = _load_overlay_font(max(10, int(round(label_px * SKILL_GRID_LABEL_FONT_SCALE))), bold=False)
    labels = [] if labels is None else list(labels)
    colors = [] if colors is None else list(colors)
    for i, (skill_idx, skill_vec, _frames) in enumerate(per_skill):
        r, c = divmod(i, cols)
        y0 = pad + r * (cell_h + pad)
        x0 = pad + c * (width + pad)
        slot = (x0, y0, x0 + width, y0 + label_px)
        label = str(labels[i]) if i < len(labels) and labels[i] is not None else f'Skill {skill_idx}'
        if i < len(colors) and colors[i] is not None:
            color = tuple(int(v) for v in colors[i])
        else:
            color = skill_index_to_color(skill_idx, count=len(per_skill))
        draw_pub_text_in_slot(
            draw, label, slot, font,
            fill=color, align='center', valign='center',
        )
    return np.asarray(image, dtype=np.uint8)

def make_skill_grid_frames(
    per_skill,
    *,
    label_px=SKILL_GRID_LABEL_PX,
    pad=4,
    bg=SKILL_GRID_BG,
    label_text_fn=None,
    label_color_fn=None,
    show_labels=True,
):
    """Tile per-skill (T,H,W,3) frame stacks into a grid video (T_max,GH,GW,3).

    `per_skill` is a list of (skill_index, skill_vector_or_None, frames). Frames are padded
    to the longest rollout (holding the last frame). Captions are drawn via the pub contract
    unless `show_labels=False`, which is reserved for media that labels each tile elsewhere.
    """
    if not per_skill:
        return None
    height, width = per_skill[0][2].shape[1:3]
    t_max = max(int(frames.shape[0]) for _, _, frames in per_skill)
    padded = []
    for skill_idx, skill_vec, frames in per_skill:
        frames = np.asarray(frames, dtype=np.uint8)
        if frames.shape[0] < t_max:
            tail = np.repeat(frames[-1:], t_max - frames.shape[0], axis=0)
            frames = np.concatenate([frames, tail], axis=0)
        padded.append((skill_idx, skill_vec, frames))

    n = len(padded)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    show_labels = bool(show_labels)
    label_px = max(0, int(label_px)) if show_labels else 0
    cell_h = label_px + height
    grid_h = rows * cell_h + (rows + 1) * pad
    grid_w = cols * width + (cols + 1) * pad

    frames = np.empty((t_max, grid_h, grid_w, 3), dtype=np.uint8)
    frames[:] = np.array(bg, dtype=np.uint8)
    for i, (_idx, _vec, skill_frames) in enumerate(padded):
        r, c = divmod(i, cols)
        y0 = pad + r * (cell_h + pad)
        x0 = pad + c * (width + pad)
        frames[:, y0 + label_px:y0 + label_px + height, x0:x0 + width] = skill_frames

    if not show_labels:
        return frames

    def copy_label_strips(target_frames, label_layer):
        for label_idx in range(n):
            r, c = divmod(label_idx, cols)
            y0 = pad + r * (cell_h + pad)
            x0 = pad + c * (width + pad)
            target_frames[:, y0:y0 + label_px, x0:x0 + width] = label_layer[y0:y0 + label_px, x0:x0 + width]

    if label_text_fn is None and label_color_fn is None:
        # captions are static across time; draw once and copy the label strips into all frames.
        label_layer = _render_skill_grid_label_layer(
            padded,
            grid_shape=(rows, cols),
            cell_geom=(cell_h, width, height),
            label_px=label_px,
            pad=pad,
            bg=bg,
        )
        copy_label_strips(frames, label_layer)
    else:
        for frame_idx in range(t_max):
            labels = [
                None if label_text_fn is None else label_text_fn(frame_idx, skill_idx, skill_vec)
                for skill_idx, skill_vec, _skill_frames in padded
            ]
            colors = [
                None if label_color_fn is None else label_color_fn(frame_idx, skill_idx, skill_vec)
                for skill_idx, skill_vec, _skill_frames in padded
            ]
            label_layer = _render_skill_grid_label_layer(
                padded,
                grid_shape=(rows, cols),
                cell_geom=(cell_h, width, height),
                label_px=label_px,
                pad=pad,
                bg=bg,
                labels=labels,
                colors=colors,
            )
            copy_label_strips(frames[frame_idx:frame_idx + 1], label_layer)
    return frames

def repeat_frame_stack(frames, repeat):
    """Repeat each video frame along time while preserving frame shape."""
    repeat = max(1, int(repeat))
    frames = np.asarray(frames, dtype=np.uint8)
    if repeat <= 1:
        return frames
    return np.repeat(frames, repeat, axis=0)
