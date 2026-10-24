"""Publication-contract fields and rollout media for Latent-GCE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from empowerment_warp._vendor.plotting.plot_style import configure_rendering_environment

configure_rendering_environment()

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from empowerment_warp._vendor.plotting.contracts.media import save_pub_env_frame_bundle  # noqa: E402
from empowerment_warp._vendor.plotting.plot_style import (  # noqa: E402
    DEFAULT_SCALAR_CMAP,
    add_pub_fixed_axes,
    apply_pub_style,
    assert_pub_axes_same_size_inches,
    assert_pub_axis_tick_labels_clear,
    high_contrast_rgb01,
    pub_uniform_typography_for_width,
    assert_pub_type_scale,
    assert_pub_uniform_text_size,
    pub_suptitle,
    save_pub_figure,
)
from empowerment_warp._vendor.plotting.video_render import draw_pub_text_in_slot  # noqa: E402

from empowerment_warp.baselines.toy_baselines.config import ExperimentConfig
from empowerment_warp.baselines.toy_baselines.evaluation import EpisodeTrace
from empowerment_warp.baselines.toy_baselines.fonts import _load_overlay_font

def _visitation_histogram(
    config: ExperimentConfig,
    observations: np.ndarray,
) -> np.ndarray:
    bins = config.evaluation.grid_bins
    if config.environment.name == 'ball_in_box':
        x_values, y_values = observations[:, 0], observations[:, 1]
        x_range, y_range = (-9.0, 9.0), (-9.0, 9.0)
    else:
        x_values = np.arctan2(observations[:, 1], observations[:, 0])
        y_values = observations[:, 2]
        x_range, y_range = (-np.pi, np.pi), (-8.0, 8.0)
    histogram, _, _ = np.histogram2d(
        y_values,
        x_values,
        bins=bins,
        range=(y_range, x_range),
    )
    return histogram


def save_field_figure(config, fields, output_dir, env_steps):
    """Raw estimates and, for controllers, separate policy/visitation panels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bins = config.evaluation.grid_bins
    x, y = np.asarray(fields['x']), np.asarray(fields['y'])
    estimate = np.asarray(fields['estimate']).reshape((bins, bins))
    unit = 'Capacity (bits)' if config.method == 'gce' else 'Empowerment (nats)'
    panels = [('estimate', estimate, unit)]
    if 'policy_actions' in fields:
        actions = np.asarray(fields['policy_actions']).reshape((bins, bins, config.environment.action_dim))
        ball = config.environment.name == 'ball_in_box'
        panels += [('policy', np.linalg.norm(actions, axis=-1) if ball else actions[..., 0],
                    'Action magnitude' if ball else 'Torque'),
                   ('visitation', _visitation_histogram(config, np.asarray(fields['visitation_observations'])),
                    'Controller visit count')]
    saved_paths = {}
    apply_pub_style(plt, compact=True, use_tex=False)
    width, height = 6.0, 5.8
    font = pub_uniform_typography_for_width(width).axis_label.fontsize
    for name, values, label in panels:
        fig = plt.figure(figsize=(width, height))
        ax = add_pub_fixed_axes(fig, width, height, 0.85, 0.8, 3.8, 3.8)
        cax = add_pub_fixed_axes(fig, width, height, 4.82, 0.8, 0.15, 3.8)
        image = ax.imshow(values, origin='lower', extent=(x[0], x[-1], y[0], y[-1]),
                          cmap=DEFAULT_SCALAR_CMAP, interpolation='bilinear' if name != 'visitation' else 'nearest',
                          aspect='equal' if config.environment.name == 'ball_in_box' else 'auto')
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.ax.tick_params(labelsize=font)
        if name == 'policy' and config.environment.name == 'ball_in_box':
            xx, yy = np.meshgrid(x, y)
            ax.quiver(xx[::4, ::4], yy[::4, ::4], actions[::4, ::4, 0], actions[::4, ::4, 1],
                      color='white', scale=18., width=0.005)
        x_label, y_label = (('$x$', '$y$') if config.environment.name == 'ball_in_box'
                            else (r'$\theta$', r'$\dot{\theta}$'))
        ax.set_xlabel(x_label, fontsize=font)
        ax.set_ylabel(y_label, fontsize=font)
        ax.tick_params(labelsize=font)
        ax.set_title(label, fontsize=font, pad=12)
        title = pub_suptitle(fig, f'{config.method}: {env_steps:,} training steps', has_subtitles=True)
        title.set_fontsize(font)
        for axis in (ax, cax):
            axis.xaxis.get_offset_text().set_fontsize(font)
            axis.yaxis.get_offset_text().set_fontsize(font)
        fig.canvas.draw()
        assert_pub_axis_tick_labels_clear(fig, ax)
        assert_pub_type_scale(fig, font)
        assert_pub_uniform_text_size(fig, font)
        stem = output_dir / f'{name}_step_{env_steps:09d}'
        paths = save_pub_figure(fig, stem, formats=('pdf', 'png'), close=True, dpi=180)
        saved_paths.update({f'{name}_{path.suffix[1:]}': path for path in paths})
        data = dict(method=config.method, environment=config.environment.name, env_steps=env_steps,
                    source_checkpoint=str(Path(config.output_dir) / 'checkpoints' / f'step_{env_steps:09d}.msgpack'),
                    metric=label, x=x.tolist(), y=y.tolist(), values=values.tolist(),
                    colormap=DEFAULT_SCALAR_CMAP, normalization='none', grid_bins=bins)
        if name == 'policy':
            data['actions'] = actions.tolist()
        data_path = stem.with_name(stem.name + '_plot_data.json')
        data_path.write_text(json.dumps(data, indent=2) + '\n')
        saved_paths[f'{name}_plot_data'] = data_path
    return saved_paths


def _render_ball_state(physical: np.ndarray, title: str, frame_step: int) -> Image.Image:
    size, header = 480, 52
    image = Image.new('RGB', (size, size + header), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    font = _load_overlay_font(22, bold=False)
    draw_pub_text_in_slot(draw, title, (14, 0, size - 14, header), font, fill=(32, 32, 32))
    inset = 34
    draw.rectangle((inset, header + inset, size - inset, header + size - inset), outline=(30, 30, 30), width=4)
    scale = (size - 2 * inset) / 20.0
    x = inset + (float(physical[0]) + 10.0) * scale
    y = header + size - inset - (float(physical[1]) + 10.0) * scale
    radius = scale
    color = tuple(int(round(255 * value)) for value in high_contrast_rgb01(1)[0])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(20, 20, 20), width=2)
    step_font = _load_overlay_font(15, bold=False)
    draw_pub_text_in_slot(
        draw,
        f'env step {frame_step:03d}',
        (size - 135, header + 5, size - 12, header + 31),
        step_font,
        fill=(32, 32, 32),
        align='right',
    )
    return image


def _render_pendulum_state(physical: np.ndarray, title: str, frame_step: int) -> Image.Image:
    size, header = 480, 52
    image = Image.new('RGB', (size, size + header), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    font = _load_overlay_font(22, bold=False)
    draw_pub_text_in_slot(draw, title, (14, 0, size - 14, header), font, fill=(32, 32, 32))
    center = np.array([size / 2.0, header + size / 2.0])
    theta = float(physical[0])
    length = 155.0
    endpoint = center + length * np.array([np.sin(theta), -np.cos(theta)])
    color = tuple(int(round(255 * value)) for value in high_contrast_rgb01(2)[1])
    draw.line((tuple(center), tuple(endpoint)), fill=color, width=22)
    draw.ellipse(
        (endpoint[0] - 18, endpoint[1] - 18, endpoint[0] + 18, endpoint[1] + 18),
        fill=color,
        outline=(20, 20, 20),
        width=2,
    )
    draw.ellipse(
        (center[0] - 8, center[1] - 8, center[0] + 8, center[1] + 8),
        fill=(20, 20, 20),
    )
    step_font = _load_overlay_font(15, bold=False)
    draw_pub_text_in_slot(
        draw,
        f'env step {frame_step:03d}',
        (size - 135, header + 5, size - 12, header + 31),
        step_font,
        fill=(32, 32, 32),
        align='right',
    )
    return image


def _interpolate_physical(env_name: str, start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    if env_name == 'ball_in_box':
        return (1.0 - alpha) * start + alpha * end
    delta_theta = (float(end[0] - start[0]) + np.pi) % (2.0 * np.pi) - np.pi
    return np.array(
        [float(start[0]) + alpha * delta_theta, (1.0 - alpha) * float(start[1]) + alpha * float(end[1])]
    )


def save_rollout_media(
    config: ExperimentConfig,
    trace: EpisodeTrace,
    output_dir: str | Path,
    env_steps: int,
    *,
    method_label: str = 'Latent-GCE',
    mode: str = 'deterministic',
) -> dict[str, Path]:
    """Render one fixed-start deterministic rollout with interpolated motion."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    physical_states = np.asarray(trace.physical_states)
    frames = []
    title = f'{method_label} policy at {env_steps:,} training steps'
    render = _render_ball_state if config.environment.name == 'ball_in_box' else _render_pendulum_state
    for step_index, (start, end) in enumerate(zip(physical_states[:-1], physical_states[1:])):
        for alpha in (0.0, 0.5):
            frames.append(render(_interpolate_physical(config.environment.name, start, end, alpha), title, step_index))
    frames.extend(render(physical_states[-1], title, config.environment.episode_steps) for _ in range(10))
    stem = f'{mode}_rollout_step_{env_steps:09d}'
    frame_deltas = [
        float(np.mean(np.abs(np.asarray(current, dtype=np.int16) - np.asarray(previous, dtype=np.int16))))
        for previous, current in zip(frames[:-1], frames[1:])
    ]
    max_rendered_delta = max(frame_deltas, default=0.0)
    render_kind = 'animated' if max_rendered_delta >= 0.5 else 'snapshot'
    paths = save_pub_env_frame_bundle(
        output_dir,
        stem,
        frames,
        fps=30,
        render_kind=render_kind,
        require_motion=render_kind == 'animated',
        save_gif=True,
    )
    start_path = output_dir / f'{stem}_start.png'
    end_path = output_dir / f'{stem}_end.png'
    frames[0].save(start_path)
    frames[-1].save(end_path)
    manifest = {
        'environment': config.environment.name,
        'seed': config.seed,
        'env_steps': int(env_steps),
        'evaluation_policy': mode,
        'source_checkpoint': str(Path(config.output_dir) / 'checkpoints' / f'step_{env_steps:09d}.msgpack'),
        'reset_bank_seed': 91_001 if mode == 'deterministic' else 92_001,
        'interpolation_frames_per_transition': 2,
        'source_episode_steps': config.environment.episode_steps,
        'render_kind': render_kind,
        'max_rendered_frame_delta': max_rendered_delta,
        'pub_env_contract': paths['contract'],
        'outputs': {
            key: str(value)
            for key, value in paths.items()
            if key != 'contract'
        }
        | {'start': str(start_path), 'end': str(end_path)},
    }
    manifest_path = output_dir / f'{stem}_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    result = {key: Path(value) for key, value in paths.items() if key != 'contract'}
    result.update({'start': start_path, 'end': end_path, 'manifest': manifest_path})
    return result
