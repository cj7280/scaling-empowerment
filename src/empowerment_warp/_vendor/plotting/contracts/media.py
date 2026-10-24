"""Public media and animation contracts for env/game render outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PUB_ENV_ANIMATION_FPS = 30
PUB_ENV_PHASE_TIMING = {
    "hold_before": 4,
    "swap": 8,
    "invalid_return": 7,
    "stripe_sweep": 16,
    "pop": 8,
    "clear": 8,
    "gravity": 10,
    "hold_after": 5,
    "creation_gather": 12,
    "creation_pop": 4,
    "creation_gravity": 9,
    "creation_hold_after": 6,
}


def _coerce_rgb_image(frame) -> Image.Image:
    return Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")


def pub_env_timing() -> dict[str, int]:
    """Return the shared frame budgets for env/game graphics clips."""

    timing = dict(PUB_ENV_PHASE_TIMING)
    timing["invalid_swap_total"] = timing["swap"] + timing["invalid_return"]
    timing["clear_total"] = timing["hold_before"] + timing["clear"] + timing["gravity"] + timing["hold_after"]
    timing["creation_total"] = (
        timing["hold_before"]
        + timing["creation_gather"]
        + timing["creation_pop"]
        + timing["creation_gravity"]
        + timing["creation_hold_after"]
    )
    return timing


@dataclass(frozen=True)
class PubEnvVideoSpec:
    """Object contract for environment/game video bundles."""

    fps: int | None = PUB_ENV_ANIMATION_FPS
    phase_timing: dict[str, int] | None = None
    render_kind: str = "animated"
    min_motion_frames: int | None = None
    require_motion: bool | None = None
    require_nonblank: bool = True
    min_frame_delta: float = 0.5
    min_pixel_std: float = 1.0

    def __post_init__(self) -> None:
        fps = PUB_ENV_ANIMATION_FPS if self.fps is None else int(self.fps)
        object.__setattr__(self, "fps", fps)

    def resolved_timing(self) -> dict[str, int]:
        timing = pub_env_timing()
        if self.phase_timing is not None:
            timing.update({str(key): int(value) for key, value in dict(self.phase_timing).items()})
        return timing

    def validate_frames(self, frames) -> dict[str, Any]:
        frames = list(frames)
        if not frames:
            raise AssertionError("Env video bundle has no frames.")
        fps = int(self.fps)
        if fps < 24:
            raise AssertionError(f"Env video fps must be at least 24; got {fps}.")

        first = _coerce_rgb_image(frames[0])
        width, height = first.size
        frame_arrays = []
        for idx, frame in enumerate(frames):
            image = _coerce_rgb_image(frame)
            if image.size != (width, height):
                raise AssertionError(
                    f"Env video frame size changed at frame {idx}: "
                    f"{image.size} != {(width, height)}."
                )
            frame_arrays.append(np.asarray(image, dtype=np.uint8))

        timing = self.resolved_timing()
        render_kind = str(self.render_kind)
        if render_kind == "animated":
            min_frames = int(self.min_motion_frames if self.min_motion_frames is not None else timing["invalid_swap_total"])
            if len(frames) < min_frames:
                raise AssertionError(
                    f"Animated env video has {len(frames)} frames, below the minimum {min_frames}. "
                    "Use shared timing constants or mark the bundle as a held snapshot."
                )
        elif render_kind == "snapshot":
            if len(frames) < int(timing["hold_before"]):
                raise AssertionError(
                    f"Snapshot env video has only {len(frames)} frame(s); repeat snapshots with "
                    "pub_env_snapshot_frames so the viewer can read them."
                )
        else:
            raise ValueError(f"Unsupported env video render_kind: {render_kind!r}")

        frame_stds = [float(np.std(frame)) for frame in frame_arrays]
        max_pixel_std = max(frame_stds, default=0.0)
        if bool(self.require_nonblank) and max_pixel_std < float(self.min_pixel_std):
            raise AssertionError(
                f"Env video frames look blank/flat: max pixel std {max_pixel_std:.3f} "
                f"is below {float(self.min_pixel_std):.3f}."
            )

        frame_deltas = [
            float(np.mean(np.abs(curr.astype(np.int16) - prev.astype(np.int16))))
            for prev, curr in zip(frame_arrays, frame_arrays[1:])
        ]
        max_frame_delta = max(frame_deltas, default=0.0)
        mean_frame_delta = float(np.mean(frame_deltas)) if frame_deltas else 0.0
        changed_frame_count = int(sum(delta >= float(self.min_frame_delta) for delta in frame_deltas))
        require_motion = render_kind == "animated" if self.require_motion is None else bool(self.require_motion)
        if require_motion and changed_frame_count <= 0:
            raise AssertionError(
                f"Animated env video has no frame-to-frame motion above delta {float(self.min_frame_delta):.3f}; "
                "mark it as a held snapshot or render the action/transition phases."
            )
        return {
            "fps": fps,
            "frame_count": len(frames),
            "width": int(width),
            "height": int(height),
            "render_kind": render_kind,
            "phase_timing": timing,
            "max_pixel_std": max_pixel_std,
            "mean_frame_delta": mean_frame_delta,
            "max_frame_delta": max_frame_delta,
            "changed_frame_count": changed_frame_count,
        }

    def snapshot_frames(self, frames, *, hold_frames: int | None = None):
        hold_frames = int(hold_frames if hold_frames is not None else self.resolved_timing()["clear_total"])
        expanded = []
        for frame in frames:
            image = _coerce_rgb_image(frame)
            expanded.extend(image.copy() for _ in range(max(1, hold_frames)))
        return expanded


@dataclass(frozen=True)
class PubEnvFrameBundle:
    """Common saver for env/game MP4, preview, contract, and provenance."""

    output_dir: str | Path
    stem: str
    spec: PubEnvVideoSpec = field(default_factory=PubEnvVideoSpec)
    primitive_sources: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    save_gif: bool = True

    def save(self, frames):
        frames = list(frames)
        contract = self.spec.validate_frames(frames)
        paths = save_pub_frame_bundle(
            self.output_dir,
            self.stem,
            frames,
            fps=int(self.spec.fps),
            save_gif=bool(self.save_gif),
        )
        paths["contract"] = contract
        return paths

    def save_base_animation(self, frames):
        output_dir = Path(self.output_dir)
        paths = self.save(frames)
        manifest = {
            "name": str(self.stem),
            "artifact_role": "base_env_animation",
            "contract": paths["contract"],
            "primitive_sources": [str(source) for source in self.primitive_sources],
            "dependencies": [str(dependency) for dependency in self.dependencies],
            "outputs": {key: str(value) for key, value in paths.items() if key != "contract"},
        }
        manifest_path = output_dir / f"{self.stem}_base_animation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths["base_manifest"] = manifest_path
        return paths


@dataclass(frozen=True)
class PubMediaTilingSpec:
    """Fixed-geometry contract for rendered-frame contact sheets."""

    tile_width_px: int = 256
    max_columns: int = 4
    gap_px: int = 12
    pad_px: int = 16
    label_height_px: int = 28
    label_font_px: int = 13
    background_rgb: tuple[int, int, int] = (255, 255, 255)
    border_rgb: tuple[int, int, int] = (208, 208, 208)
    label_rgb: tuple[int, int, int] = (20, 20, 20)

    def __post_init__(self) -> None:
        if int(self.tile_width_px) < 16:
            raise ValueError(f"tile_width_px must be at least 16; got {self.tile_width_px}.")
        if int(self.max_columns) < 1:
            raise ValueError(f"max_columns must be positive; got {self.max_columns}.")
        if int(self.gap_px) < 0 or int(self.pad_px) < 0 or int(self.label_height_px) < 0:
            raise ValueError("Tiling gaps, padding, and label height must be nonnegative.")
        if int(self.label_font_px) < 1:
            raise ValueError(f"label_font_px must be positive; got {self.label_font_px}.")

    def grid_shape(self, count: int) -> tuple[int, int]:
        count = int(count)
        if count <= 0:
            raise ValueError("A media contact sheet needs at least one frame.")
        columns = min(int(self.max_columns), count)
        rows = int(np.ceil(count / columns))
        return rows, columns


def assert_pub_env_video_contract(
    frames,
    *,
    fps: int,
    phase_timing: dict | None = None,
    render_kind: str = "animated",
    min_motion_frames: int | None = None,
    require_motion: bool | None = None,
    require_nonblank: bool = True,
    min_frame_delta: float = 0.5,
    min_pixel_std: float = 1.0,
):
    """Validate basic publication/env-graphics video invariants."""

    return PubEnvVideoSpec(
        fps=int(fps),
        phase_timing=phase_timing,
        render_kind=render_kind,
        min_motion_frames=min_motion_frames,
        require_motion=require_motion,
        require_nonblank=require_nonblank,
        min_frame_delta=min_frame_delta,
        min_pixel_std=min_pixel_std,
    ).validate_frames(frames)


def assert_pub_env_symmetry_contract(mechanic_frame_counts: dict, symmetric_pairs):
    """Assert paired env mechanics have identical frame counts."""

    counts = {str(key): int(value) for key, value in dict(mechanic_frame_counts).items()}
    checks = []
    for pair in symmetric_pairs:
        if len(pair) == 2:
            name = f"{pair[0]}=={pair[1]}"
            left, right = pair
        elif len(pair) == 3:
            name, left, right = pair
        else:
            raise ValueError(f"Symmetry pair must have 2 or 3 fields, got {pair!r}")
        left = str(left)
        right = str(right)
        if left not in counts or right not in counts:
            raise AssertionError(f"Missing symmetry frame count for {left!r} or {right!r}: {counts}")
        if counts[left] != counts[right]:
            raise AssertionError(
                f"Env symmetry {name} failed: {left}={counts[left]} frames, "
                f"{right}={counts[right]} frames"
            )
        checks.append({"name": str(name), "left": left, "right": right, "frame_count": counts[left]})
    return checks


def pub_env_snapshot_frames(frames, *, hold_frames: int | None = None):
    """Repeat env snapshot frames enough to make still-state videos readable."""

    return PubEnvVideoSpec().snapshot_frames(frames, hold_frames=hold_frames)


def pub_env_ease(progress, *, easing: str = "smoothstep"):
    """Apply a named easing curve to scalar or array-like progress in [0, 1]."""

    values = np.clip(np.asarray(progress, dtype=np.float32), 0.0, 1.0)
    easing = str(easing)
    if easing == "linear":
        eased = values
    elif easing == "smoothstep":
        eased = values * values * (3.0 - 2.0 * values)
    elif easing == "sin_out":
        eased = np.sin(0.5 * np.pi * values)
    elif easing == "sin_in":
        eased = 1.0 - np.cos(0.5 * np.pi * values)
    else:
        raise ValueError(f"Unsupported env easing curve: {easing!r}")
    if np.isscalar(progress):
        return float(np.asarray(eased))
    return eased


def pub_env_progress_values(frame_count: int, *, easing: str = "smoothstep", endpoint: bool = True):
    """Return eased progress samples for a reusable animation phase."""

    frame_count = max(1, int(frame_count))
    if frame_count == 1:
        raw = np.asarray([1.0], dtype=np.float32)
    else:
        raw = np.linspace(0.0, 1.0, frame_count, endpoint=bool(endpoint), dtype=np.float32)
    return pub_env_ease(raw, easing=easing)


def pub_env_lerp(start, end, progress):
    """Linearly interpolate coordinates or numeric arrays at progress in [0, 1]."""

    start_arr = np.asarray(start, dtype=np.float32)
    end_arr = np.asarray(end, dtype=np.float32)
    value = start_arr + (end_arr - start_arr) * float(np.clip(progress, 0.0, 1.0))
    return tuple(float(x) for x in np.ravel(value)) if value.ndim > 0 else float(value)


def pub_env_motion_path(start, end, frame_count: int, *, easing: str = "smoothstep"):
    """Sample a coordinate path once and reuse it for equivalent motions."""

    return tuple(pub_env_lerp(start, end, progress) for progress in pub_env_progress_values(frame_count, easing=easing))


def pub_env_phase_at(frame_idx: int, phase_timing: dict, *, easing: str = "smoothstep"):
    """Return ``(phase_name, raw_progress, eased_progress)`` for a frame index."""

    cursor = 0
    for name, count in dict(phase_timing).items():
        count = max(1, int(count))
        end = cursor + count
        if int(frame_idx) < end:
            raw = (int(frame_idx) - cursor) / max(count - 1, 1)
            return str(name), float(raw), float(pub_env_ease(raw, easing=easing))
        cursor = end
    return str(next(reversed(dict(phase_timing)), "hold")), 1.0, 1.0


def _mp4_safe_frames(frames):
    """Return RGB frames padded to dimensions accepted by yuv420p encoders."""

    first = np.asarray(frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    safe_width = int(width) + int(width % 2)
    safe_height = int(height) + int(height % 2)
    if safe_width == int(width) and safe_height == int(height):
        return [_coerce_rgb_image(frame) for frame in frames]

    safe_frames = []
    for frame in frames:
        image = _coerce_rgb_image(frame)
        padded = Image.new("RGB", (safe_width, safe_height), (248, 248, 244))
        padded.paste(image, (0, 0))
        safe_frames.append(padded)
    return safe_frames


def _save_mp4(path: Path, frames, *, fps: int):
    path = Path(path)
    path.unlink(missing_ok=True)
    frames = _mp4_safe_frames(frames)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        first = np.asarray(frames[0], dtype=np.uint8)
        height, width = first.shape[:2]
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{int(width)}x{int(height)}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(max(int(fps), 1)),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for frame in frames:
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with code {return_code}: {stderr[-1200:]}")
        return path

    try:
        import imageio.v2 as imageio

        with imageio.get_writer(
            str(path),
            format="FFMPEG",
            fps=max(int(fps), 1),
            codec="libx264",
            quality=8,
            macro_block_size=1,
        ) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
        return path
    except Exception:
        pass

    try:
        import cv2

        first = np.asarray(frames[0], dtype=np.uint8)
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(int(fps), 1),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open MP4 writer for {path}")
        for frame in frames:
            rgb = np.asarray(frame, dtype=np.uint8)
            writer.write(rgb[..., ::-1])
        writer.release()
        return path
    except Exception as exc:
        raise RuntimeError(f"Could not write MP4 video to {path}") from exc


def _pub_media_label_font(font_px: int):
    from empowerment_warp._vendor.plotting.plot_style import materialize_pub_font, prepare_pub_style

    prepare_pub_style()
    font_path = materialize_pub_font(bold=False)
    return ImageFont.truetype(font_path, size=int(font_px)), {
        "font_path": font_path,
        "font_px": int(font_px),
        "font_source": "plot_style.materialize_pub_font",
    }


def _pub_media_ellipsize(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    text = str(text)
    max_width = int(max_width)
    if max_width <= 0:
        return ""
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    suffix = "..."
    suffix_width = draw.textbbox((0, 0), suffix, font=font)[2]
    if suffix_width > max_width:
        return ""
    lo, hi = 0, len(text)
    best = suffix
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + suffix
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def make_pub_frame_contact_sheet(
    frames,
    *,
    labels=None,
    frame_indices=None,
    spec: PubMediaTilingSpec | None = None,
):
    """Tile rendered frames into a compact, fixed-geometry contact sheet.

    The tile size, label band, gutters, and page padding are explicit pixel
    quantities so a growing number of stills wraps into rows instead of making
    a too-wide strip.
    """

    frames = [_coerce_rgb_image(frame) for frame in frames]
    if not frames:
        raise ValueError("A media contact sheet needs at least one frame.")
    spec = PubMediaTilingSpec() if spec is None else spec

    if frame_indices is None:
        indices = list(range(len(frames)))
    else:
        indices = [int(idx) for idx in frame_indices]
    if not indices:
        raise ValueError("frame_indices must select at least one frame.")
    for idx in indices:
        if idx < 0 or idx >= len(frames):
            raise IndexError(f"frame index {idx} is outside 0..{len(frames) - 1}.")

    if labels is None:
        labels = [f"frame {idx}" for idx in indices]
    labels = [str(label) for label in labels]
    if len(labels) != len(indices):
        raise ValueError(f"labels has {len(labels)} entries but {len(indices)} frames were selected.")

    selected = [frames[idx] for idx in indices]
    scaled = []
    scaled_sizes = []
    for image in selected:
        width, height = image.size
        scale = float(spec.tile_width_px) / max(1.0, float(width))
        scaled_height = max(1, int(round(float(height) * scale)))
        resized = image.resize((int(spec.tile_width_px), scaled_height), Image.Resampling.NEAREST)
        scaled.append(resized)
        scaled_sizes.append(resized.size)

    tile_height = max(height for _, height in scaled_sizes)
    rows, columns = spec.grid_shape(len(selected))
    sheet_width = int(2 * spec.pad_px + columns * spec.tile_width_px + (columns - 1) * spec.gap_px)
    cell_height = int(spec.label_height_px + tile_height)
    sheet_height = int(2 * spec.pad_px + rows * cell_height + (rows - 1) * spec.gap_px)
    sheet = Image.new("RGB", (sheet_width, sheet_height), tuple(spec.background_rgb))
    draw = ImageDraw.Draw(sheet)
    font, font_metadata = _pub_media_label_font(int(spec.label_font_px))

    label_layout = []
    for out_idx, (image, label, source_idx) in enumerate(zip(scaled, labels, indices)):
        row = out_idx // columns
        col = out_idx % columns
        x0 = int(spec.pad_px + col * (spec.tile_width_px + spec.gap_px))
        y0 = int(spec.pad_px + row * (cell_height + spec.gap_px))
        label_slot = (x0, y0, x0 + int(spec.tile_width_px), y0 + int(spec.label_height_px))
        display = _pub_media_ellipsize(draw, label, font, int(spec.tile_width_px))
        bbox = draw.textbbox((0, 0), display, font=font)
        text_w = int(bbox[2] - bbox[0])
        text_h = int(bbox[3] - bbox[1])
        tx = int(round(0.5 * (label_slot[0] + label_slot[2] - text_w) - bbox[0]))
        ty = int(round(0.5 * (label_slot[1] + label_slot[3] - text_h) - bbox[1]))
        draw.text((tx, ty), display, fill=tuple(spec.label_rgb), font=font)
        image_y = y0 + int(spec.label_height_px) + int(round(0.5 * (tile_height - image.size[1])))
        sheet.paste(image, (x0, image_y))
        draw.rectangle(
            (x0, y0 + int(spec.label_height_px), x0 + int(spec.tile_width_px), y0 + int(spec.label_height_px) + tile_height),
            outline=tuple(spec.border_rgb),
            width=1,
        )
        label_layout.append(
            {
                "source_frame_index": int(source_idx),
                "label": label,
                "display_label": display,
                "cell": [int(x0), int(y0), int(spec.tile_width_px), int(cell_height)],
            }
        )

    metadata = {
        "rows": int(rows),
        "columns": int(columns),
        "tile_width_px": int(spec.tile_width_px),
        "tile_height_px": int(tile_height),
        "gap_px": int(spec.gap_px),
        "pad_px": int(spec.pad_px),
        "label_height_px": int(spec.label_height_px),
        "label_font_px": int(spec.label_font_px),
        "font": font_metadata,
        "style": {
            "background_rgb": [int(value) for value in spec.background_rgb],
            "border_rgb": [int(value) for value in spec.border_rgb],
            "label_rgb": [int(value) for value in spec.label_rgb],
        },
        "selected_frame_indices": [int(idx) for idx in indices],
        "source_frame_count": int(len(frames)),
        "source_frame_size_px": [int(frames[0].size[0]), int(frames[0].size[1])],
        "sheet_size_px": [int(sheet_width), int(sheet_height)],
        "labels": label_layout,
    }
    return sheet, metadata


def save_pub_frame_contact_sheet(
    output_dir: str | Path,
    stem: str,
    frames,
    *,
    labels=None,
    frame_indices=None,
    spec: PubMediaTilingSpec | None = None,
    metadata: dict[str, Any] | None = None,
    save_pdf: bool = True,
):
    """Save a tiled rendered-frame contact sheet as PNG, optional PDF, and JSON."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet, sheet_meta = make_pub_frame_contact_sheet(
        frames,
        labels=labels,
        frame_indices=frame_indices,
        spec=spec,
    )
    png_path = output_dir / f"{stem}.png"
    sheet.save(png_path)
    paths: dict[str, Any] = {"png": png_path}
    if save_pdf:
        pdf_path = output_dir / f"{stem}.pdf"
        sheet.save(pdf_path)
        paths["pdf"] = pdf_path

    manifest = {
        "artifact_role": "rendered_frame_contact_sheet",
        "contact_sheet": sheet_meta,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    if metadata:
        manifest["metadata"] = metadata
    manifest_path = output_dir / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["manifest"] = manifest_path
    paths["metadata"] = manifest
    return paths


def save_pub_frame_bundle(output_dir: str | Path, stem: str, frames, *, fps: int, save_gif: bool = True):
    """Save RGB/PIL frames as MP4, optional GIF, and a middle-frame preview PNG."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [_coerce_rgb_image(frame) for frame in frames]
    mp4_path = output_dir / f"{stem}.mp4"
    png_path = output_dir / f"{stem}_preview.png"
    frames[len(frames) // 2].save(png_path)
    paths = {"mp4": mp4_path, "preview": png_path}
    if save_gif:
        gif_path = output_dir / f"{stem}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=max(1, int(round(1000.0 / max(int(fps), 1)))),
            loop=0,
            optimize=False,
        )
        paths["gif"] = gif_path
    _save_mp4(mp4_path, frames, fps=fps)
    return paths


def save_pub_env_frame_bundle(
    output_dir: str | Path,
    stem: str,
    frames,
    *,
    fps: int | None = None,
    phase_timing: dict | None = None,
    render_kind: str = "animated",
    min_motion_frames: int | None = None,
    require_motion: bool | None = None,
    require_nonblank: bool = True,
    min_frame_delta: float = 0.5,
    min_pixel_std: float = 1.0,
    save_gif: bool = True,
):
    """Save env/game graphics media after enforcing the env video contract."""

    spec = PubEnvVideoSpec(
        fps=fps,
        phase_timing=phase_timing,
        render_kind=render_kind,
        min_motion_frames=min_motion_frames,
        require_motion=require_motion,
        require_nonblank=require_nonblank,
        min_frame_delta=min_frame_delta,
        min_pixel_std=min_pixel_std,
    )
    return PubEnvFrameBundle(output_dir=output_dir, stem=stem, spec=spec, save_gif=bool(save_gif)).save(frames)


def save_pub_env_base_animation_bundle(
    output_dir: str | Path,
    stem: str,
    frames,
    *,
    fps: int | None = None,
    phase_timing: dict | None = None,
    primitive_sources=(),
    dependencies=(),
    render_kind: str = "animated",
    min_motion_frames: int | None = None,
    require_motion: bool | None = None,
    require_nonblank: bool = True,
    min_frame_delta: float = 0.5,
    min_pixel_std: float = 1.0,
    save_gif: bool = True,
):
    """Save a reusable env animation primitive bundle plus its provenance manifest."""

    spec = PubEnvVideoSpec(
        fps=fps,
        phase_timing=phase_timing,
        render_kind=render_kind,
        min_motion_frames=min_motion_frames,
        require_motion=require_motion,
        require_nonblank=require_nonblank,
        min_frame_delta=min_frame_delta,
        min_pixel_std=min_pixel_std,
    )
    return PubEnvFrameBundle(
        output_dir=output_dir,
        stem=stem,
        spec=spec,
        primitive_sources=tuple(str(source) for source in primitive_sources),
        dependencies=tuple(str(dependency) for dependency in dependencies),
        save_gif=bool(save_gif),
    ).save_base_animation(frames)


__all__ = [
    "PUB_ENV_ANIMATION_FPS",
    "PUB_ENV_PHASE_TIMING",
    "PubEnvFrameBundle",
    "PubMediaTilingSpec",
    "PubEnvVideoSpec",
    "assert_pub_env_symmetry_contract",
    "assert_pub_env_video_contract",
    "make_pub_frame_contact_sheet",
    "pub_env_ease",
    "pub_env_lerp",
    "pub_env_motion_path",
    "pub_env_phase_at",
    "pub_env_progress_values",
    "pub_env_snapshot_frames",
    "pub_env_timing",
    "save_pub_frame_contact_sheet",
    "save_pub_env_base_animation_bundle",
    "save_pub_env_frame_bundle",
    "save_pub_frame_bundle",
]
