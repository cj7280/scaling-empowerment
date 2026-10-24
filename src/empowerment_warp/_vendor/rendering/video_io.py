"""Save rendered RGB frames as local MP4 and PDF artifacts."""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the active smoke env.
    cv2 = None


def _mp4_safe_frames(frames):
    first = np.asarray(frames[0], dtype=np.uint8)
    height, width = first.shape[:2]
    safe_width = int(width) + int(width % 2)
    safe_height = int(height) + int(height % 2)
    if safe_width == int(width) and safe_height == int(height):
        return [np.asarray(frame, dtype=np.uint8)[..., :3] for frame in frames]

    safe_frames = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)[..., :3]).convert('RGB')
        padded = Image.new('RGB', (safe_width, safe_height), (248, 248, 244))
        padded.paste(image, (0, 0))
        safe_frames.append(np.asarray(padded, dtype=np.uint8))
    return safe_frames


def _save_video_without_cv2(path, frames, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = _mp4_safe_frames(frames)
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        first = np.asarray(frames[0], dtype=np.uint8)
        height, width = first.shape[:2]
        cmd = [
            ffmpeg,
            '-y',
            '-f',
            'rawvideo',
            '-vcodec',
            'rawvideo',
            '-s',
            f'{int(width)}x{int(height)}',
            '-pix_fmt',
            'rgb24',
            '-r',
            str(max(int(fps), 1)),
            '-i',
            '-',
            '-an',
            '-vcodec',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            '-movflags',
            '+faststart',
            str(path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for frame in frames:
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        proc.stdin.close()
        proc.stderr.read() if proc.stderr is not None else b''
        if proc.wait() == 0:
            return path

    try:
        import imageio.v2 as imageio

        with imageio.get_writer(
            str(path),
            format='FFMPEG',
            fps=max(int(fps), 1),
            codec='libx264',
            quality=8,
            macro_block_size=1,
        ) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))
        return path
    except Exception:
        return None


def _save_video(path, frames, fps):
    if frames is None or len(frames) == 0:
        return None
    if cv2 is None:
        return _save_video_without_cv2(path, frames, fps)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (int(width), int(height)))
    if not writer.isOpened():
        return None
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path




def _save_frame_pdf(path, frame):
    """Save a rendered RGB still as the canonical PDF artifact."""
    if frame is None:
        return None
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim < 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
        return None

    os.makedirs(os.path.dirname(path), exist_ok=True)
    from empowerment_warp._vendor.plotting.plot_style import configure_rendering_environment

    configure_rendering_environment()
    from matplotlib import pyplot as plt
    from empowerment_warp._vendor.plotting.plot_style import pub_rcparams, save_pub_figure

    height, width = frame.shape[:2]
    aspect = float(width) / float(max(height, 1))
    with plt.rc_context(pub_rcparams()):
        fig, ax = plt.subplots(figsize=(7.2 * aspect, 7.2), dpi=300)
        ax.imshow(frame)
        ax.axis('off')
        paths = save_pub_figure(
            fig,
            path,
            formats=('pdf',),
            bbox_inches='tight',
            pad_inches=0.0,
        )
        plt.close(fig)
    return str(paths[0]) if paths else None
