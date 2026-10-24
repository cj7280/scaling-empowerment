"""Render MuJoCo overlay videos for empowerment evaluations.
This module contains scene helpers for ghost trajectories, branch comparisons, and saved video bundles.
"""

import contextlib

from concurrent.futures import ThreadPoolExecutor, as_completed

import json

import os

import threading

import time

import mujoco

import numpy as np

from PIL import Image, ImageDraw

from .mujoco_cleanup import install_safe_mujoco_renderer_cleanup

from empowerment_warp._vendor.plotting.plot_style import (
    BUILDERBENCH_HIGH_CONTRAST_RGB01,
    evenly_spaced_twilight_rgb01,
    skill_to_rgb01,
)

from .video_io import _save_frame_pdf, _save_video

from .visualization import add_rollout_label_band, add_video_label_band

from empowerment_warp._vendor.plotting.video_render import (
    _draw_legend as _draw_existing_video_legend,
    save_pub_env_frame_bundle,
)

ANT_BODY_ALPHA = 0.86

ANT_BODY_ALPHA_GRAY = 0.94

ANT_TRAIL_ALPHA = 0.78

ANT_TRAIL_ALPHA_GRAY = 0.84

ANT_TRAIL_RADIUS = 0.048

ANT_TRAIL_Z = 0.03

ANT_MARKER_Z = 0.06

ANT_BRANCH_MARKER_RADIUS = 0.11

BALL_SOLID_ALPHA = 0.96

BALL_GHOST_ALPHA = 0.84

BALL_SOLID_RGB = np.array([0.92, 0.93, 0.95], dtype=np.float32)

TAN_RGB = np.array([0.67, 0.50, 0.33], dtype=np.float32)

GRAY_RGB = np.array([0.90, 0.92, 0.96], dtype=np.float32)

GROUND_RGB = np.array([0.85, 0.80, 0.74], dtype=np.float32)

WALL_RGB = np.array([0.78, 0.73, 0.67], dtype=np.float32)

TARGET_RGB = np.array([0.73, 0.68, 0.61], dtype=np.float32)

TARGET_ALPHA = 0.0

MATTE_SPECULAR = 0.03

MATTE_SHININESS = 0.02

HEADLIGHT_DIFFUSE = np.array([0.28, 0.25, 0.22], dtype=np.float32)

HEADLIGHT_AMBIENT = np.array([0.16, 0.14, 0.12], dtype=np.float32)

HEADLIGHT_SPECULAR = np.array([0.05, 0.04, 0.03], dtype=np.float32)

OVERLAY_SKILL_RGBS = np.asarray(evenly_spaced_twilight_rgb01(6, phase_offset=0.04), dtype=np.float32)

MAX_OVERLAY_SKILLS = 6

ANT_OVERLAY_RENDER_MIN_SIZE = 960

BUILDERBENCH_OVERLAY_SKILL_RGBS = np.asarray(
    BUILDERBENCH_HIGH_CONTRAST_RGB01,
    dtype=np.float32,
)

MAX_BUILDERBENCH_OVERLAY_SKILLS = 4

OVERLAY_RENDER_MIN_SIZE = 640

BALL_SPLIT_DISTANCE = 0.08

BUILDERBENCH_ARM_ALPHA = 0.42

BUILDERBENCH_GRAY_ARM_ALPHA = 0.48

BUILDERBENCH_BLOCK_ALPHA = 0.28

BUILDERBENCH_GHOST_BLOCK_ALPHA = 0.22

BUILDERBENCH_GRAY_BLOCK_ALPHA = 0.34

BUILDERBENCH_TRAIL_ALPHA = 0.64

BUILDERBENCH_GRAY_TRAIL_ALPHA = 0.72

BUILDERBENCH_TRAIL_RADIUS = 0.0055

BUILDERBENCH_BRANCH_MARKER_RADIUS = 0.022

BUILDERBENCH_BLOCK_TINT_BASE = np.array([0.86, 0.82, 0.78], dtype=np.float32)

BUILDERBENCH_GRAY_BLOCK_RGB = np.array([0.82, 0.84, 0.88], dtype=np.float32)

BUILDERBENCH_WORKSPACE_CORNER_RGB = np.array([0.94, 0.95, 0.97], dtype=np.float32)

BUILDERBENCH_WORKSPACE_CORNER_ALPHA = 0.72

BUILDERBENCH_WORKSPACE_CORNER_RADIUS = 0.0038

BRANCH_PREFIX_TOL = 1e-6

MIN_ENV_VIDEO_FPS = 24

install_safe_mujoco_renderer_cleanup(mujoco)

def _get_render_base_env(env):
    base_env = getattr(env, 'unwrapped', env)
    if not (
        hasattr(base_env, 'model')
        and hasattr(base_env, 'data')
        and hasattr(base_env, 'set_state')
        and hasattr(base_env, 'render')
        and hasattr(base_env, 'custom_renderer')
    ):
        return None
    return base_env

def _get_builderbench_render_base_env(env):
    base_env = getattr(env, 'unwrapped', env)
    render_env = getattr(base_env, '_base_env', None)
    if render_env is None:
        render_env = getattr(base_env, '_env', None)
    if render_env is None:
        render_env = base_env
    if not (
        hasattr(render_env, 'model')
        and hasattr(render_env, 'data')
        and hasattr(render_env, 'render')
    ):
        return None
    return render_env

def _body_is_descendant(model, body_id, ancestor_id):
    while body_id >= 0:
        if body_id == ancestor_id:
            return True
        if body_id == 0:
            break
        body_id = int(model.body_parentid[body_id])
    return False

def _find_ant_geom_ids(model):
    return _find_body_descendant_geom_ids(model, 'torso')

def _find_body_descendant_geom_ids(model, body_name):
    ancestor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if ancestor_id < 0:
        return []
    geom_ids = []
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if _body_is_descendant(model, body_id, ancestor_id):
            geom_ids.append(geom_id)
    return geom_ids

def _find_optional_geom_ids(model, names):
    geom_ids = []
    for name in names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            geom_ids.append(int(geom_id))
    return geom_ids

def _find_prefixed_geom_ids(model, prefix):
    return [geom_id for geom_id in range(model.ngeom) if _geom_name(model, geom_id).startswith(prefix)]

def _geom_name(model, geom_id):
    try:
        return model.geom(geom_id).name or ''
    except Exception:
        return ''

def _uniform_subsample_items(items, max_count):
    items = list(items)
    if len(items) <= max_count:
        return items

    raw_indices = np.linspace(0, len(items) - 1, num=max_count)
    selected = []
    seen = set()
    for value in raw_indices:
        idx = int(round(float(value)))
        idx = min(max(idx, 0), len(items) - 1)
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
    if len(selected) < max_count:
        for idx in range(len(items)):
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
            if len(selected) == max_count:
                break
    return [items[idx] for idx in selected[:max_count]]

def _array_prefix_max_abs_diff(reference, candidate, prefix_len):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    prefix_len = int(prefix_len)
    if prefix_len <= 0:
        return 0.0
    if reference.ndim == 0 or candidate.ndim == 0:
        return float('inf')
    if reference.shape[0] < prefix_len or candidate.shape[0] < prefix_len:
        return float('inf')
    if reference.shape[1:] != candidate.shape[1:]:
        return float('inf')
    reference = reference[:prefix_len]
    candidate = candidate[:prefix_len]
    if not np.issubdtype(reference.dtype, np.number):
        return 0.0 if np.array_equal(reference, candidate) else float('inf')
    diff = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    return float(np.max(diff)) if diff.size else 0.0

def _branch_prefix_invariant_report(traces, activation_index, *, pe_trace=None, keys=()):
    traces = [trace for trace in (traces or []) if trace]
    if pe_trace is not None:
        traces = traces + [pe_trace]
    prefix_len = max(1, int(activation_index) + 1)
    report = {
        'checked': False,
        'valid': True,
        'prefix_frame_count': int(prefix_len),
        'max_abs_delta_by_key': {},
    }
    if len(traces) <= 1:
        return report

    for key in keys:
        values = [
            trace[key]
            for trace in traces
            if key in trace and np.asarray(trace[key]).ndim > 0 and np.asarray(trace[key]).shape[0] >= prefix_len
        ]
        if len(values) <= 1:
            continue
        reference = values[0]
        max_delta = 0.0
        for value in values[1:]:
            max_delta = max(max_delta, _array_prefix_max_abs_diff(reference, value, prefix_len))
        report['checked'] = True
        report['max_abs_delta_by_key'][key] = max_delta
        if max_delta > BRANCH_PREFIX_TOL:
            report['valid'] = False
    return report

def _assert_branch_prefix_invariant(report, label):
    if report.get('checked') and not report.get('valid', True):
        raise AssertionError(
            f'{label} branch traces do not share an identical potential prefix: '
            f'{report.get("max_abs_delta_by_key", {})}'
        )

def _env_contract_fps(fps):
    return max(MIN_ENV_VIDEO_FPS, int(fps))

def _skill_count_label(count):
    count = int(count)
    return 'no skills' if count <= 0 else ('1 skill' if count == 1 else f'{count} skills')

def _branch_rollout_label(raw_frame_idx, activation_index, skill_count):
    policy = 'potential' if int(raw_frame_idx) <= int(activation_index) else 'skill'
    return f'policy: {policy} | targets: {_skill_count_label(skill_count)}'

def _rendered_raw_frame_index(rendered_indices, frame_idx):
    if not rendered_indices:
        return int(frame_idx)
    idx = min(max(int(frame_idx), 0), len(rendered_indices) - 1)
    return int(rendered_indices[idx])

def _save_env_overlay_bundle(
    *,
    bundle,
    output_dir,
    stem,
    saved_key,
    frames,
    fps,
    artifact_role,
    metadata,
    require_motion=False,
):
    if frames is None or len(frames) == 0:
        return
    paths = save_pub_env_frame_bundle(
        output_dir,
        stem,
        frames,
        fps=_env_contract_fps(fps),
        render_kind='animated',
        require_motion=bool(require_motion),
        save_gif=False,
    )
    saved = {key: str(value) for key, value in paths.items() if key != 'contract'}
    manifest = {
        'name': stem,
        'artifact_role': artifact_role,
        'metadata': dict(metadata or {}),
        'pub_env_contract': dict(paths.get('contract') or {}),
        'outputs': saved,
    }
    manifest_path = os.path.join(output_dir, f'{stem}_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    if saved.get('mp4'):
        bundle['saved_paths'][f'{saved_key}:{output_dir}'] = saved['mp4']
    if saved.get('preview'):
        bundle['saved_paths'][f'{saved_key}_preview:{output_dir}'] = saved['preview']
    bundle['saved_paths'][f'{saved_key}_manifest:{output_dir}'] = manifest_path

def supports_builderbench_scene_overlay_videos(env):
    base_env = _get_builderbench_render_base_env(env)
    if base_env is None:
        return False
    return (
        len(_find_body_descendant_geom_ids(base_env.model, 'base_mount')) > 0
        and len(_find_prefixed_geom_ids(base_env.model, 'block_')) > 0
    )

def _builderbench_block_rgba(rgb, alpha):
    tint = 0.82 * np.asarray(rgb, dtype=np.float32) + 0.18 * BUILDERBENCH_BLOCK_TINT_BASE
    return np.array([*tint, alpha], dtype=np.float32)

class _AntGhostRenderer:
    def __init__(self, env, trail_steps=72, trail_stride=3):
        self.base_env = _get_render_base_env(env)
        self.model = self.base_env.model
        self.ant_geom_ids = _find_ant_geom_ids(self.model)
        self.ball_geom_ids = _find_optional_geom_ids(self.model, ['ball'])
        self.target_geom_ids = _find_optional_geom_ids(self.model, ['target'])
        self.floor_geom_ids = _find_optional_geom_ids(self.model, ['floor'])
        self.wall_geom_ids = [
            geom_id for geom_id in range(self.model.ngeom) if _geom_name(self.model, geom_id).startswith('block_')
        ]
        self.hide_geom_ids = sorted(
            set(self.ant_geom_ids + self.ball_geom_ids + self.target_geom_ids)
        )
        self.floor_mat_ids = sorted(
            {
                int(self.model.geom_matid[geom_id])
                for geom_id in self.floor_geom_ids
                if int(self.model.geom_matid[geom_id]) >= 0
            }
        )
        self.wall_mat_ids = sorted(
            {
                int(self.model.geom_matid[geom_id])
                for geom_id in self.wall_geom_ids
                if int(self.model.geom_matid[geom_id]) >= 0
            }
        )
        self.ant_mat_ids = sorted(
            {
                int(self.model.geom_matid[geom_id])
                for geom_id in self.ant_geom_ids
                if int(self.model.geom_matid[geom_id]) >= 0
            }
        )
        self.target_mat_ids = sorted(
            {
                int(self.model.geom_matid[geom_id])
                for geom_id in self.target_geom_ids
                if int(self.model.geom_matid[geom_id]) >= 0
            }
        )
        self._thread_local = threading.local()
        self.trail_steps = max(0, int(trail_steps))
        self.trail_stride = max(1, int(trail_stride))

    def _temp_data(self):
        temp_data = getattr(self._thread_local, 'temp_data', None)
        if temp_data is None:
            temp_data = mujoco.MjData(self.model)
            self._thread_local.temp_data = temp_data
        return temp_data

    @contextlib.contextmanager
    def hidden_base_geoms(self):
        original_rgba = np.array(self.model.geom_rgba, copy=True)
        if self.hide_geom_ids:
            self.model.geom_rgba[self.hide_geom_ids, 3] = 0.0
        try:
            yield
        finally:
            self.model.geom_rgba[:] = original_rgba

    @contextlib.contextmanager
    def empowerment_background(self):
        original_geom_rgba = np.array(self.model.geom_rgba, copy=True)
        original_mat_rgba = np.array(self.model.mat_rgba, copy=True)
        original_mat_specular = (
            np.array(self.model.mat_specular, copy=True) if hasattr(self.model, 'mat_specular') else None
        )
        original_mat_shininess = (
            np.array(self.model.mat_shininess, copy=True) if hasattr(self.model, 'mat_shininess') else None
        )
        original_light_ambient = (
            np.array(self.model.light_ambient, copy=True) if hasattr(self.model, 'light_ambient') else None
        )
        original_light_diffuse = (
            np.array(self.model.light_diffuse, copy=True) if hasattr(self.model, 'light_diffuse') else None
        )
        original_light_specular = (
            np.array(self.model.light_specular, copy=True) if hasattr(self.model, 'light_specular') else None
        )
        headlight = getattr(getattr(self.model, 'vis', None), 'headlight', None)
        original_headlight_ambient = (
            np.array(headlight.ambient, copy=True) if headlight is not None and hasattr(headlight, 'ambient') else None
        )
        original_headlight_diffuse = (
            np.array(headlight.diffuse, copy=True) if headlight is not None and hasattr(headlight, 'diffuse') else None
        )
        original_headlight_specular = (
            np.array(headlight.specular, copy=True)
            if headlight is not None and hasattr(headlight, 'specular')
            else None
        )
        tex_attr_name = 'tex_rgb' if hasattr(self.model, 'tex_rgb') else 'tex_data'
        tex_buffer = getattr(self.model, tex_attr_name)
        original_tex_buffer = np.array(tex_buffer, copy=True)
        try:
            ground_rgb255 = np.round(255.0 * GROUND_RGB).astype(np.uint8)
            tex_rgb = np.tile(ground_rgb255[None, :], (tex_buffer.size // 3, 1)).reshape(-1)
            tex_buffer[:] = tex_rgb
            if self.floor_geom_ids:
                self.model.geom_rgba[self.floor_geom_ids] = np.array([*GROUND_RGB, 1.0], dtype=np.float32)
            if self.wall_geom_ids:
                self.model.geom_rgba[self.wall_geom_ids] = np.array([*WALL_RGB, 1.0], dtype=np.float32)
            if self.floor_mat_ids:
                self.model.mat_rgba[self.floor_mat_ids] = np.array([*GROUND_RGB, 1.0], dtype=np.float32)
            if self.wall_mat_ids:
                self.model.mat_rgba[self.wall_mat_ids] = np.array([*WALL_RGB, 1.0], dtype=np.float32)
            if self.ant_mat_ids:
                self.model.mat_rgba[self.ant_mat_ids] = np.array([*TAN_RGB, 1.0], dtype=np.float32)
            if self.target_geom_ids:
                self.model.geom_rgba[self.target_geom_ids] = np.array(
                    [*TARGET_RGB, TARGET_ALPHA],
                    dtype=np.float32,
                )
            if self.target_mat_ids:
                self.model.mat_rgba[self.target_mat_ids] = np.array([*TARGET_RGB, TARGET_ALPHA], dtype=np.float32)
            matte_mat_ids = sorted(set(self.floor_mat_ids + self.wall_mat_ids + self.target_mat_ids))
            if matte_mat_ids and original_mat_specular is not None:
                self.model.mat_specular[matte_mat_ids] = MATTE_SPECULAR
            if matte_mat_ids and original_mat_shininess is not None:
                self.model.mat_shininess[matte_mat_ids] = MATTE_SHININESS
            if original_light_ambient is not None:
                self.model.light_ambient[:] = np.minimum(self.model.light_ambient, 0.18)
            if original_light_diffuse is not None:
                self.model.light_diffuse[:] = np.minimum(self.model.light_diffuse, 0.26)
            if original_light_specular is not None:
                self.model.light_specular[:] = np.minimum(self.model.light_specular, 0.04)
            if original_headlight_ambient is not None:
                headlight.ambient[:] = HEADLIGHT_AMBIENT
            if original_headlight_diffuse is not None:
                headlight.diffuse[:] = HEADLIGHT_DIFFUSE
            if original_headlight_specular is not None:
                headlight.specular[:] = HEADLIGHT_SPECULAR
            yield
        finally:
            self.model.geom_rgba[:] = original_geom_rgba
            self.model.mat_rgba[:] = original_mat_rgba
            if original_mat_specular is not None:
                self.model.mat_specular[:] = original_mat_specular
            if original_mat_shininess is not None:
                self.model.mat_shininess[:] = original_mat_shininess
            if original_light_ambient is not None:
                self.model.light_ambient[:] = original_light_ambient
            if original_light_diffuse is not None:
                self.model.light_diffuse[:] = original_light_diffuse
            if original_light_specular is not None:
                self.model.light_specular[:] = original_light_specular
            if original_headlight_ambient is not None:
                headlight.ambient[:] = original_headlight_ambient
            if original_headlight_diffuse is not None:
                headlight.diffuse[:] = original_headlight_diffuse
            if original_headlight_specular is not None:
                headlight.specular[:] = original_headlight_specular
            tex_buffer[:] = original_tex_buffer

    def _append_geom(self, scene, geom_type, size, pos, mat, rgba):
        if scene.ngeom >= scene.maxgeom:
            return False
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            geom_type,
            np.asarray(size, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.asarray(mat, dtype=np.float64),
            np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1
        return True

    def _append_connector(self, scene, from_pt, to_pt, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return False
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([radius, radius, radius], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(radius),
            np.asarray(from_pt, dtype=np.float64),
            np.asarray(to_pt, dtype=np.float64),
        )
        scene.ngeom += 1
        return True

    def add_ant(self, scene, qpos, rgba):
        temp_data = self._temp_data()
        temp_data.qpos[:] = qpos
        if temp_data.qvel.size > 0:
            temp_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, temp_data)

        for geom_id in self.ant_geom_ids:
            if scene.ngeom >= scene.maxgeom:
                break
            geom_type = int(self.model.geom_type[geom_id])
            size = np.array(self.model.geom_size[geom_id], copy=True)
            pos = np.array(temp_data.geom_xpos[geom_id], copy=True)
            mat = np.array(temp_data.geom_xmat[geom_id], copy=True)
            self._append_geom(scene, geom_type, size, pos, mat, rgba)

    def add_ball(self, scene, qpos, rgba):
        if not self.ball_geom_ids:
            return
        temp_data = self._temp_data()
        temp_data.qpos[:] = qpos
        if temp_data.qvel.size > 0:
            temp_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, temp_data)

        for geom_id in self.ball_geom_ids:
            if scene.ngeom >= scene.maxgeom:
                break
            geom_type = int(self.model.geom_type[geom_id])
            size = np.array(self.model.geom_size[geom_id], copy=True)
            pos = np.array(temp_data.geom_xpos[geom_id], copy=True)
            mat = np.array(temp_data.geom_xmat[geom_id], copy=True)
            self._append_geom(scene, geom_type, size, pos, mat, rgba)

    def add_trail(self, scene, positions, frame_idx, rgba, start_idx=0):
        if self.trail_steps <= 0 or len(positions) < 2:
            return
        end_idx = min(int(frame_idx), len(positions) - 1)
        start_idx = max(int(start_idx), 0)
        if end_idx - start_idx < 1:
            return

        # Keep the full visible history from `start_idx` onward so paths remain
        # legible throughout the rollout. `trail_stride` still subsamples the
        # support points to keep scene geometry bounded.
        indices = list(range(start_idx, end_idx + 1, self.trail_stride))
        if indices[-1] != end_idx:
            indices.append(end_idx)
        if len(indices) < 2:
            return

        seg_rgba = np.array(rgba, copy=True, dtype=np.float32)
        for idx0, idx1 in zip(indices[:-1], indices[1:]):
            p0 = np.array([positions[idx0, 0], positions[idx0, 1], ANT_TRAIL_Z], dtype=np.float64)
            p1 = np.array([positions[idx1, 0], positions[idx1, 1], ANT_TRAIL_Z], dtype=np.float64)
            if not self._append_connector(scene, p0, p1, ANT_TRAIL_RADIUS, seg_rgba):
                break

    def add_branch_marker(self, scene, xy, color_rgba):
        marker_rgba = np.array(color_rgba, copy=True)
        marker_rgba[3] = max(marker_rgba[3], 0.95)
        self._append_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([ANT_BRANCH_MARKER_RADIUS, 0.0, 0.0], dtype=np.float64),
            np.array([xy[0], xy[1], ANT_MARKER_Z], dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            marker_rgba,
        )

class _BuilderBenchGhostRenderer:
    def __init__(self, env, trail_steps=120, trail_stride=3):
        self.source_env = getattr(env, 'unwrapped', env)
        self.base_env = _get_builderbench_render_base_env(env)
        self.model = self.base_env.model
        full_gripper_geom_ids = _find_body_descendant_geom_ids(self.model, 'base_mount')
        minimalist_gripper_geom_ids = _find_optional_geom_ids(
            self.model,
            ['right_pad1', 'right_pad2', 'left_pad1', 'left_pad2'],
        )
        self.arm_geom_ids = (
            minimalist_gripper_geom_ids
            if len(minimalist_gripper_geom_ids) >= 2
            else full_gripper_geom_ids
        )
        self.block_geom_ids = _find_prefixed_geom_ids(self.model, 'block_')
        self.target_geom_ids = _find_prefixed_geom_ids(self.model, 'target_mocap_')
        self.hide_geom_ids = sorted(set(full_gripper_geom_ids + self.block_geom_ids + self.target_geom_ids))
        self.hide_mat_ids = sorted(
            {
                int(self.model.geom_matid[geom_id])
                for geom_id in self.hide_geom_ids
                if int(self.model.geom_matid[geom_id]) >= 0
            }
        )
        self.task_mocap_targets = np.asarray(getattr(self.base_env, '_task_mocap_targets', ()), dtype=np.int32)
        self._thread_local = threading.local()
        self.trail_steps = max(0, int(trail_steps))
        self.trail_stride = max(1, int(trail_stride))
        self.workspace_bounds = self._resolve_workspace_bounds()

    def _temp_data(self):
        temp_data = getattr(self._thread_local, 'temp_data', None)
        if temp_data is None:
            temp_data = mujoco.MjData(self.model)
            self._thread_local.temp_data = temp_data
        return temp_data

    def _smooth_trail_positions(self, positions):
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[0] < 4:
            return positions
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        kernel /= np.sum(kernel)
        smoothed = np.array(positions, copy=True)
        for dim in range(positions.shape[1]):
            padded = np.pad(positions[:, dim], (2, 2), mode='edge')
            smoothed[:, dim] = np.convolve(padded, kernel, mode='valid')
        return smoothed

    def _resolve_workspace_bounds(self):
        for candidate in (self.source_env, self.base_env, getattr(self.source_env, '_env', None)):
            if candidate is None:
                continue
            workspace_bounds = getattr(candidate, '_workspace_bounds', None)
            if workspace_bounds is None:
                continue
            workspace_bounds = np.asarray(workspace_bounds, dtype=np.float32)
            if workspace_bounds.shape == (2, 3):
                return workspace_bounds
        return None

    @contextlib.contextmanager
    def hidden_base_geoms(self):
        original_rgba = np.array(self.model.geom_rgba, copy=True)
        original_mat_rgba = np.array(self.model.mat_rgba, copy=True) if hasattr(self.model, 'mat_rgba') else None
        if self.hide_geom_ids:
            self.model.geom_rgba[self.hide_geom_ids, 3] = 0.0
        if self.hide_mat_ids and original_mat_rgba is not None:
            self.model.mat_rgba[self.hide_mat_ids, 3] = 0.0
        try:
            yield
        finally:
            self.model.geom_rgba[:] = original_rgba
            if original_mat_rgba is not None:
                self.model.mat_rgba[:] = original_mat_rgba

    def _append_geom(self, scene, geom_type, size, pos, mat, rgba, dataid=None, force_flat_shading=False):
        if scene.ngeom >= scene.maxgeom:
            return False
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            geom_type,
            np.asarray(size, dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            np.asarray(mat, dtype=np.float64),
            np.asarray(rgba, dtype=np.float32),
        )
        if dataid is not None and hasattr(geom, 'dataid'):
            geom.dataid = int(dataid)
        if force_flat_shading:
            if hasattr(geom, 'matid'):
                geom.matid = -1
            if hasattr(geom, 'specular'):
                geom.specular = MATTE_SPECULAR
            if hasattr(geom, 'shininess'):
                geom.shininess = MATTE_SHININESS
            if hasattr(geom, 'reflectance'):
                geom.reflectance = 0.0
            if hasattr(geom, 'emission'):
                geom.emission = 0.18
        scene.ngeom += 1
        return True

    def _append_connector(self, scene, from_pt, to_pt, radius, rgba):
        if scene.ngeom >= scene.maxgeom:
            return False
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([radius, radius, radius], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(radius),
            np.asarray(from_pt, dtype=np.float64),
            np.asarray(to_pt, dtype=np.float64),
        )
        scene.ngeom += 1
        return True

    def _forward_state(self, qpos, qvel=None, mocap_pos=None, mocap_quat=None):
        temp_data = self._temp_data()
        temp_data.qpos[:] = qpos
        if temp_data.qvel.size > 0:
            if qvel is None:
                temp_data.qvel[:] = 0.0
            else:
                temp_data.qvel[:] = qvel
        if self.task_mocap_targets.size > 0:
            if mocap_pos is not None:
                temp_data.mocap_pos[self.task_mocap_targets] = np.asarray(mocap_pos)
            if mocap_quat is not None:
                temp_data.mocap_quat[self.task_mocap_targets] = np.asarray(mocap_quat)
        mujoco.mj_forward(self.model, temp_data)
        return temp_data

    def add_arm(self, scene, qpos, rgba, qvel=None, mocap_pos=None, mocap_quat=None):
        temp_data = self._forward_state(qpos, qvel=qvel, mocap_pos=mocap_pos, mocap_quat=mocap_quat)
        for geom_id in self.arm_geom_ids:
            if scene.ngeom >= scene.maxgeom:
                break
            geom_type = int(self.model.geom_type[geom_id])
            size = np.array(self.model.geom_size[geom_id], copy=True)
            pos = np.array(temp_data.geom_xpos[geom_id], copy=True)
            mat = np.array(temp_data.geom_xmat[geom_id], copy=True)
            dataid = int(self.model.geom_dataid[geom_id]) if hasattr(self.model, 'geom_dataid') else -1
            self._append_geom(
                scene,
                geom_type,
                size,
                pos,
                mat,
                rgba,
                dataid=(dataid if dataid >= 0 else None),
                force_flat_shading=True,
            )

    def add_blocks(self, scene, qpos, rgba, qvel=None, mocap_pos=None, mocap_quat=None):
        temp_data = self._forward_state(qpos, qvel=qvel, mocap_pos=mocap_pos, mocap_quat=mocap_quat)
        for geom_id in self.block_geom_ids:
            if scene.ngeom >= scene.maxgeom:
                break
            geom_type = int(self.model.geom_type[geom_id])
            size = np.array(self.model.geom_size[geom_id], copy=True)
            pos = np.array(temp_data.geom_xpos[geom_id], copy=True)
            mat = np.array(temp_data.geom_xmat[geom_id], copy=True)
            dataid = int(self.model.geom_dataid[geom_id]) if hasattr(self.model, 'geom_dataid') else -1
            self._append_geom(
                scene,
                geom_type,
                size,
                pos,
                mat,
                rgba,
                dataid=(dataid if dataid >= 0 else None),
                force_flat_shading=True,
            )

    def add_trail(self, scene, positions, frame_idx, rgba, start_idx=0):
        if self.trail_steps <= 0 or len(positions) < 2:
            return
        positions = np.asarray(positions, dtype=np.float32)
        end_idx = min(int(frame_idx), len(positions) - 1)
        start_idx = max(int(start_idx), 0)
        if end_idx - start_idx < 1:
            return

        trail_positions = self._smooth_trail_positions(positions[start_idx : end_idx + 1])
        if len(trail_positions) < 2:
            return

        indices = list(range(0, len(trail_positions), self.trail_stride))
        if indices[-1] != len(trail_positions) - 1:
            indices.append(len(trail_positions) - 1)
        if len(indices) < 2:
            return

        for idx0, idx1 in zip(indices[:-1], indices[1:]):
            p0 = trail_positions[idx0]
            p1 = trail_positions[idx1]
            if p0.shape[0] >= 3 and p1.shape[0] >= 3:
                from_pt = np.asarray(p0[:3], dtype=np.float64)
                to_pt = np.asarray(p1[:3], dtype=np.float64)
            else:
                from_pt = np.array([p0[0], p0[1], 0.05], dtype=np.float64)
                to_pt = np.array([p1[0], p1[1], 0.05], dtype=np.float64)
            if not self._append_connector(
                scene,
                from_pt,
                to_pt,
                BUILDERBENCH_TRAIL_RADIUS,
                np.asarray(rgba, dtype=np.float32),
            ):
                break

    def add_branch_marker(self, scene, xyz, color_rgba):
        xyz = np.asarray(xyz, dtype=np.float32)
        if xyz.shape[0] >= 3:
            pos = np.array(xyz[:3], dtype=np.float64)
        else:
            pos = np.array([xyz[0], xyz[1], 0.06], dtype=np.float64)
        marker_rgba = np.array(color_rgba, copy=True, dtype=np.float32)
        marker_rgba[3] = max(marker_rgba[3], 0.95)
        self._append_geom(
            scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([BUILDERBENCH_BRANCH_MARKER_RADIUS, 0.0, 0.0], dtype=np.float64),
            pos,
            np.eye(3, dtype=np.float64).reshape(-1),
            marker_rgba,
        )

    def add_workspace_corner_marks(self, scene):
        if self.workspace_bounds is None:
            return

        x_min = float(self.workspace_bounds[0, 0])
        x_max = float(self.workspace_bounds[1, 0])
        y_min = float(self.workspace_bounds[0, 1])
        y_max = float(self.workspace_bounds[1, 1])
        z = float(self.workspace_bounds[0, 2]) + 0.002
        span_x = max(x_max - x_min, 1e-3)
        span_y = max(y_max - y_min, 1e-3)
        corner_len = float(np.clip(0.12 * min(span_x, span_y), 0.025, 0.055))
        rgba = np.array(
            [*BUILDERBENCH_WORKSPACE_CORNER_RGB, BUILDERBENCH_WORKSPACE_CORNER_ALPHA],
            dtype=np.float32,
        )

        corners = (
            ((x_min, y_min), (+corner_len, 0.0), (0.0, +corner_len)),
            ((x_min, y_max), (+corner_len, 0.0), (0.0, -corner_len)),
            ((x_max, y_min), (-corner_len, 0.0), (0.0, +corner_len)),
            ((x_max, y_max), (-corner_len, 0.0), (0.0, -corner_len)),
        )
        for (x0, y0), (dx1, dy1), (dx2, dy2) in corners:
            p_corner = np.array([x0, y0, z], dtype=np.float64)
            p_leg1 = np.array([x0 + dx1, y0 + dy1, z], dtype=np.float64)
            p_leg2 = np.array([x0 + dx2, y0 + dy2, z], dtype=np.float64)
            if not self._append_connector(
                scene,
                p_corner,
                p_leg1,
                BUILDERBENCH_WORKSPACE_CORNER_RADIUS,
                rgba,
            ):
                return
            if not self._append_connector(
                scene,
                p_corner,
                p_leg2,
                BUILDERBENCH_WORKSPACE_CORNER_RADIUS,
                rgba,
            ):
                return

def _render_video_frames(
    base_env,
    scene_renderer,
    frame_count,
    scene_builder,
    frame_stride,
    min_render_size=None,
    light_qpos_seq=None,
    render_workers=1,
):
    del scene_renderer
    if hasattr(base_env, 'initialize_renderer') and getattr(base_env, 'custom_renderer', None) is None:
        base_env.initialize_renderer()
    frame_count = max(0, int(frame_count))
    frame_stride = max(1, int(frame_stride))
    light_qpos_seq = None if light_qpos_seq is None else np.asarray(light_qpos_seq, dtype=np.float64)
    original_qpos = np.array(base_env.data.qpos, copy=True) if light_qpos_seq is not None else None
    original_qvel = np.array(base_env.data.qvel, copy=True) if light_qpos_seq is not None else None
    min_render_size = OVERLAY_RENDER_MIN_SIZE if min_render_size is None else int(min_render_size)
    render_width = max(int(getattr(base_env, 'width', 0) or 0), min_render_size)
    render_height = max(int(getattr(base_env, 'height', 0) or 0), min_render_size)
    vis_global = getattr(getattr(base_env.model, 'vis', None), 'global_', None)
    original_offwidth = int(vis_global.offwidth) if vis_global is not None else None
    original_offheight = int(vis_global.offheight) if vis_global is not None else None
    if vis_global is not None:
        vis_global.offwidth = max(original_offwidth, render_width)
        vis_global.offheight = max(original_offheight, render_height)

    frame_indices = [
        raw_frame_idx
        for raw_frame_idx in range(frame_count)
        if raw_frame_idx % frame_stride == 0 or raw_frame_idx == frame_count - 1
    ]
    if not frame_indices:
        if vis_global is not None and original_offwidth is not None and original_offheight is not None:
            vis_global.offwidth = original_offwidth
            vis_global.offheight = original_offheight
        return None

    camera = getattr(base_env, 'custom_camera', -1)

    def _render_one_frame(renderer, raw_frame_idx):
        if light_qpos_seq is not None and light_qpos_seq.ndim == 2 and light_qpos_seq.shape[0] > 0:
            light_idx = min(raw_frame_idx, light_qpos_seq.shape[0] - 1)
            qpos = np.array(base_env.data.qpos, copy=True)
            qvel = np.zeros_like(base_env.data.qvel)
            qpos[: min(qpos.size, light_qpos_seq.shape[1])] = light_qpos_seq[
                light_idx,
                : min(qpos.size, light_qpos_seq.shape[1]),
            ]
            base_env.set_state(qpos, qvel)
        renderer.update_scene(base_env.data, camera=camera)
        scene_builder(renderer.scene, raw_frame_idx)
        return np.asarray(renderer.render().copy(), dtype=np.uint8)

    def _render_serial():
        frames_local = []
        renderer = mujoco.Renderer(base_env.model, width=render_width, height=render_height)
        try:
            for raw_frame_idx in frame_indices:
                frames_local.append(_render_one_frame(renderer, raw_frame_idx))
        finally:
            renderer.close()
        return frames_local

    def _render_chunk(chunk_indices):
        renderer = mujoco.Renderer(base_env.model, width=render_width, height=render_height)
        try:
            return [(raw_frame_idx, _render_one_frame(renderer, raw_frame_idx)) for raw_frame_idx in chunk_indices]
        finally:
            renderer.close()

    frames = None
    render_workers = max(1, int(render_workers or 1))
    can_parallel_render = render_workers > 1 and light_qpos_seq is None and len(frame_indices) > 1
    try:
        if can_parallel_render:
            actual_workers = min(render_workers, len(frame_indices))
            chunks = [frame_indices[offset::actual_workers] for offset in range(actual_workers)]
            frames_by_index = {}
            try:
                with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                    futures = [executor.submit(_render_chunk, chunk) for chunk in chunks if chunk]
                    for future in as_completed(futures):
                        for raw_frame_idx, frame in future.result():
                            frames_by_index[raw_frame_idx] = frame
                frames = [frames_by_index[raw_frame_idx] for raw_frame_idx in frame_indices]
            except Exception as exc:
                print(
                    f"[eval] parallel MuJoCo overlay rendering failed ({exc}); falling back to serial.",
                    flush=True,
                )
                frames = _render_serial()
        else:
            frames = _render_serial()
    finally:
        if original_qpos is not None and original_qvel is not None:
            base_env.set_state(original_qpos, original_qvel)
        if vis_global is not None and original_offwidth is not None and original_offheight is not None:
            vis_global.offwidth = original_offwidth
            vis_global.offheight = original_offheight
    return np.asarray(frames, dtype=np.uint8) if frames else None

def _render_single_scene_frame(
    base_env,
    scene_renderer,
    scene_builder,
    frame_idx,
    min_render_size=None,
    light_qpos_seq=None,
):
    """Render one exact raw scene frame for publication stills."""
    frame_idx = max(0, int(frame_idx))
    frames = _render_video_frames(
        base_env=base_env,
        scene_renderer=scene_renderer,
        frame_count=frame_idx + 1,
        scene_builder=scene_builder,
        frame_stride=frame_idx + 1,
        min_render_size=min_render_size,
        light_qpos_seq=light_qpos_seq,
    )
    if frames is None or len(frames) == 0:
        return None
    return np.asarray(frames[-1], dtype=np.uint8)

def _overlay_existing_renderer_legend(frames, okabe_rgbs, *, legend_style='builderbench_gray',
                                      include_potential_policy=True, publication=False):
    """Apply the video legend from the configured shared plotting checkout.

    ``include_potential_policy=False`` drops the dashed "Potential policy" row. Pass it
    for potential_steps=0 runs (the skill overlay path): the skill is on from step 0, no
    dashed segment is ever drawn, and keeping the row claims the video shows a
    potential->skill transition it does not contain.
    """
    if frames is None or len(frames) == 0:
        return frames

    if publication:
        from empowerment_warp._vendor.plotting.plot_style import pub_overlay_px_for_frame, pub_uniform_typography_for_width
        from empowerment_warp._vendor.plotting.video_render import add_pub_overlay_legend

        width = np.asarray(frames[0]).shape[1]
        font = pub_uniform_typography_for_width(7.0, pt_per_inch=3.6).legend.fontsize
        return add_pub_overlay_legend(
            frames, okabe_rgbs, font_px=pub_overlay_px_for_frame(font, width, 7.0),
            legend_style=legend_style, include_potential_policy=include_potential_policy,
        )

    okabe_rgbs = np.asarray(okabe_rgbs, dtype=np.float32)
    if okabe_rgbs.ndim != 2 or okabe_rgbs.shape[0] == 0:
        okabe_rgbs = BUILDERBENCH_OVERLAY_SKILL_RGBS
    if okabe_rgbs.shape[0] < 2:
        okabe_rgbs = np.concatenate([okabe_rgbs, okabe_rgbs], axis=0)
    phase_rgb = okabe_rgbs[0]

    frames_out = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode='RGB').convert('RGBA')
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay, 'RGBA')
        _draw_existing_video_legend(
            draw,
            image.size[0],
            okabe_rgbs,
            phase_rgb,
            height=image.size[1],
            legend_style=legend_style,
            include_potential_policy=include_potential_policy,
        )
        frames_out.append(np.asarray(Image.alpha_composite(image, overlay).convert('RGB'), dtype=np.uint8))
    return np.asarray(frames_out, dtype=np.uint8)

def _rendered_frame_indices(frame_count, frame_stride):
    frame_count = max(int(frame_count), 0)
    frame_stride = max(int(frame_stride), 1)
    return [
        int(raw_frame_idx)
        for raw_frame_idx in range(frame_count)
        if raw_frame_idx % frame_stride == 0 or raw_frame_idx == frame_count - 1
    ]

def _first_ball_split_index(traces, threshold=BALL_SPLIT_DISTANCE):
    valid_traces = [trace for trace in traces if len(trace.get('ball_positions', ())) > 0]
    if len(valid_traces) <= 1:
        return None

    frame_count = min(len(trace['ball_positions']) for trace in valid_traces)
    for frame_idx in range(frame_count):
        positions = np.stack([trace['ball_positions'][frame_idx] for trace in valid_traces], axis=0)
        center = np.mean(positions, axis=0)
        max_dist = np.max(np.linalg.norm(positions - center[None, :], axis=-1))
        if max_dist >= float(threshold):
            return frame_idx
    return None

def _select_builderbench_overlay_traces(traces, preserve_skill_bank):
    traces = list(traces)
    if preserve_skill_bank:
        return traces
    return _uniform_subsample_items(traces, MAX_BUILDERBENCH_OVERLAY_SKILLS)

def _builderbench_overlay_rgbs(skill_traces, branch_traces, cyclic_skill_colors):
    color_count = max(len(skill_traces), len(branch_traces))
    if not cyclic_skill_colors:
        return [
            BUILDERBENCH_OVERLAY_SKILL_RGBS[
                idx % len(BUILDERBENCH_OVERLAY_SKILL_RGBS)
            ]
            for idx in range(color_count)
        ]
    color_traces = (
        branch_traces if len(branch_traces) >= len(skill_traces) else skill_traces
    )
    return [
        np.asarray(
            skill_to_rgb01(trace.get('skill', np.zeros(2, dtype=np.float32))),
            dtype=np.float32,
        )
        for trace in color_traces
    ]

def create_builderbench_scene_video_bundle(
    env,
    skill_only_traces,
    two_phase_traces,
    pe_only_traces=None,
    output_dirs=None,
    fps=15,
    trail_steps=120,
    trail_stride=3,
    frame_stride=1,
    render_skill=False,
    preserve_skill_bank=False,
    cyclic_skill_colors=False,
    publication_chrome=False,
    min_render_size=None,
    block_opacity=None,
):
    fps = _env_contract_fps(fps)
    bundle = {
        'skill_arm_video_frames': None,
        'branch_arm_video_frames': None,
        'skill_arm_video_fps': int(fps),
        'branch_arm_video_fps': int(fps),
        'builderbench_overlay_metadata': None,
        'saved_paths': {},
    }
    output_dirs = [] if output_dirs is None else [d for d in output_dirs if d]

    if not supports_builderbench_scene_overlay_videos(env):
        return bundle

    skill_only_traces = list(skill_only_traces or [])
    two_phase_traces = list(two_phase_traces or [])
    pe_only_traces = list(pe_only_traces or [])

    branch_traces = [
        trace
        for trace in two_phase_traces
        if len(trace.get('qpos_seq', ())) > 0 and len(trace.get('effector_positions', ())) > 0
    ]
    if not branch_traces:
        return bundle
    branch_traces = _select_builderbench_overlay_traces(
        branch_traces, preserve_skill_bank
    )

    if render_skill:
        skill_traces = [
            trace
            for trace in skill_only_traces
            if len(trace.get('qpos_seq', ())) > 0 and len(trace.get('effector_positions', ())) > 0
        ]
        if not skill_traces:
            skill_traces = list(branch_traces)
        skill_traces = _select_builderbench_overlay_traces(
            skill_traces, preserve_skill_bank
        )
    else:
        skill_traces = []

    pe_continuation_trace = next(
        (
            trace
            for trace in pe_only_traces
            if len(trace.get('qpos_seq', ())) > 0 and len(trace.get('effector_positions', ())) > 0
        ),
        None,
    )

    base_env = _get_builderbench_render_base_env(env)
    ghost_renderer = _BuilderBenchGhostRenderer(env, trail_steps=trail_steps, trail_stride=trail_stride)

    overlay_rgbs = _builderbench_overlay_rgbs(
        skill_traces, branch_traces, cyclic_skill_colors
    )
    skill_arm_colors = [np.array([*rgb, BUILDERBENCH_ARM_ALPHA], dtype=np.float32) for rgb in overlay_rgbs[: len(skill_traces)]]
    branch_arm_colors = [np.array([*rgb, BUILDERBENCH_ARM_ALPHA], dtype=np.float32) for rgb in overlay_rgbs[: len(branch_traces)]]
    skill_trail_colors = [np.array([*rgb, BUILDERBENCH_TRAIL_ALPHA], dtype=np.float32) for rgb in overlay_rgbs[: len(skill_traces)]]
    branch_trail_colors = [np.array([*rgb, BUILDERBENCH_TRAIL_ALPHA], dtype=np.float32) for rgb in overlay_rgbs[: len(branch_traces)]]
    gray_arm = np.array([*GRAY_RGB, BUILDERBENCH_GRAY_ARM_ALPHA], dtype=np.float32)
    gray_block = np.array([*BUILDERBENCH_GRAY_BLOCK_RGB, BUILDERBENCH_GRAY_BLOCK_ALPHA], dtype=np.float32)
    gray_trail = np.array([*GRAY_RGB, BUILDERBENCH_GRAY_TRAIL_ALPHA], dtype=np.float32)
    if publication_chrome:
        # Adjacent capsule endcaps overlap: opacity removes dark joint bands.
        for trail_color in [*skill_trail_colors, *branch_trail_colors, gray_trail]:
            trail_color[3] = 1.0
    branch_marker_color = np.array([0.98, 0.97, 0.95, 0.94], dtype=np.float32)
    shared_block = np.array([*BUILDERBENCH_GRAY_BLOCK_RGB, BUILDERBENCH_BLOCK_ALPHA], dtype=np.float32)
    skill_block_colors = [
        _builderbench_block_rgba(rgb, BUILDERBENCH_GHOST_BLOCK_ALPHA)
        for rgb in overlay_rgbs[: len(skill_traces)]
    ]
    branch_block_colors = [
        _builderbench_block_rgba(rgb, BUILDERBENCH_GHOST_BLOCK_ALPHA)
        for rgb in overlay_rgbs[: len(branch_traces)]
    ]

    if block_opacity is not None:
        if not 0.0 <= float(block_opacity) <= 1.0:
            raise ValueError('block_opacity must lie in [0, 1]')
        for rgba in [gray_block, shared_block, *skill_block_colors, *branch_block_colors]:
            rgba[3] = float(block_opacity)

    skill_frame_count = min((len(trace['qpos_seq']) for trace in skill_traces), default=0)
    branch_frame_count_candidates = [len(trace['qpos_seq']) for trace in branch_traces]
    if pe_continuation_trace is not None:
        branch_frame_count_candidates.append(len(pe_continuation_trace['qpos_seq']))
    branch_frame_count = min(branch_frame_count_candidates)

    branch_activation_index = min(int(trace.get('activation_index', 0) or 0) for trace in branch_traces)
    branch_activation_index = max(0, min(branch_activation_index, branch_frame_count - 1))
    branch_prefix_report = _branch_prefix_invariant_report(
        branch_traces,
        branch_activation_index,
        pe_trace=pe_continuation_trace,
        keys=(
            'qpos_seq',
            'qvel_seq',
            'effector_positions',
            'cube_positions',
            'mocap_pos_seq',
            'mocap_quat_seq',
        ),
    )
    _assert_branch_prefix_invariant(branch_prefix_report, 'BuilderBench')
    branch_rendered_frame_indices = _rendered_frame_indices(branch_frame_count, frame_stride)

    shared_prefix_qpos = branch_traces[0]['qpos_seq']
    shared_prefix_positions = np.asarray(
        branch_traces[0].get('effector_positions', branch_traces[0].get('positions', ())),
        dtype=np.float32,
    )

    bundle['builderbench_overlay_metadata'] = {
        'skill_count': int(len(branch_traces)),
        'frame_stride': int(frame_stride),
        'fps': int(fps),
        'skill_frame_count': int(skill_frame_count),
        'branch_frame_count': int(branch_frame_count),
        'branch_activation_index': int(branch_activation_index),
        'branch_prefix_invariant': branch_prefix_report,
        'config': {
            'skill_dim': int(branch_traces[0].get('skill', np.zeros(0, dtype=np.float32)).shape[-1]),
            'num_cubes': int(len(ghost_renderer.block_geom_ids)),
        },
    }

    def _trace_positions_3d(trace):
        positions = np.asarray(trace.get('effector_positions', trace.get('positions', ())), dtype=np.float32)
        return positions if positions.ndim == 2 else np.zeros((0, 3), dtype=np.float32)

    def _trace_render_state(trace, frame_idx):
        qpos_seq = np.asarray(trace.get('qpos_seq', ()), dtype=np.float64)
        if qpos_seq.ndim != 2 or qpos_seq.shape[0] == 0:
            return None
        idx = min(int(frame_idx), qpos_seq.shape[0] - 1)
        qvel_seq = np.asarray(trace.get('qvel_seq', ()), dtype=np.float64)
        mocap_pos_seq = np.asarray(trace.get('mocap_pos_seq', ()), dtype=np.float32)
        mocap_quat_seq = np.asarray(trace.get('mocap_quat_seq', ()), dtype=np.float32)
        qvel = qvel_seq[idx] if qvel_seq.ndim == 2 and qvel_seq.shape[0] > idx else None
        mocap_pos = (
            mocap_pos_seq[idx]
            if mocap_pos_seq.ndim == 3 and mocap_pos_seq.shape[0] > idx
            else None
        )
        mocap_quat = (
            mocap_quat_seq[idx]
            if mocap_quat_seq.ndim == 3 and mocap_quat_seq.shape[0] > idx
            else None
        )
        return {
            'qpos': qpos_seq[idx],
            'qvel': qvel,
            'mocap_pos': mocap_pos,
            'mocap_quat': mocap_quat,
        }

    def build_skill_scene(scene, frame_idx):
        ghost_renderer.add_workspace_corner_marks(scene)
        for trace, trail_rgba in zip(skill_traces, skill_trail_colors):
            ghost_renderer.add_trail(scene, _trace_positions_3d(trace), frame_idx, trail_rgba)
        shared_render_state = _trace_render_state(skill_traces[0], frame_idx)
        if shared_render_state is not None:
            ghost_renderer.add_blocks(
                scene,
                shared_render_state['qpos'],
                shared_block,
                shared_render_state['qvel'],
                shared_render_state['mocap_pos'],
                shared_render_state['mocap_quat'],
            )
        for trace, block_rgba in zip(skill_traces, skill_block_colors):
            render_state = _trace_render_state(trace, frame_idx)
            if render_state is None:
                continue
            ghost_renderer.add_blocks(
                scene,
                render_state['qpos'],
                block_rgba,
                render_state['qvel'],
                render_state['mocap_pos'],
                render_state['mocap_quat'],
            )
        for trace, arm_rgba in zip(skill_traces, skill_arm_colors):
            render_state = _trace_render_state(trace, frame_idx)
            if render_state is None:
                continue
            ghost_renderer.add_arm(scene, render_state['qpos'], arm_rgba, render_state['qvel'], render_state['mocap_pos'], render_state['mocap_quat'])

    def _build_branch_like_scene(
        scene,
        frame_idx,
        active_traces,
        active_trail_colors,
        active_block_colors,
        active_arm_colors,
        active_activation_index,
    ):
        ghost_renderer.add_workspace_corner_marks(scene)
        if not active_traces or len(shared_prefix_qpos) == 0 or len(shared_prefix_positions) == 0:
            return
        active_activation_index = int(active_activation_index)
        prefix_idx = min(frame_idx, active_activation_index, len(shared_prefix_positions) - 1)
        ghost_renderer.add_trail(scene, shared_prefix_positions, prefix_idx, gray_trail)
        if frame_idx <= active_activation_index:
            render_state = _trace_render_state(active_traces[0], frame_idx)
            if render_state is None:
                return
            ghost_renderer.add_blocks(scene, render_state['qpos'], gray_block, render_state['qvel'], render_state['mocap_pos'], render_state['mocap_quat'])
            ghost_renderer.add_arm(scene, render_state['qpos'], gray_arm, render_state['qvel'], render_state['mocap_pos'], render_state['mocap_quat'])
            return

        branch_xyz = shared_prefix_positions[active_activation_index]
        ghost_renderer.add_branch_marker(scene, branch_xyz, branch_marker_color)
        if pe_continuation_trace is not None:
            ghost_renderer.add_trail(
                scene,
                _trace_positions_3d(pe_continuation_trace),
                frame_idx,
                gray_trail,
                start_idx=active_activation_index,
            )
            render_state = _trace_render_state(pe_continuation_trace, frame_idx)
            if render_state is not None:
                ghost_renderer.add_blocks(scene, render_state['qpos'], gray_block, render_state['qvel'], render_state['mocap_pos'], render_state['mocap_quat'])
                # Paper snapshots show the skill arms after the switch; the
                # shared white approach trail already identifies the potential policy.
                if not publication_chrome:
                    ghost_renderer.add_arm(scene, render_state['qpos'], gray_arm, render_state['qvel'], render_state['mocap_pos'], render_state['mocap_quat'])
        elif block_opacity is None or len(active_traces) > 1:
            render_state = _trace_render_state(active_traces[0], frame_idx)
            if render_state is not None:
                ghost_renderer.add_blocks(
                    scene,
                    render_state['qpos'],
                    shared_block,
                    render_state['qvel'],
                    render_state['mocap_pos'],
                    render_state['mocap_quat'],
                )

        for trace, trail_rgba in zip(active_traces, active_trail_colors):
            ghost_renderer.add_trail(
                scene,
                _trace_positions_3d(trace),
                frame_idx,
                trail_rgba,
                start_idx=active_activation_index,
            )
        for trace, block_rgba in zip(active_traces, active_block_colors):
            render_state = _trace_render_state(trace, frame_idx)
            if render_state is None:
                continue
            ghost_renderer.add_blocks(
                scene,
                render_state['qpos'],
                block_rgba,
                render_state['qvel'],
                render_state['mocap_pos'],
                render_state['mocap_quat'],
            )
        for trace, arm_rgba in zip(active_traces, active_arm_colors):
            render_state = _trace_render_state(trace, frame_idx)
            if render_state is None:
                continue
            ghost_renderer.add_arm(scene, render_state['qpos'], arm_rgba, render_state['qvel'], render_state['mocap_pos'], render_state['mocap_quat'])

    def build_branch_scene(scene, frame_idx):
        _build_branch_like_scene(
            scene,
            frame_idx,
            branch_traces,
            branch_trail_colors,
            branch_block_colors,
            branch_arm_colors,
            branch_activation_index,
        )

    arm_still_rgbs = overlay_rgbs if len(overlay_rgbs) else BUILDERBENCH_OVERLAY_SKILL_RGBS
    arm_still_frames = {}
    arm_still_frame_indices = {}

    def _render_labeled_arm_still(label, scene_builder, frame_idx, frame_count):
        if int(frame_count) <= 0:
            return
        raw_frame_idx = max(0, min(int(frame_idx), int(frame_count) - 1))
        with ghost_renderer.hidden_base_geoms():
            frame = _render_single_scene_frame(
                base_env=base_env,
                scene_renderer=ghost_renderer,
                scene_builder=scene_builder,
                frame_idx=raw_frame_idx,
                min_render_size=min_render_size,
            )
        if frame is None:
            return
        frames = _overlay_existing_renderer_legend(
            np.asarray([frame], dtype=np.uint8),
            arm_still_rgbs,
            publication=publication_chrome,
        )
        if frames is None or len(frames) == 0:
            return
        arm_still_frames[label] = np.asarray(frames[0], dtype=np.uint8)
        arm_still_frame_indices[label] = int(raw_frame_idx)

    if render_skill and skill_frame_count > 0:
        with ghost_renderer.hidden_base_geoms():
            bundle['skill_arm_video_frames'] = _render_video_frames(
                base_env=base_env,
                scene_renderer=ghost_renderer,
                frame_count=skill_frame_count,
                scene_builder=build_skill_scene,
                frame_stride=frame_stride,
                min_render_size=min_render_size,
            )
        bundle['skill_arm_video_frames'] = _overlay_existing_renderer_legend(
            bundle['skill_arm_video_frames'],
            overlay_rgbs if len(overlay_rgbs) else BUILDERBENCH_OVERLAY_SKILL_RGBS,
        )
        bundle['skill_arm_video_frames'] = add_video_label_band(
            bundle['skill_arm_video_frames'],
            lambda _frame_idx: f'policy: skill | targets: {_skill_count_label(len(skill_traces))}',
        )

    _render_labeled_arm_still(
        'branch_arm_branch_point',
        build_branch_scene,
        branch_activation_index,
        branch_frame_count,
    )
    _render_labeled_arm_still(
        'branch_arm_rollout_end',
        build_branch_scene,
        branch_frame_count - 1,
        branch_frame_count,
    )

    with ghost_renderer.hidden_base_geoms():
        bundle['branch_arm_video_frames'] = _render_video_frames(
            base_env=base_env,
            scene_renderer=ghost_renderer,
            frame_count=branch_frame_count,
            scene_builder=build_branch_scene,
            frame_stride=frame_stride,
            min_render_size=min_render_size,
        )
    bundle['branch_arm_video_frames'] = _overlay_existing_renderer_legend(
        bundle['branch_arm_video_frames'],
        overlay_rgbs if len(overlay_rgbs) else BUILDERBENCH_OVERLAY_SKILL_RGBS,
        publication=publication_chrome,
    )
    if not publication_chrome:
        bundle['branch_arm_video_frames'] = add_video_label_band(
            bundle['branch_arm_video_frames'],
            lambda frame_idx: _branch_rollout_label(
                _rendered_raw_frame_index(branch_rendered_frame_indices, frame_idx),
                branch_activation_index,
                len(branch_traces),
            ),
        )

    bundle['builderbench_overlay_metadata']['arm_still_frame_indices'] = dict(arm_still_frame_indices)

    for output_dir in output_dirs:
        if render_skill:
            skill_path = _save_video(
                os.path.join(output_dir, 'skill_arm_overlay.mp4'),
                bundle['skill_arm_video_frames'],
                fps=fps,
            )
            if skill_path is not None:
                bundle['saved_paths'][f'skill_arm_video:{output_dir}'] = skill_path
        _save_env_overlay_bundle(
            bundle=bundle,
            output_dir=output_dir,
            stem='branch_arm_overlay',
            saved_key='branch_arm_video',
            frames=bundle['branch_arm_video_frames'],
            fps=fps,
            artifact_role='builderbench_branch_arm_overlay',
            metadata=bundle.get('builderbench_overlay_metadata'),
            require_motion=False,
        )
        for still_name, still_frame in sorted(arm_still_frames.items()):
            still_path = _save_frame_pdf(
                os.path.join(output_dir, f'{still_name}.pdf'),
                still_frame,
            )
            if still_path is not None:
                bundle['saved_paths'][f'{still_name}_pdf:{output_dir}'] = still_path
        metadata_path = os.path.join(output_dir, 'builderbench_overlay_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(bundle['builderbench_overlay_metadata'], f, indent=2, sort_keys=True)
        bundle['saved_paths'][f'builderbench_overlay_metadata:{output_dir}'] = metadata_path

    return bundle
