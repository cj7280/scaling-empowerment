"""Fresh shared-prefix branch media for both AntSoccer observation contracts."""

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.training.sac.evaluation import load_run
from empowerment_warp.training.sac.evaluation import eval_skill_library
from empowerment_warp.training.sac.evaluation import evaluation_switch_steps, potential_schedule_metadata
from empowerment_warp.envs.state import gather_env_state_lanes
from empowerment_warp.envs.factory import make_environment
from empowerment_warp.media.builderbench import (
    incumbent_collector,
    _sha256,
)
from empowerment_warp.tracking import WandbMirror, record_pending_upload


BRANCH_PREFIX_STEPS = 100
BRANCH_HORIZON_STEPS = 400
BRANCH_SKILL_COUNT = 4
BRANCH_FPS = 38
BRANCH_FRAME_STRIDE = 1
BRANCH_TRAIL_STEPS = 72
BRANCH_TRAIL_STRIDE = 3
BRANCH_PREFIX_TOL = 1e-6


def circular_skill_indices(num_skills: int, count: int) -> tuple[int, ...]:
    """Cover a periodic library without sampling its wraparound twice."""
    if num_skills <= 0 or count <= 0:
        raise ValueError('num_skills and count must be positive')
    count = min(count, num_skills)
    return tuple(index * num_skills // count for index in range(count))


def _video_skills(
    config: Any, count: int
) -> tuple[jax.Array, tuple[int, ...], str]:
    """Use a circle for continuous 2D video commands, independent of metric probes."""
    if count <= 0:
        raise ValueError("skill_count must be positive")
    if config.skill_dim == 2:
        angles = np.arange(count) * (2.0 * np.pi / count)
        skills = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        skills[np.abs(skills) < 1e-7] = 0.0
        return jnp.asarray(skills, dtype=jnp.float32), tuple(range(count)), "uniform_circle"
    library = eval_skill_library(config)
    indices = circular_skill_indices(library.shape[0], count)
    return library[jnp.asarray(indices)], indices, "eval_library_subset"


def _prefix_report(arrays: dict[str, np.ndarray], activation_index: int) -> dict[str, Any]:
    prefix_len = int(activation_index) + 1
    keys = ["qpos", "qvel", "agent_positions"]
    if "ball_positions" in arrays:
        keys.append("ball_positions")
    maxima = {}
    valid = True
    for name in keys:
        values = np.asarray(arrays[name])[:, :prefix_len]
        delta = np.abs(values.astype(np.float64) - values[0:1].astype(np.float64))
        maximum = float(np.max(delta)) if delta.size else 0.0
        maxima[name] = maximum
        valid = valid and maximum <= BRANCH_PREFIX_TOL
    return {
        "checked": True,
        "valid": bool(valid),
        "prefix_frame_count": prefix_len,
        "max_abs_delta_by_key": maxima,
    }


def collect_locomotion_branch_trace(
    run_dir: str | Path,
    output: str | Path,
    *,
    prefix_steps: int | None = None,
    horizon_steps: int = BRANCH_HORIZON_STEPS,
    skill_count: int = BRANCH_SKILL_COUNT,
    seed_offset: int = 50_000,
    deterministic: bool = False,
    checkpoint: str | Path | None = None,
) -> Path:
    """Collect an identical potential prefix followed by skill branches.

    By default, sample the learned action distribution exactly as training does.
    Random schedules draw one shared boundary; an explicit prefix is a fixed diagnostic override.
    """

    config, experiment, training_state = (load_run(run_dir) if checkpoint is None
                                               else load_run(run_dir, checkpoint=checkpoint))
    checkpoint_step = int(training_state.env_steps)
    prefix_override = prefix_steps
    prefix_steps = int(evaluation_switch_steps(config, 1, seed_offset)[0]) if prefix_steps is None else prefix_steps
    if not 0 <= prefix_steps < horizon_steps:
        raise ValueError("prefix_steps must be nonnegative and less than horizon_steps")
    if config.environment not in {"ant", "ant_soccer"}:
        raise ValueError("locomotion branch media requires an Ant or AntSoccer comparison run")
    selected_skills, selected, skill_selection = _video_skills(
        config, skill_count
    )
    include_potential = config.potential_steps > 0 or config.random_potential_switch_max_steps > 0 or config.potential_lane_fraction > 0
    if (config.skill_start_mode == "marginal_branch" and not config.live_potential_only
            and not config.branch_live_two_phase):
        include_potential = False
    branch_skills = jnp.concatenate(
        (jnp.zeros((1, config.skill_dim), dtype=jnp.float32), selected_skills),
        axis=0,
    ) if include_potential else selected_skills
    branch_count = int(branch_skills.shape[0])
    prefix_config = replace(
        config,
        num_envs=1,
        allow_reference_deviation=True,
    )
    branch_config = replace(config, num_envs=branch_count, allow_reference_deviation=True)
    prefix_env = make_environment(prefix_config)
    branch_env = make_environment(branch_config)
    uses_soccer_dynamics = bool(branch_env.soccer)
    reset_key, prefix_key, branch_key = jax.random.split(
        jax.random.PRNGKey(config.seed + seed_offset), 3
    )
    initial_state = prefix_env.reset(reset_key[None, :])
    initial_anchors = initial_state.obs
    sac_state = training_state.sac
    del training_state

    def snapshot(state: Any) -> dict[str, jax.Array]:
        row = {
            "qpos": state.data.qpos,
            "qvel": state.data.qvel,
            "ctrl": state.data.ctrl,
            "mocap_pos": state.data.mocap_pos,
            "mocap_quat": state.data.mocap_quat,
            "observations": state.obs,
            "agent_positions": state.data.qpos[:, :2],
        }
        if uses_soccer_dynamics:
            row["ball_positions"] = state.data.qpos[:, -7:-5]
        return row

    def prefix_step(carry: tuple[Any, jax.Array, jax.Array], _: None):
        state, anchors, key = carry
        key, action_root = jax.random.split(key)
        actions = experiment.sac.sample_actions(
            sac_state,
            state.obs,
            anchors,
            jnp.zeros((1, config.skill_dim), dtype=jnp.float32),
            action_root[None, :],
            deterministic=deterministic,
        )
        next_state = prefix_env.step(state, actions)
        return (next_state, next_state.obs, key), snapshot(next_state)

    (branch_source_state, branch_source_anchors, _), prefix_scanned = jax.jit(
        lambda state, anchors, key: jax.lax.scan(
            prefix_step,
            (state, anchors, key),
            None,
            length=prefix_steps,
        )
    )(initial_state, initial_anchors, prefix_key)

    branch_template = branch_env.reset(
        jnp.repeat(reset_key[None, :], branch_count, axis=0)
    )
    branch_initial_state = gather_env_state_lanes(
        branch_source_state,
        branch_template,
        jnp.zeros((branch_count,), dtype=jnp.int32),
    )
    branch_initial_anchors = jnp.repeat(branch_source_anchors, branch_count, axis=0)

    def branch_step(carry: tuple[Any, jax.Array, jax.Array], _: None):
        state, anchors, key = carry
        key, action_root = jax.random.split(key)
        actions = experiment.sac.sample_actions(
            sac_state,
            state.obs,
            anchors,
            branch_skills,
            jax.random.split(action_root, branch_count),
            deterministic=deterministic,
        )
        next_state = branch_env.step(state, actions)
        next_anchors = anchors
        return (next_state, next_anchors, key), snapshot(next_state)

    (_, _, _), branch_scanned = jax.jit(
        lambda state, anchors, key: jax.lax.scan(
            branch_step,
            (state, anchors, key),
            None,
            length=horizon_steps - prefix_steps,
        )
    )(branch_initial_state, branch_initial_anchors, branch_key)
    initial = snapshot(initial_state)
    arrays = {
        name: np.swapaxes(
            np.asarray(
                jax.device_get(
                    jnp.concatenate(
                        (
                            jnp.repeat(
                                jnp.concatenate((value[None], prefix_scanned[name]), axis=0),
                                branch_count,
                                axis=1,
                            ),
                            branch_scanned[name],
                        ),
                        axis=0,
                    )
                )
            ),
            0,
            1,
        )
        for name, value in initial.items()
    }
    activation_index = int(prefix_steps)
    prefix_report = _prefix_report(arrays, activation_index)
    if not prefix_report["valid"]:
        raise AssertionError(
            "locomotion branch traces do not share an identical potential prefix: "
            f"{prefix_report['max_abs_delta_by_key']}"
        )

    metadata = {
        "artifact": "locomotion_shared_prefix_branch_trace",
        "source_run": str(Path(run_dir).expanduser().resolve()),
        "checkpoint_step": checkpoint_step,
        "environment": config.environment,
        "model_environment": "ant_soccer" if uses_soccer_dynamics else "ant",
        "encoder_obs_mode": config.encoder_obs_mode,
        "observation_contract": config.locomotion_observation_contract,
        "seed": int(config.seed + seed_offset),
        "policy_mode": "deterministic" if deterministic else "stochastic",
        "potential_continuation_lane": 0 if include_potential else None,
        "skill_indices": list(selected),
        "skill_selection": skill_selection,
        "skill_vectors": np.asarray(selected_skills).tolist(),
        "branch_count": branch_count,
        "prefix_steps": int(prefix_steps),
        "potential_schedule": potential_schedule_metadata(
            config, [prefix_steps], seed_offset, shared=True, prefix_steps_override=prefix_override,
        ),
        "activation_index": activation_index,
        "horizon_steps": int(horizon_steps),
        "control_dt": float(branch_env.config.ctrl_dt),
        "branch_prefix_invariant": prefix_report,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **arrays,
        skills=np.asarray(jax.device_get(branch_skills)),
        potential_switch_steps=np.full(branch_count, prefix_steps, dtype=np.int32),
        skill_indices=np.asarray((-1, *selected) if include_potential else selected, dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return destination


def _native_locomotion_model(model_environment: str) -> Any:
    from empowerment_warp.envs.locomotion.config import AntConfig, AntSoccerConfig
    from empowerment_warp.envs.locomotion.model import build_ant_model

    if model_environment == "ant_soccer":
        return build_ant_model(AntSoccerConfig(num_envs=1), soccer=True)
    if model_environment == "ant":
        return build_ant_model(AntConfig(num_envs=1), soccer=False)
    raise ValueError(f"unsupported locomotion model environment: {model_environment}")


class _IncumbentAntRenderEnv:
    """Minimal native-MuJoCo state holder required by the bundled overlay renderer."""

    def __init__(self, model: Any, qpos: np.ndarray, qvel: np.ndarray):
        import mujoco

        self.model = model
        self.data = mujoco.MjData(model)
        self.width = 640
        self.height = 480
        if model.nq == 22 and model.nv == 20:
            camera = mujoco.MjvCamera()
            camera.lookat[0] = 10.0
            camera.lookat[1] = 10.0
            camera.distance = 30.0
            camera.elevation = -90.0
            self.custom_camera = camera
        else:
            self.custom_camera = -1
        self.custom_renderer = None
        self.set_state(qpos, qvel)

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        import mujoco

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def render(self) -> np.ndarray:
        raise RuntimeError("the bundled overlay renderer owns its MuJoCo renderer")


def _load_incumbent_locomotion_renderer() -> Any:
    from empowerment_warp._vendor.rendering import trajectory_videos
    return trajectory_videos


def render_locomotion_branch_trace(
    trace_path: str | Path,
    output_dir: str | Path,
    *,
    fps: int = BRANCH_FPS,
    publication_chrome: bool = False,
    frame_stride: int = BRANCH_FRAME_STRIDE,
    min_render_size: int | None = None,
    still_frame_index: int | None = None,
) -> dict[str, Any]:
    """Render the Warp trace with the AntSoccer ghost overlay."""

    source = Path(trace_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        skills = np.asarray(archive["skills"])
        skill_indices = np.asarray(archive["skill_indices"])
        arrays = {
            name: np.asarray(archive[name])
            for name in ("qpos", "qvel", "agent_positions", "ball_positions")
        }

    activation_index = int(metadata["activation_index"])
    prefix_report = _prefix_report(arrays, activation_index)
    if not prefix_report["valid"]:
        raise AssertionError(f"branch prefix invariant failed before rendering: {prefix_report}")

    traces = [
        {
            "qpos_seq": arrays["qpos"][lane],
            "positions": arrays["agent_positions"][lane],
            "ball_positions": arrays["ball_positions"][lane],
            "skill": skills[lane],
            "skill_index": int(skill_indices[lane]),
            "activation_index": activation_index,
        }
        for lane in range(len(skill_indices))
    ]
    pe_trace = traces[0]
    include_potential = metadata.get("potential_continuation_lane", 0) is not None
    branch_traces = traces[1:] if include_potential else traces
    module = _load_incumbent_locomotion_renderer()
    render_env = _IncumbentAntRenderEnv(
        _native_locomotion_model(metadata.get("model_environment", "ant_soccer")),
        pe_trace["qpos_seq"][0],
        arrays["qvel"][0, 0],
    )
    ghost_renderer = module._AntGhostRenderer(
        render_env,
        trail_steps=BRANCH_TRAIL_STEPS,
        trail_stride=BRANCH_TRAIL_STRIDE,
    )
    overlay_rgbs = [
        np.asarray(module.skill_to_rgb01(trace["skill"]), dtype=np.float32)
        for trace in branch_traces
    ]
    skill_colors = [
        np.array([*rgb, module.ANT_BODY_ALPHA], dtype=np.float32) for rgb in overlay_rgbs
    ]
    skill_trail_colors = [
        np.array([*rgb, module.ANT_TRAIL_ALPHA], dtype=np.float32) for rgb in overlay_rgbs
    ]
    ghost_ball_colors = [
        np.array([*rgb, module.BALL_GHOST_ALPHA], dtype=np.float32) for rgb in overlay_rgbs
    ]
    gray_body = np.array([*module.GRAY_RGB, module.ANT_BODY_ALPHA_GRAY], dtype=np.float32)
    gray_trail = np.array([*module.GRAY_RGB, module.ANT_TRAIL_ALPHA_GRAY], dtype=np.float32)
    gray_ball = np.array([*module.GRAY_RGB, module.BALL_GHOST_ALPHA], dtype=np.float32)
    solid_ball = np.array([*module.BALL_SOLID_RGB, module.BALL_SOLID_ALPHA], dtype=np.float32)
    marker = np.array([0.98, 0.97, 0.95, 0.94], dtype=np.float32)
    encoder_obs_mode = metadata.get(
        "encoder_obs_mode",
        "ball_xy" if metadata["environment"] == "ant_soccer" else "full",
    )
    use_ball_trails = encoder_obs_mode == "ball_xy"
    shared_positions = (
        pe_trace["ball_positions"] if use_ball_trails else pe_trace["positions"]
    )
    ball_split_index = module._first_ball_split_index(branch_traces + [pe_trace])
    frame_count = min(len(trace["qpos_seq"]) for trace in traces)

    def trail_positions(trace: dict[str, Any]) -> np.ndarray:
        return trace["ball_positions"] if use_ball_trails else trace["positions"]

    def build_branch_scene(scene: Any, frame_index: int) -> None:
        prefix_index = min(frame_index, activation_index, len(shared_positions) - 1)
        ghost_renderer.add_trail(scene, shared_positions, prefix_index, gray_trail)
        if frame_index <= activation_index:
            ghost_renderer.add_ant(scene, pe_trace["qpos_seq"][frame_index], gray_body)
        else:
            ghost_renderer.add_branch_marker(
                scene,
                shared_positions[min(activation_index, len(shared_positions) - 1)],
                marker,
            )
            if include_potential:
                ghost_renderer.add_trail(
                    scene,
                    trail_positions(pe_trace),
                    frame_index,
                    gray_trail,
                    start_idx=activation_index,
                )
            for trace, trail_color in zip(branch_traces, skill_trail_colors, strict=True):
                ghost_renderer.add_trail(
                    scene,
                    trail_positions(trace),
                    frame_index,
                    trail_color,
                    start_idx=activation_index,
                )
            for trace, body_color in zip(branch_traces, skill_colors, strict=True):
                ghost_renderer.add_ant(scene, trace["qpos_seq"][frame_index], body_color)
            if include_potential:
                ghost_renderer.add_ant(scene, pe_trace["qpos_seq"][frame_index], gray_body)
        if ball_split_index is not None and frame_index >= ball_split_index:
            if include_potential:
                ghost_renderer.add_ball(scene, pe_trace["qpos_seq"][frame_index], gray_ball)
            for trace, ball_color in zip(branch_traces, ghost_ball_colors, strict=True):
                ghost_renderer.add_ball(scene, trace["qpos_seq"][frame_index], ball_color)
        else:
            ghost_renderer.add_ball(scene, branch_traces[0]["qpos_seq"][frame_index], solid_ball)

    if still_frame_index is not None:
        if not 0 < still_frame_index < frame_count:
            raise ValueError(f"still_frame_index must be in [1, {frame_count - 1}]")
        with ghost_renderer.empowerment_background(), ghost_renderer.hidden_base_geoms():
            still_frames = module._render_video_frames(
                base_env=render_env,
                scene_renderer=ghost_renderer,
                frame_count=still_frame_index + 1,
                scene_builder=build_branch_scene,
                frame_stride=still_frame_index,
                min_render_size=min_render_size,
                light_qpos_seq=None,
                render_workers=1,
            )
        still_path = destination / f"branch_ant_frame_{still_frame_index}.png"
        module.Image.fromarray(still_frames[-1]).save(still_path)
        return {"source_trace": str(source), "still_frame_index": still_frame_index,
                "still": str(still_path), "render_size": min_render_size}

    with ghost_renderer.empowerment_background(), ghost_renderer.hidden_base_geoms():
        frames = module._render_video_frames(
            base_env=render_env,
            scene_renderer=ghost_renderer,
            frame_count=frame_count,
            scene_builder=build_branch_scene,
            frame_stride=frame_stride,
            min_render_size=(module.ANT_OVERLAY_RENDER_MIN_SIZE if min_render_size is None else min_render_size),
            light_qpos_seq=None,
            render_workers=1,
        )
    # Publication sizing is explicit; normal evaluation keeps its existing defaults.
    frames = module._overlay_existing_renderer_legend(
        frames,
        overlay_rgbs,
        legend_style="builderbench_gray",
        include_potential_policy=include_potential,
        publication=publication_chrome,
    )
    rendered_indices = module._rendered_frame_indices(frame_count, frame_stride)
    if not publication_chrome:
        frames = module.add_video_label_band(
            frames,
            lambda frame_index: module._branch_rollout_label(
                module._rendered_raw_frame_index(rendered_indices, frame_index),
                activation_index,
                len(branch_traces),
            ) if include_potential else f"policy: skill | targets: {len(branch_traces)}",
        )
    overlay_metadata = {
        **metadata,
        "branch_prefix_invariant": prefix_report,
        "branch_frame_count": int(frame_count),
        "branch_activation_index": activation_index,
        "skill_indices": (skill_indices[1:] if include_potential else skill_indices).tolist(),
        "use_ball_trails": bool(use_ball_trails),
        "branch_ball_split_index": None if ball_split_index is None else int(ball_split_index),
        "frame_stride": frame_stride,
        "publication_chrome": publication_chrome,
        "fps": int(fps),
        "incumbent_source": str(incumbent_collector()),
        "incumbent_source_sha256": _sha256(incumbent_collector()),
        "incumbent_renderer_primitives_reused_directly": True,
    }
    bundle = {"saved_paths": {}}
    module._save_env_overlay_bundle(
        bundle=bundle,
        output_dir=str(destination),
        stem="branch_ant_overlay",
        saved_key="branch_ant_video",
        frames=frames,
        fps=int(fps),
        artifact_role="ogbench_branch_scene_overlay",
        metadata=overlay_metadata,
        require_motion=False,
    )
    branch_point_still = destination / "branch_ant_branch_point.png"
    rollout_end_still = destination / "branch_ant_rollout_end.png"
    if activation_index not in rendered_indices:
        raise ValueError("Frame stride omitted the branch-point still.")
    module.Image.fromarray(frames[list(rendered_indices).index(activation_index)]).save(
        branch_point_still
    )
    module.Image.fromarray(frames[-1]).save(rollout_end_still)
    video_paths = [
        str(value) for value in bundle["saved_paths"].values() if str(value).endswith(".mp4")
    ]
    if len(video_paths) != 1:
        raise RuntimeError(f"expected one branch MP4, found {video_paths}")
    result = {
        "source_trace": str(source),
        "source_run": metadata["source_run"],
        "checkpoint_step": metadata["checkpoint_step"],
        "environment": metadata["environment"],
        "fps": int(fps),
        "branch_prefix_invariant": prefix_report,
        "video": video_paths[0],
        "stills": [str(branch_point_still), str(rollout_end_still)],
        "saved_paths": {key: str(value) for key, value in bundle["saved_paths"].items()},
        "metadata": overlay_metadata,
    }
    result_path = destination / "locomotion_branch_media.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def upload_locomotion_branch_media(
    run_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Best-effort W&B upload of the branch overlay."""

    run_path = Path(run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    result_path = output / "locomotion_branch_media.json"
    result = json.loads(result_path.read_text())
    config = json.loads((run_path / "config.json").read_text())
    mirror = WandbMirror.start(run_path, config, resume=True)
    if mirror.run is None:
        record_pending_upload(
            run_path,
            {
                "kind": "locomotion_branch_media",
                "source_metadata": str(result_path),
                "video": result["video"],
                "wandb_key": "media/trajectory_ant_overlay_video",
                "error": json.loads((run_path / "wandb_status.json").read_text()).get("error"),
            },
        )
        raise RuntimeError(
            f"W&B media upload unavailable: {(run_path / 'wandb_status.json').read_text()}"
        )
    wandb = import_module("wandb")
    last_row = json.loads((run_path / "metrics.jsonl").read_text().splitlines()[-1])
    # W&B drops a history row logged at a step <= the run's last history step when a finished run
    # is resumed (the files still upload, so nothing shows a panel). Log strictly past it.
    try:
        last_history_step = int(mirror.run.summary.get("_step", 0) or 0)
    except Exception:
        last_history_step = 0
    media_step = max(int(last_row["env_steps"]), last_history_step) + 1
    media_row = {
        "env_steps": float(media_step),
        "media/trajectory_ant_overlay_video": wandb.Video(
            result["video"], fps=int(result["fps"]), format="mp4"
        ),
    }
    # The empowerment-along-the-episode figures (evaluate-potential-trace + potential-trace-figure):
    # E(s_t) vs step and the skill paths, uploaded alongside the overlay video whenever they
    # exist for this run.
    field_dir = run_path / "media" / "field"
    for stem, key in (
        ("potential_trace_value", "media/empowerment_along_episode"),
        ("potential_trace_path", "media/empowerment_skill_paths"),
    ):
        png = field_dir / f"{stem}.png"
        if png.exists():
            media_row[key] = wandb.Image(str(png))
    skill_paths = output / "skill_path.png"
    if skill_paths.exists():
        media_row["media/empowerment_skill_paths"] = wandb.Image(str(skill_paths))
    mirror.log(media_row)
    mirror.finish()
    upload = {
        "wandb": json.loads((run_path / "wandb_run.json").read_text()),
        "env_steps": float(media_step),
        "keys": sorted(key for key in media_row if key.startswith("media/")),
        "source_metadata": str(result_path),
    }
    (output / "wandb_media.json").write_text(
        json.dumps(upload, indent=2, sort_keys=True) + "\n"
    )
    return upload


def create_locomotion_branch_media(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    prefix_steps: int | None = None,
    horizon_steps: int = BRANCH_HORIZON_STEPS,
    skill_count: int = BRANCH_SKILL_COUNT,
    seed_offset: int = 50_000,
    fps: int = BRANCH_FPS,
    upload_wandb: bool = True,
    deterministic: bool = False,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "branch_trace.npz"
    if trace_path.exists():
        with np.load(trace_path, allow_pickle=False) as archive:
            cached = json.loads(str(archive["metadata"]))
        requested_mode = "deterministic" if deterministic else "stochastic"
        if cached.get("policy_mode") != requested_mode:
            raise ValueError(
                f"cached trace policy_mode={cached.get('policy_mode')} differs from "
                f"requested {requested_mode}; use a separate output directory"
            )
        if checkpoint is not None and cached["checkpoint_step"] != int(Path(checkpoint).name.removeprefix("step_")):
            raise ValueError("cached branch trace uses a different checkpoint")
        config = json.loads((Path(run_dir) / "config.json").read_text())
        if config["skill_dim"] == 2:
            if cached.get("skill_selection") != "uniform_circle" or len(
                cached.get("skill_vectors", [])
            ) != skill_count:
                raise ValueError(
                    "cached trace does not use the requested circle of video skills; "
                    "use a separate output directory"
                )
        schedule = cached.get("potential_schedule", {})
        random_max = int(config.get("random_potential_switch_max_steps", 0))
        nominal_prefix = int(config.get("potential_steps", 0))
        expected_seed = int(config.get("seed", 0)) + seed_offset
        timing_matches = (
            schedule.get("random_potential_switch_max_steps", 0) == random_max
            and schedule.get("nominal_potential_steps", cached.get("prefix_steps")) == nominal_prefix
            and schedule.get("prefix_steps_override") == prefix_steps
            and cached.get("seed") == expected_seed
            and cached.get("horizon_steps") == horizon_steps
        )
        # Traces without schedule metadata are reusable only at their recorded fixed default.
        if random_max > 0:
            timing_matches = timing_matches and schedule.get("switch_seed") == expected_seed + 1_000_000
        if prefix_steps is not None:
            timing_matches = timing_matches and cached.get("prefix_steps") == prefix_steps
        if not timing_matches:
            raise ValueError(
                "cached trace uses a different potential schedule, prefix override, seed, or horizon; "
                "use a separate output directory"
            )
    else:
        collect_locomotion_branch_trace(
            run_dir,
            trace_path,
            prefix_steps=prefix_steps,
            horizon_steps=horizon_steps,
            skill_count=skill_count,
            seed_offset=seed_offset,
            checkpoint=checkpoint,
            deterministic=deterministic,
        )
    from empowerment_warp.plotting.potential_trace import render_branch_skill_paths

    render_branch_skill_paths(trace_path, output / "skill")
    rendered = render_locomotion_branch_trace(trace_path, output, fps=fps)
    uploaded = upload_locomotion_branch_media(run_dir, output) if upload_wandb else None
    result = {"trace": str(trace_path), "rendered": rendered, "wandb": uploaded}
    (output / "branch_media_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result
