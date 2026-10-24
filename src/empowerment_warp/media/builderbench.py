"""BuilderBench shared-prefix branch media."""

from __future__ import annotations

import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.training.sac.config import ExperimentConfig
from empowerment_warp.training.sac.evaluation import eval_skill_library, evaluation_switch_steps
from empowerment_warp.training.sac.evaluation import load_run
from empowerment_warp.envs.state import gather_env_state_lanes
from empowerment_warp.envs.factory import make_environment
from empowerment_warp.tracking import WandbMirror, record_pending_upload


BRANCH_PREFIX_STEPS = 100
BRANCH_HORIZON_STEPS = 400
BRANCH_SKILL_COUNT = 4
BRANCH_FRAME_STRIDE = 3
BRANCH_FPS = 14
BRANCH_TRAIL_STEPS = 90
BRANCH_TRAIL_STRIDE = 2
BRANCH_PREFIX_TOL = 1e-6

def incumbent_collector() -> Path:
    from empowerment_warp._vendor import rendering
    return Path(rendering.__file__).parent / "trajectory_videos.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced_skill_indices(num_skills: int, count: int) -> tuple[int, ...]:
    """Return `count` evenly spaced skill indices, including the first and last."""

    if num_skills <= 0 or count <= 0:
        raise ValueError("num_skills and count must be positive")
    count = min(int(count), int(num_skills))
    if count == num_skills:
        return tuple(range(num_skills))
    if count == 1:
        return (0,)
    return tuple(
        int(round(index * (num_skills - 1) / (count - 1))) for index in range(count)
    )


def _prefix_report(arrays: dict[str, np.ndarray], activation_index: int) -> dict[str, Any]:
    prefix_len = int(activation_index) + 1
    report: dict[str, Any] = {
        "checked": True,
        "valid": True,
        "prefix_frame_count": prefix_len,
        "max_abs_delta_by_key": {},
    }
    for name in ("qpos", "qvel", "effector_positions", "cube_positions", "mocap_pos", "mocap_quat"):
        values = np.asarray(arrays[name])[:, :prefix_len]
        delta = np.abs(values.astype(np.float64) - values[0:1].astype(np.float64))
        maximum = float(np.max(delta)) if delta.size else 0.0
        report["max_abs_delta_by_key"][name] = maximum
        report["valid"] = bool(report["valid"] and maximum <= BRANCH_PREFIX_TOL)
    return report


def collect_builderbench_branch_trace(
    run_dir: str | Path,
    output: str | Path,
    *,
    prefix_steps: int | None = None,
    horizon_steps: int = BRANCH_HORIZON_STEPS,
    skill_count: int = BRANCH_SKILL_COUNT,
    seed_offset: int = 50_000,
    sample_potential: bool = False,
    policy_mode: str = "stochastic",
    checkpoint: str | Path | None = None,
) -> Path:
    """Collect skill branches after the run's configured potential prefix."""

    config, experiment, training_state = (load_run(run_dir) if checkpoint is None
                                               else load_run(run_dir, checkpoint=checkpoint))
    checkpoint_step = int(training_state.env_steps)
    prefix_steps = int(evaluation_switch_steps(config, 1, seed_offset)[0]) if prefix_steps is None else int(prefix_steps)
    if not 0 <= prefix_steps < horizon_steps:
        raise ValueError("prefix_steps must be nonnegative and less than horizon_steps")
    if policy_mode not in {"deterministic", "stochastic"}:
        raise ValueError("policy_mode must be 'deterministic' or 'stochastic'")
    deterministic = policy_mode == "deterministic"
    if config.environment != "builderbench_corner3":
        raise ValueError("BuilderBench branch media requires a builderbench_corner3 run")
    # Continuous (sym-CSF) runs have no library; sweep the seeded probe directions instead.
    full_library = eval_skill_library(config)

    skill_indices = evenly_spaced_skill_indices(full_library.shape[0], skill_count)
    skill_library = full_library[jnp.asarray(skill_indices)]
    include_potential = prefix_steps > 0 or config.potential_lane_fraction > 0
    branch_skills = (
        jnp.concatenate(
            (jnp.zeros((1, config.skill_dim), dtype=jnp.float32), skill_library), axis=0
        )
        if include_potential
        else skill_library
    )
    branch_count = int(branch_skills.shape[0])
    prefix_config = ExperimentConfig.from_dict(
        {
            **config.to_dict(),
            "num_envs": 1,
            "allow_reference_deviation": True,
        }
    )
    branch_config = ExperimentConfig.from_dict(
        {
            **config.to_dict(),
            "num_envs": branch_count,
            "allow_reference_deviation": True,
        }
    )
    prefix_env = make_environment(prefix_config)
    branch_env = make_environment(branch_config)
    reset_key, prefix_key, branch_key = jax.random.split(
        jax.random.PRNGKey(config.seed + seed_offset), 3
    )
    initial_state = prefix_env.reset(reset_key[None, :])
    initial_anchors = initial_state.obs
    sac_state = training_state.sac
    del training_state

    def snapshot(state: Any, env: Any) -> dict[str, jax.Array]:
        return {
            "qpos": state.data.qpos,
            "qvel": state.data.qvel,
            "ctrl": state.data.ctrl,
            "mocap_pos": state.data.mocap_pos,
            "mocap_quat": state.data.mocap_quat,
            "observations": state.obs,
            "effector_positions": state.obs[:, :3],
            "cube_positions": jax.vmap(env._object_positions)(state.data),
            "terminated": state.terminated,
            "truncated": state.truncated,
        }

    def prefix_step(carry: tuple[Any, jax.Array, jax.Array], _: None):
        state, anchors, key = carry
        key, action_root = jax.random.split(key)
        skills = jnp.zeros((1, config.skill_dim), dtype=jnp.float32)
        actions = experiment.sac.sample_actions(
            sac_state,
            state.obs,
            anchors,
            skills,
            action_root[None, :],
            deterministic=deterministic and not sample_potential,
        )
        next_state = prefix_env.step(state, actions)
        return (next_state, next_state.obs, key), snapshot(next_state, prefix_env)

    (branch_source_state, branch_source_anchors, _), prefix_scanned = jax.jit(
        lambda state, anchors, key: jax.lax.scan(
            prefix_step,
            (state, anchors, key),
            None,
            length=prefix_steps,
        )
    )(initial_state, initial_anchors, prefix_key)

    branch_template = branch_env.reset(jnp.repeat(reset_key[None, :], branch_count, axis=0))
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
        zero_skill = jnp.linalg.norm(branch_skills, axis=-1) <= 1e-6
        next_anchors = jnp.where(zero_skill[:, None], next_state.obs, anchors)
        return (next_state, next_anchors, key), snapshot(next_state, branch_env)

    (_, _, _), branch_scanned = jax.jit(
        lambda state, anchors, key: jax.lax.scan(
            branch_step,
            (state, anchors, key),
            None,
            length=horizon_steps - prefix_steps,
        )
    )(branch_initial_state, branch_initial_anchors, branch_key)

    initial = snapshot(initial_state, prefix_env)
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
            "BuilderBench branch traces do not share an identical potential prefix: "
            f"{prefix_report['max_abs_delta_by_key']}"
        )

    metadata = {
        "artifact": (
            "builderbench_shared_prefix_branch_trace"
            if include_potential
            else "builderbench_skill_from_reset_branch_trace"
        ),
        "source_run": str(Path(run_dir).expanduser().resolve()),
        "checkpoint_step": checkpoint_step,
        "seed": int(config.seed + seed_offset),
        "policy_mode": policy_mode,
        "potential_policy_mode": (
            ("sampled" if sample_potential or not deterministic else "deterministic")
            if include_potential
            else None
        ),
        "potential_continuation_lane": 0 if include_potential else None,
        "skill_indices": list(skill_indices),
        "branch_count": branch_count,
        "prefix_steps": int(prefix_steps),
        "activation_index": activation_index,
        "horizon_steps": int(horizon_steps),
        "control_dt": float(branch_env.config.ctrl_dt),
        "branch_prefix_invariant": prefix_report,
        "incumbent_source": str(incumbent_collector()),
        "incumbent_source_sha256": _sha256(incumbent_collector()),
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **arrays,
        skills=np.asarray(jax.device_get(branch_skills)),
        skill_indices=np.asarray(
            (-1, *skill_indices) if include_potential else skill_indices, dtype=np.int32
        ),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return destination


class _IncumbentBuilderBenchRenderEnv:
    """Minimal state holder expected by the bundled overlay renderer."""

    def __init__(self, model_bundle: Any, initial_trace: dict[str, np.ndarray]):
        import mujoco

        self.model = model_bundle.model
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = initial_trace["qpos_seq"][0]
        self.data.qvel[:] = initial_trace["qvel_seq"][0]
        self.data.mocap_pos[model_bundle.mocap_ids] = initial_trace["mocap_pos_seq"][0]
        self.data.mocap_quat[model_bundle.mocap_ids] = initial_trace["mocap_quat_seq"][0]
        mujoco.mj_forward(self.model, self.data)
        self.width = 640
        self.height = 480
        self.custom_camera = -1
        self._task_mocap_targets = np.asarray(model_bundle.mocap_ids, dtype=np.int32)
        self._workspace_bounds = np.asarray(model_bundle.workspace_bounds, dtype=np.float32)

    def render(self) -> np.ndarray:
        raise RuntimeError("the bundled overlay renderer owns its MuJoCo renderer")


def _load_incumbent_builderbench_renderer():
    """Load the bundled BuilderBench overlay renderer."""

    from empowerment_warp._vendor.rendering.trajectory_videos import create_builderbench_scene_video_bundle
    return create_builderbench_scene_video_bundle


def _incumbent_trace(
    arrays: dict[str, np.ndarray],
    skills: np.ndarray,
    skill_indices: np.ndarray,
    lane: int,
    activation_index: int,
) -> dict[str, Any]:
    """Rename Warp trace fields into the bundled renderer's trace contract."""

    return {
        "qpos_seq": arrays["qpos"][lane],
        "qvel_seq": arrays["qvel"][lane],
        "mocap_pos_seq": arrays["mocap_pos"][lane],
        "mocap_quat_seq": arrays["mocap_quat"][lane],
        "effector_positions": arrays["effector_positions"][lane],
        "cube_positions": arrays["cube_positions"][lane],
        "positions": arrays["effector_positions"][lane],
        "skill": skills[lane],
        "skill_index": int(skill_indices[lane]),
        "activation_index": int(activation_index),
    }


def render_builderbench_branch_trace(
    trace_path: str | Path,
    output_dir: str | Path,
    *,
    frame_stride: int = BRANCH_FRAME_STRIDE,
    fps: int = BRANCH_FPS,
    publication_chrome: bool = False,
    min_render_size: int | None = None,
    still_frame_index: int | None = None,
) -> dict[str, Any]:
    """Render through empowerment's unchanged BuilderBench branching function."""

    os.environ.setdefault("MUJOCO_GL", "egl")
    source = Path(trace_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as archive:
        trace_metadata = json.loads(str(archive["metadata"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in (
                "qpos",
                "qvel",
                "mocap_pos",
                "mocap_quat",
                "effector_positions",
                "cube_positions",
            )
        }
        skills = np.asarray(archive["skills"], dtype=np.float32)
        skill_indices = np.asarray(archive["skill_indices"], dtype=np.int32)

    activation_index = int(trace_metadata["activation_index"])
    prefix_report = _prefix_report(arrays, activation_index)
    if not prefix_report["valid"]:
        raise AssertionError(f"branch prefix invariant failed before rendering: {prefix_report}")

    traces = [
        _incumbent_trace(arrays, skills, skill_indices, lane, activation_index)
        for lane in range(len(skill_indices))
    ]
    from empowerment_warp.envs.builderbench.config import BuilderBenchConfig
    from empowerment_warp.envs.builderbench.model import build_model

    model_bundle = build_model(BuilderBenchConfig(num_envs=1))
    render_env = _IncumbentBuilderBenchRenderEnv(model_bundle, traces[0])
    incumbent_renderer = _load_incumbent_builderbench_renderer()
    include_potential = trace_metadata.get("potential_continuation_lane") is not None
    frame_count = min(len(trace["qpos_seq"]) for trace in traces)
    if still_frame_index is not None:
        if not 0 < still_frame_index < frame_count:
            raise ValueError(f"still_frame_index must be in [1, {frame_count - 1}]")
        frame_stride = still_frame_index
    bundle = incumbent_renderer(
        env=render_env,
        skill_only_traces=[],
        # The bundled renderer names its rendered branch slot `two_phase_traces`,
        # but it accepts activation_index=0 skill-from-reset traces there as well.
        two_phase_traces=traces[1:] if include_potential else traces,
        pe_only_traces=[traces[0]] if include_potential else [],
        output_dirs=[] if still_frame_index is not None else [str(destination)],
        fps=int(fps),
        trail_steps=BRANCH_TRAIL_STEPS,
        trail_stride=BRANCH_TRAIL_STRIDE,
        frame_stride=int(frame_stride),
        render_skill=False,
        preserve_skill_bank=False,
        cyclic_skill_colors=False,
        publication_chrome=publication_chrome,
        min_render_size=min_render_size,
    )
    rendered_metadata = dict(bundle["builderbench_overlay_metadata"])
    rendered_metadata.update(
        {
            "source_run": trace_metadata["source_run"],
            "checkpoint_step": trace_metadata["checkpoint_step"],
            "skill_indices": (
                skill_indices[1:].tolist() if include_potential else skill_indices.tolist()
            ),
            "policy_mode": trace_metadata["policy_mode"],
            "potential_continuation_lane": trace_metadata["potential_continuation_lane"],
            "requested_fps": int(fps),
            "incumbent_source": str(incumbent_collector()),
            "incumbent_source_sha256": _sha256(incumbent_collector()),
            "renderer_reused_directly": True,
        }
    )
    metadata_path = destination / "builderbench_overlay_metadata.json"
    metadata_path.write_text(json.dumps(rendered_metadata, indent=2, sort_keys=True) + "\n")
    saved_paths = {key: str(value) for key, value in bundle["saved_paths"].items()}
    if still_frame_index is not None:
        from PIL import Image

        rendered_indices = list(range(0, frame_count, frame_stride))
        if rendered_indices[-1] != frame_count - 1:
            rendered_indices.append(frame_count - 1)
        still_path = destination / f"branch_arm_frame_{still_frame_index}.png"
        Image.fromarray(bundle["branch_arm_video_frames"][rendered_indices.index(still_frame_index)]).save(still_path)
        saved_paths["publication_still"] = str(still_path)
    return {
        "metadata": rendered_metadata,
        "metadata_path": str(metadata_path),
        "saved_paths": saved_paths,
    }


def upload_builderbench_branch_media(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Attach the copied branch media to the run's W&B panel key."""

    run_path = Path(run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((output / "builderbench_overlay_metadata.json").read_text())
    config = json.loads((run_path / "config.json").read_text())
    mirror = WandbMirror.start(run_path, config, resume=(run_path / "wandb_run.json").exists())
    if mirror.run is None:
        record_pending_upload(
            run_path,
            {
                "kind": "builderbench_branch_media",
                "source_metadata": str(output / "builderbench_overlay_metadata.json"),
                "video": str(output / "branch_arm_overlay.mp4"),
                "wandb_key": "media/trajectory_arm_overlay_video",
                "error": json.loads((run_path / "wandb_status.json").read_text()).get("error"),
            },
        )
        raise RuntimeError(f"W&B media upload unavailable: {(run_path / 'wandb_status.json').read_text()}")
    wandb = import_module("wandb")
    last_row = json.loads((run_path / "metrics.jsonl").read_text().splitlines()[-1])
    media_row = {
        "env_steps": float(last_row["env_steps"]),
        "media/trajectory_arm_overlay_video": wandb.Video(
            str(output / "branch_arm_overlay.mp4"), fps=int(metadata["fps"]), format="mp4"
        ),
    }
    mirror.log(media_row)
    mirror.finish()
    identity = json.loads((run_path / "wandb_run.json").read_text())
    upload = {
        "wandb": identity,
        "env_steps": float(last_row["env_steps"]),
        "keys": sorted(key for key in media_row if key.startswith("media/")),
        "source_metadata": str(output / "builderbench_overlay_metadata.json"),
    }
    (output / "wandb_media.json").write_text(json.dumps(upload, indent=2, sort_keys=True) + "\n")
    return upload


def create_builderbench_branch_media(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    prefix_steps: int | None = None,
    horizon_steps: int = BRANCH_HORIZON_STEPS,
    skill_count: int = BRANCH_SKILL_COUNT,
    seed_offset: int = 50_000,
    frame_stride: int = BRANCH_FRAME_STRIDE,
    fps: int = BRANCH_FPS,
    upload_wandb: bool = True,
    policy_mode: str = "stochastic",
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    if policy_mode not in {"deterministic", "stochastic"}:
        raise ValueError("policy_mode must be 'deterministic' or 'stochastic'")
    if prefix_steps is None:
        saved_config = json.loads(
            (Path(run_dir).expanduser().resolve() / "config.json").read_text()
        )
        prefix_steps = int(saved_config["potential_steps"])
        if (saved_config.get("skill_start_mode") == "marginal_branch"
                and not saved_config.get("live_potential_only", False)
                and not saved_config.get("branch_live_two_phase", False)):
            prefix_steps = 0
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "branch_trace.npz"
    if trace_path.exists():
        with np.load(trace_path, allow_pickle=False) as archive:
            existing = json.loads(str(archive["metadata"]))
        if checkpoint is not None and existing["checkpoint_step"] != int(Path(checkpoint).name.removeprefix("step_")):
            raise ValueError("cached branch trace uses a different checkpoint")
        expected = {
            "source_run": str(Path(run_dir).expanduser().resolve()),
            "prefix_steps": int(prefix_steps),
            "horizon_steps": int(horizon_steps),
            "branch_count": int(skill_count) + int(prefix_steps > 0),
            "policy_mode": policy_mode,
        }
        mismatches = {
            name: {"expected": value, "actual": existing.get(name)}
            for name, value in expected.items()
            if existing.get(name) != value
        }
        if mismatches:
            raise ValueError(f"existing branch trace does not match requested media: {mismatches}")
    else:
        trace_path = collect_builderbench_branch_trace(
            run_dir,
            trace_path,
            prefix_steps=prefix_steps,
            horizon_steps=horizon_steps,
            skill_count=skill_count,
            seed_offset=seed_offset,
            checkpoint=checkpoint,
            policy_mode=policy_mode,
        )
    rendered = render_builderbench_branch_trace(
        trace_path,
        output,
        frame_stride=frame_stride,
        fps=fps,
    )
    uploaded = upload_builderbench_branch_media(run_dir, output) if upload_wandb else None
    result = {
        "trace": str(trace_path),
        "rendered": rendered,
        "wandb": uploaded,
        "policy_mode": policy_mode,
        "prefix_steps": int(prefix_steps),
    }
    (output / "branch_media_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
