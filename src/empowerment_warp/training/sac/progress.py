"""Checkpoint-consistent evaluation and media snapshots during training."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import shutil
import subprocess
import sys
import time

from empowerment_warp.training.sac.config import ExperimentConfig


def append_evaluation_metrics(run_dir: str | Path, row: dict[str, float]) -> None:
    """Keep scalar evaluation history beside train metrics, before W&B upload."""
    path = Path(run_dir) / "eval.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    fields = ["env_steps"] + sorted(set().union(*(r.keys() for r in rows)) - {"env_steps"})
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def log_progress_snapshot(run_dir, mirror, manifest):
    """Persist scalars first; a failed upload never discards local evaluation."""
    from empowerment_warp.tracking import record_pending_upload
    row = progress_wandb_row(manifest)
    append_evaluation_metrics(run_dir, row)
    uploaded = mirror.log_video(row, key=str(manifest["video_key"]),
                                path=str(manifest["video"]), fps=int(manifest["video_fps"]))
    if not uploaded:
        record_pending_upload(run_dir, {
            "kind": "progress_branch_media",
            "source_metadata": str(Path(run_dir) / "progress" /
                                   f"target_{int(manifest['target_env_steps']):012d}" / "progress_snapshot.json"),
            "video": str(manifest["video"]), "wandb_key": str(manifest["video_key"]),
            "env_steps": float(manifest["checkpoint_step"]),
        })
    if not mirror.log_video({"env_steps": row["env_steps"]},
                            key=manifest["branch_video_key"],
                            path=manifest["branch_video"], fps=manifest["branch_video_fps"]):
        record_pending_upload(run_dir, {
            "kind": "progress_branch_media", "video": manifest["branch_video"],
            "wandb_key": manifest["branch_video_key"], "env_steps": row["env_steps"],
        })


def due_progress_targets(
    current_env_steps: int,
    total_env_steps: int,
    cadence: int,
) -> tuple[int, ...]:
    """Return reached periodic targets and the final training-budget target."""

    upper = min(current_env_steps, total_env_steps)
    targets = tuple(range(cadence, upper + 1, cadence))
    if current_env_steps >= total_env_steps and total_env_steps not in targets:
        targets += (total_env_steps,)
    return targets


def _commands(
    run_dir: Path,
    output_dir: Path,
    environment: str,
) -> tuple[tuple[str, ...], ...]:
    executable = (sys.executable, "-m", "empowerment_warp.cli")
    if environment not in {'ant', 'ant_soccer', 'builderbench_corner3'}:
        raise ValueError(f"progress evaluation is unsupported for {environment}")
    return (
        (*executable, 'evaluate', str(run_dir), '--output', str(output_dir / 'evaluation'),
         '--target-step', output_dir.name.removeprefix('target_')),
        (*executable, 'branch-media', str(run_dir), '--output', str(output_dir / 'media'),
         '--policy-mode', 'stochastic'),
    )



def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def create_progress_snapshot(
    run_dir: str | Path,
    config: ExperimentConfig,
    *,
    target_env_steps: int,
    checkpoint_step: int,
) -> dict[str, Any]:
    """Run one local-first snapshot against the latest saved model checkpoint."""

    run_path = Path(run_dir).expanduser().resolve()
    output = run_path / "progress" / f"target_{target_env_steps:012d}"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "progress_snapshot.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous.get("state") != "complete":
        for partial in (output / "evaluation", output / "media"):
            if partial.exists():
                shutil.rmtree(partial)
    checkpoint = run_path / 'checkpoints' / f'step_{checkpoint_step:012d}'
    retained = run_path / 'evaluation_checkpoints' / checkpoint.name
    if not retained.exists():
        retained.parent.mkdir(exist_ok=True)
        shutil.copytree(checkpoint, retained)
    shutil.copy2(run_path / 'checkpoints' / 'latest.json', retained.parent / 'latest.json')
    commands = tuple((*command, '--checkpoint', str(retained))
                     for command in _commands(run_path, output, config.environment))
    manifest: dict[str, Any] = {
        "checkpoint_step": checkpoint_step,
        "commands": [list(command) for command in commands],
        "environment": config.environment,
        "started_unix_seconds": time.time(),
        "state": "running",
        "target_env_steps": target_env_steps,
    }
    _write_json(manifest_path, manifest)
    log_path = output / "progress.log"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            for command in commands:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
    except Exception as exc:
        manifest.update(
            completed_unix_seconds=time.time(),
            error=f"{type(exc).__name__}: {exc}",
            state="failed",
        )
        _write_json(manifest_path, manifest)
        raise

    try:
        policy_mode = "stochastic"
        evaluation_path = output / "evaluation" / "metrics.json"
        evaluation = json.loads(evaluation_path.read_text())
        media_path = Path(evaluation['media']['metadata'])
        media = json.loads(media_path.read_text())
        video_path = Path(evaluation['media']['video'])
        video_key = 'media/potential_policy_video'
        if int(evaluation["checkpoint_step"]) != checkpoint_step:
            raise RuntimeError(
                f"progress evaluation used checkpoint {evaluation['checkpoint_step']}, "
                f"expected {checkpoint_step}"
            )
        if not video_path.is_file():
            raise FileNotFoundError(f"progress media did not create {video_path}")
        branch = json.loads((output / 'media' / 'branch_media_result.json').read_text())['rendered']
        branch_metadata = branch['metadata']
        branch_step = branch.get('checkpoint_step', branch_metadata.get('checkpoint_step'))
        if int(branch_step) != checkpoint_step:
            raise RuntimeError(f'branch media used checkpoint {branch_step}, expected {checkpoint_step}')
        branch_videos = [Path(p) for p in branch['saved_paths'].values() if p.endswith('.mp4')]
        if len(branch_videos) != 1 or not branch_videos[0].is_file():
            raise FileNotFoundError('shared-prefix skill video missing')
        branch_fps = int(branch.get('fps', branch_metadata.get('fps')))
        branch_key = ('media/trajectory_arm_overlay_video' if config.environment == 'builderbench_corner3'
                      else 'media/trajectory_ant_overlay_video')
    except Exception as exc:
        manifest.update(
            completed_unix_seconds=time.time(),
            error=f"{type(exc).__name__}: {exc}",
            state="failed",
        )
        _write_json(manifest_path, manifest)
        raise
    manifest.update(
        completed_unix_seconds=time.time(),
        evaluation=str(evaluation_path),
        policy_mode=policy_mode,
        media_metadata=str(media_path),
        state="complete",
        video=str(video_path),
        video_fps=int(media["fps"]),
        video_key=video_key,
        branch_video=str(branch_videos[0]),
        branch_video_fps=branch_fps,
        branch_video_key=branch_key,
    )
    _write_json(manifest_path, manifest)
    return manifest


def progress_wandb_row(manifest: dict[str, Any]) -> dict[str, float]:
    evaluation = json.loads(Path(manifest["evaluation"]).read_text())
    row = {
        "env_steps": float(manifest["checkpoint_step"]),
        "progress/target_env_steps": float(manifest["target_env_steps"]),
    }
    # Keep the established panels and make the evaluation protocol explicit.
    row["evaluation/stochastic_policy"] = float(evaluation.get("policy_mode") == "stochastic")
    for name, statistics in evaluation["summary"].items():
        row[f"evaluation/{name}/mean"] = float(statistics["mean"])
    for track, coverage in evaluation.get("state_coverage", {}).items():
        for name, value in coverage.items():
            if name != "cell":
                row[f"evaluation/state_coverage/{track}/{name}"] = float(value)
    for name, value in evaluation.get("potential_policy", {}).get("metrics", {}).items():
        row[f"evaluation/potential_policy/{name}"] = float(value)
    return row
