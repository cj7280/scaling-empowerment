"""Local-first Weights & Biases mirroring for training metrics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
from typing import Any, Mapping


DEFAULT_WANDB_PROJECT = "empowerment-warp"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _reserved_identity(
    run_dir: Path,
    config: Mapping[str, Any],
    previous: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Resolve a stable run handle before the network-dependent init call."""

    entity = (
        previous.get("entity")
        or previous.get("wandb_entity")
        or os.environ.get("WANDB_ENTITY") or None
    )
    project = str(
        previous.get("project")
        or previous.get("wandb_project")
        or os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    )
    environment = str(config.get("environment", "unknown"))
    algorithm = str(config.get("algorithm", "unknown"))
    group = str(
        previous.get("group")
        or previous.get("wandb_group")
        or os.environ.get("WANDB_RUN_GROUP", f"{environment}-{algorithm}-warp")
    )
    name = str(
        previous.get("name")
        or previous.get("wandb_run_name")
        or os.environ.get("WANDB_RUN_NAME", run_dir.name)
    )
    previous_id = previous.get("id") or previous.get("wandb_run_id")
    requested_id = os.environ.get("WANDB_RUN_ID")
    if previous_id:
        run_id = str(previous_id)
    elif requested_id:
        run_id = requested_id
    else:
        material = f"{run_dir.resolve()}\0{entity}\0{project}\0{group}\0{name}"
        run_id = hashlib.sha256(material.encode()).hexdigest()[:12]
    url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}" if entity else None
    return {
        "entity": entity,
        "project": project,
        "id": run_id,
        "name": name,
        "group": group,
        "mode": mode,
        "url": url,
        "local_run_dir": str(run_dir.resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "wandb_entity": entity,
        "wandb_project": project,
        "wandb_run_id": run_id,
        "wandb_run_name": name,
        "wandb_group": group,
        "wandb_mode": mode,
        "wandb_url": url,
    }


def _write_identity(run_dir: Path, identity: Mapping[str, Any]) -> None:
    _write_json(run_dir / "wandb_run.json", identity)
    (run_dir / "wandb_links.txt").write_text(f"{identity['url'] or ''}\n")


def record_pending_upload(run_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    """Append one recoverable upload request after all referenced files exist."""

    path = Path(run_dir) / "wandb_pending_uploads.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
    return path


@dataclass
class WandbMirror:
    """Best-effort mirror whose failures never interrupt local training."""

    run_dir: Path
    run: Any | None = None
    metadata: dict[str, Any] | None = None
    last_segment: int = 0

    @classmethod
    def start(
        cls,
        run_dir: Path,
        config: Mapping[str, Any],
        *,
        resume: bool = False,
    ) -> WandbMirror:
        run_dir = Path(run_dir)
        status_path = run_dir / "wandb_status.json"
        mode = os.environ.get("WANDB_MODE", "online").lower()
        previous = _read_json(run_dir / "wandb_run.json") if resume else {}
        previous_status = _read_json(status_path) if resume else {}
        identity = _reserved_identity(run_dir, config, previous, mode)
        _write_identity(run_dir, identity)
        if mode in {"disabled", "off", "false", "0"}:
            _write_json(
                status_path,
                {
                    "state": "disabled",
                    "mode": mode,
                    "last_segment": int(previous_status.get("last_segment", 0)),
                    "run_id": identity["id"],
                    "url": identity["url"],
                },
            )
            return cls(run_dir=run_dir, metadata=identity)

        _write_json(
            status_path,
            {
                "state": "initializing",
                "mode": mode,
                "last_segment": int(previous_status.get("last_segment", 0)),
                "run_id": identity["id"],
                "url": identity["url"],
            },
        )
        try:
            wandb = import_module("wandb")
            tags = os.environ.get("WANDB_TAGS")
            run = wandb.init(
                entity=identity["entity"],
                project=identity["project"],
                group=identity["group"],
                name=identity["name"],
                tags=(
                    tags.split(",")
                    if tags
                    else [
                        "warp",
                        "gpu",
                        str(config.get("environment", "unknown")),
                        str(config.get("algorithm", "unknown")),
                    ]
                ),
                config=dict(config),
                dir=str(run_dir),
                mode=mode,
                id=identity["id"],
                resume="allow",
                settings=wandb.Settings(
                    init_timeout=120,
                    x_service_wait=120,
                    silent=True,
                ),
            )
            if run is None:
                raise RuntimeError("wandb.init returned no run")
            step_metric = (
                "train/wall_seconds"
                if config.get("max_train_seconds") is not None
                else "env_steps"
            )
            run.define_metric(step_metric)
            run.define_metric("*", step_metric=step_metric)
            # env_steps is the explicit axis; W&B uses its own log-call counter.
            metadata = {
                **identity,
                "entity": run.entity,
                "project": run.project,
                "id": run.id,
                "name": run.name,
                "url": run.url,
                "wandb_entity": run.entity,
                "wandb_project": run.project,
                "wandb_run_id": run.id,
                "wandb_run_name": run.name,
                "wandb_url": run.url,
            }
            _write_identity(run_dir, metadata)
            last_segment = int(previous_status.get("last_segment", 0))
            _write_json(
                status_path,
                {
                    "state": "running",
                    "mode": mode,
                    "last_segment": last_segment,
                    "run_id": run.id,
                    "url": run.url,
                },
            )
            return cls(
                run_dir=run_dir,
                run=run,
                metadata=metadata,
                last_segment=last_segment,
            )
        except Exception as exc:  # W&B is secondary to local artifacts.
            _write_json(
                status_path,
                {
                    "state": "failed",
                    "mode": mode,
                    "error": f"{type(exc).__name__}: {exc}",
                    "last_segment": int(previous_status.get("last_segment", 0)),
                    "run_id": identity["id"],
                    "url": identity["url"],
                    "retryable": True,
                },
            )
            print(f"W&B mirroring disabled after initialization failure: {exc}", flush=True)
            return cls(
                run_dir=run_dir,
                metadata=identity,
                last_segment=int(previous_status.get("last_segment", 0)),
            )

    def log(self, row: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        try:
            # Several eval/media rows share one env step. W&B commits each log call;
            # an explicit repeated step silently drops later rows. The defined
            # env_steps axis remains the scientific clock.
            self.run.log(dict(row))
            self.last_segment = int(row.get("segment", self.last_segment))
            _write_json(
                self.run_dir / "wandb_status.json",
                {
                    "state": "running",
                    "mode": os.environ.get("WANDB_MODE", "online"),
                    "last_segment": self.last_segment,
                    "last_env_steps": float(row["env_steps"]),
                    "run_id": self.metadata["id"] if self.metadata else None,
                    "url": self.metadata["url"] if self.metadata else None,
                },
            )
        except Exception as exc:  # W&B is secondary to local artifacts.
            _write_json(
                self.run_dir / "wandb_status.json",
                {
                    "state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "last_segment": self.last_segment,
                    "run_id": self.metadata["id"] if self.metadata else None,
                    "url": self.metadata["url"] if self.metadata else None,
                    "retryable": True,
                },
            )
            print(f"W&B mirroring disabled after log failure: {exc}", flush=True)
            self.run = None

    def log_video(
        self,
        row: Mapping[str, Any],
        *,
        key: str,
        path: str | Path,
        fps: int,
    ) -> bool:
        """Log scalar context and one local MP4 through the active run."""

        if self.run is None:
            return False
        try:
            wandb = import_module("wandb")
            self.log(
                {
                    **dict(row),
                    key: wandb.Video(str(path), fps=int(fps), format="mp4"),
                }
            )
        except Exception as exc:  # W&B is secondary to local artifacts.
            print(f"W&B video construction failed: {exc}", flush=True)
            return False
        return self.run is not None

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
            _write_json(
                self.run_dir / "wandb_status.json",
                {
                    "state": "finished",
                    "mode": os.environ.get("WANDB_MODE", "online"),
                    "last_segment": self.last_segment,
                    "run_id": self.metadata["id"] if self.metadata else None,
                    "url": self.metadata["url"] if self.metadata else None,
                },
            )
        except Exception as exc:  # W&B is secondary to local artifacts.
            _write_json(
                self.run_dir / "wandb_status.json",
                {
                    "state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "last_segment": self.last_segment,
                    "run_id": self.metadata["id"] if self.metadata else None,
                    "url": self.metadata["url"] if self.metadata else None,
                    "retryable": True,
                },
            )
            print(f"W&B finish failed: {exc}", flush=True)
        finally:
            self.run = None


def sync_wandb_run(run_dir: Path) -> dict[str, Any]:
    """Mirror locally authoritative JSONL rows, resuming without duplicates."""

    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not config_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"{run_dir} must contain config.json and metrics.jsonl")

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    mirror = WandbMirror.start(
        run_dir,
        json.loads(config_path.read_text()),
        resume=(run_dir / "wandb_run.json").exists(),
    )
    if mirror.run is None:
        return _read_json(run_dir / "wandb_status.json")
    for row in rows:
        if int(row["segment"]) > mirror.last_segment:
            mirror.log(row)
    mirror.finish()
    return _read_json(run_dir / "wandb_run.json")
