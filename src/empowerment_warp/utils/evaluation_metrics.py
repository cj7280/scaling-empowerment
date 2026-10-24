"""Evaluation metrics shared by the continuous evaluators: state coverage, phase
displacement, summary statistics, and the potential-policy position probe."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _metric_statistics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }






# State-coverage metric (METRA / CSF "state coverage"): the number of distinct grid cells the
# controlled object's x-y trajectory visits, pooled over every evaluation lane (all skills, all
# repeats) -- METRA counts 1 x 1 cells of the Ant's x-y position over its skill rollouts, CSF reports
# the same. Ant: agent x-y at 1.0; AntSoccer: agent AND ball x-y at 1.0; BuilderBench: cube x-y at
# 0.05 m (the cubes live in a ~1 m workspace, so a 1 m cell would be a single bin -- our choice,
# stated in the metric name). Reported for the whole episode, for the potential prefix alone and
# for the skill phase alone, plus the mean per-skill coverage (cells one skill's repeats visit).
COVERAGE_CELL = {
    "agent_xy": 1.0, "ball_xy": 1.0, "cube_xy": 0.05,
    "block_0_xyz": 0.05, "block_1_xyz": 0.05, "block_2_xyz": 0.05,
    "gripper_xyz": 0.05,
}


def _potential_phase_mask(alive: np.ndarray, onset: int | np.ndarray) -> np.ndarray:
    """Post-action rows before each lane's skill activation belong to its prefix."""
    boundaries = np.broadcast_to(np.asarray(onset), (alive.shape[1],))
    return np.arange(alive.shape[0])[:, None] < boundaries[None, :]


def state_coverage(positions: np.ndarray, alive: np.ndarray, cell: float, *,
                   onset: int | np.ndarray, skill_indices: np.ndarray | None = None,
                   floor_cells: set[tuple[int, int]] | None = None) -> dict[str, float]:
    """Positions [T, N, D], D=2 or 3; alive [T, N] masks valid lane steps."""
    prefix_mask = _potential_phase_mask(alive, onset)
    cell_ids = np.floor(positions / cell).astype(np.int64)
    def cells(mask: np.ndarray) -> set:
        return {tuple(c) for c in cell_ids[alive & mask]}
    visited = {
        "": cells(np.ones_like(alive)),
        "_potential_phase": cells(prefix_mask),
        "_skill_phase": cells(~prefix_mask),
    }
    out = {f"cells{suffix}": float(len(cell_set)) for suffix, cell_set in visited.items()}
    possible = len(floor_cells) if floor_cells is not None else 0
    if floor_cells is not None:
        if possible <= 0:
            raise ValueError("arena floor cell set is empty")
        out["coverage_cells_possible"] = float(possible)
        for suffix, cell_set in visited.items():
            in_floor = len(cell_set & floor_cells)
            out[f"coverage_cells_visited_in_floor{suffix}"] = float(in_floor)
            out[f"coverage_fraction{suffix}"] = in_floor / possible
    if skill_indices is not None:
        per_skill = []
        for k in np.unique(skill_indices):
            lanes = skill_indices == k
            ok = alive[:, lanes] & ~prefix_mask[:, lanes]
            ids = cell_ids[:, lanes]
            skill_cells = {tuple(c) for c in ids[ok]}
            per_skill.append(len(skill_cells))
        out["cells_per_skill_mean"] = float(np.mean(per_skill)) if per_skill else 0.0
    return out


def builderbench_state_coverage(
    blocks: np.ndarray, gripper: np.ndarray, alive: np.ndarray, *, onset: int,
) -> dict[str, dict[str, float]]:
    """Pooled block XY and separate block/gripper XYZ counts at 5 cm resolution."""
    from empowerment_warp.core.rollout import COVERAGE_TRACKS

    time, lanes, count, _ = blocks.shape
    coverage = {"cube_xy": state_coverage(
        blocks[..., :2].reshape(time, lanes * count, 2),
        np.repeat(alive, count, axis=1), COVERAGE_CELL["cube_xy"], onset=onset,
    )}
    positions = (blocks[:, :, 0], blocks[:, :, 1], blocks[:, :, 2], gripper)
    for (name, cell, bound), track in zip(COVERAGE_TRACKS["objects"][1:], positions, strict=True):
        in_grid = np.all((track >= -bound) & (track < bound), axis=-1)
        coverage[name] = state_coverage(track, alive & in_grid, cell, onset=onset)
    return coverage


def _last_alive_position(track: np.ndarray, alive: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Per lane, the position at the last alive step of `track` [T, N, ...]; `fallback` where none."""
    if track.shape[0] == 0:
        return fallback
    any_alive = alive.any(axis=0)
    last = track.shape[0] - 1 - np.argmax(alive[::-1], axis=0)          # index of the last True per lane
    picked = track[last, np.arange(track.shape[1])]
    return np.where(any_alive.reshape((-1,) + (1,) * (track.ndim - 2)), picked, fallback)


def phase_displacement(track: np.ndarray, alive: np.ndarray, initial: np.ndarray, *, onset: int | np.ndarray
                       ) -> dict[str, np.ndarray]:
    """Net x-y displacement of each lane in the two episode phases.

    `track` [T, N, ..., 2] positions after each step, `alive` [T, N], `initial` [N, ..., 2] reset
    positions. potential = |p(end of z=0 prefix) - p(reset)|; skill = |p(last alive step) - p(end of
    prefix)| (0 for a lane that died inside the prefix). Both sum to at most the whole-episode
    displacement the evaluators already report.
    """
    prefix_end = _last_alive_position(track, alive & _potential_phase_mask(alive, onset), initial)
    final = _last_alive_position(track, alive, initial)
    return {"potential": np.linalg.norm(prefix_end - initial, axis=-1),
            "skill": np.linalg.norm(final - prefix_end, axis=-1)}


def save_potential_policy_probe(output, tracks, initial, alive, cells, *, checkpoint_step, seed):
    """Count reset plus valid post-action cells; displacement includes terminal actions."""
    alive = np.asarray(alive, dtype=bool)
    metrics = {"horizon_steps": float(len(alive)), "num_envs": float(alive.shape[1]),
               "valid_steps_mean": float(alive.sum(axis=0).mean()),
               "early_termination_fraction": float((~alive[-1]).mean())}
    arrays = {"alive": alive}
    for name, track in tracks.items():
        track, start = np.asarray(track), np.asarray(initial[name])
        assert track.shape[:2] == alive.shape and start.shape == track.shape[1:]
        assert np.isfinite(track[alive]).all() and np.isfinite(start).all()
        positions = np.concatenate([start[None], track], axis=0)
        valid = np.concatenate([np.ones((1, alive.shape[1]), dtype=bool), alive], axis=0)
        ids = np.floor(positions[valid] / cells[name]).astype(np.int64)
        last = len(valid) - 1 - np.argmax(valid[::-1], axis=0)
        final = positions[last, np.arange(alive.shape[1])]
        displacement = np.linalg.norm(final - start, axis=-1)
        metrics.update({f"{name}/cells_visited": float(len(np.unique(ids, axis=0))),
                        f"{name}/displacement_mean": float(displacement.mean()),
                        f"{name}/cell_size": float(cells[name])})
        arrays.update({name: positions, f"{name}_displacement": displacement})
    if "agent_xy" in tracks:
        agent = arrays["agent_xy"][last, np.arange(alive.shape[1])]
        # Distance of the final agent position from the arena center.
        distance = np.linalg.norm(agent - np.array([10., 10.]), axis=-1)
        metrics["agent_to_center_xy/distance_mean"] = float(distance.mean())
        arrays["agent_to_center_xy_distance"] = distance
        if "ball_xy" in tracks:
            ball = arrays["ball_xy"][last, np.arange(alive.shape[1])]
            distance = np.linalg.norm(agent - ball, axis=-1)
            metrics["agent_to_ball_xy/distance_mean"] = float(distance.mean())
            arrays["agent_to_ball_xy_distance"] = distance
    if "gripper_xyz" in tracks:
        # Match the figure's endpoint: first episode, last valid action, not reset.
        gripper = arrays["gripper_xyz"][last, np.arange(alive.shape[1])]
        blocks = np.stack([
            arrays[f"block_{i}_xyz"][last, np.arange(alive.shape[1])]
            for i in range(3)
        ], axis=1)
        for axes, dimensions in (("xy", 2), ("xyz", 3)):
            distances = np.linalg.norm(
                blocks[..., :dimensions] - gripper[:, None, :dimensions], axis=-1,
            )
            nearest = distances.min(axis=1)
            metrics[f"gripper_to_nearest_block_{axes}/distance_mean"] = float(nearest.mean())
            arrays[f"gripper_to_blocks_{axes}_distance"] = distances
            arrays[f"gripper_to_nearest_block_{axes}_distance"] = nearest
    path = Path(output).with_name(Path(output).stem + "_potential_positions.npz")
    np.savez_compressed(path, **arrays, checkpoint_step=checkpoint_step, seed=seed)
    return {"metrics": metrics, "positions": str(path), "checkpoint_step": checkpoint_step,
            "seed": seed, "policy": "z=0", "reset_included_in_visitation": True,
            "protocol": "first episode from fresh reset; 100 actions; mask after first done",
            "position_units": "environment length units (meters for BuilderBench)"}
