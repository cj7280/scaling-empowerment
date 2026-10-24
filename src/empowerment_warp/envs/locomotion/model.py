"""Standalone MuJoCo model construction for Ant and arena AntSoccer."""

from __future__ import annotations

from importlib import resources

import mujoco
import numpy as np

from empowerment_warp.envs.locomotion.config import AntConfig, SOCCER_TASKS


_ARENA_MAP = np.asarray(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.int8,
)
_MAZE_UNIT = 4.0
_OFFSET_X = 4.0
_OFFSET_Y = 4.0


def _integrator(value: str) -> int:
    return {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }[value]


def arena_cell_xy(i: int, j: int) -> np.ndarray:
    return np.asarray([j * _MAZE_UNIT - _OFFSET_X, i * _MAZE_UNIT - _OFFSET_Y], dtype=np.float32)


def arena_floor_cell_indices(
    cell: float,
    *,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Integer coverage bins contained in the arena floor (wall cells excluded).

    Bin ``(x, y)`` spans ``origin_xy + [x, y] * cell`` through its upper,
    exclusive edge. Arena tile edges must align with the requested coverage grid;
    this keeps the mask exact rather than deciding boundary bins by sampling.
    """

    if cell <= 0.0:
        raise ValueError("coverage cell size must be positive")
    origin = np.asarray(origin_xy, dtype=np.float64)
    bins: list[tuple[int, int]] = []
    half_unit = _MAZE_UNIT / 2.0
    for i, row in enumerate(_ARENA_MAP):
        for j, occupied in enumerate(row):
            if occupied:
                continue
            center = arena_cell_xy(i, j).astype(np.float64)
            lower = (center - half_unit - origin) / cell
            upper = (center + half_unit - origin) / cell
            rounded_lower = np.rint(lower).astype(np.int64)
            rounded_upper = np.rint(upper).astype(np.int64)
            if not (np.allclose(lower, rounded_lower) and np.allclose(upper, rounded_upper)):
                raise ValueError(
                    f"arena tile edges do not align with {cell:g} m coverage cells "
                    f"from origin {tuple(origin_xy)}"
                )
            bins.extend(
                (x, y)
                for x in range(int(rounded_lower[0]), int(rounded_upper[0]))
                for y in range(int(rounded_lower[1]), int(rounded_upper[1]))
            )
    return np.unique(np.asarray(bins, dtype=np.int64), axis=0)


def build_ant_model(config: AntConfig, *, soccer: bool = False) -> mujoco.MjModel:
    """Build the vendored locomaze Ant, optionally with arena walls and ball."""

    package = resources.files("empowerment_warp.envs.locomotion")
    spec = mujoco.MjSpec.from_file(str(package.joinpath("assets", "ant.xml")))
    if soccer:
        task_id = int(getattr(config, "task_id", 4))
        goal_xy = arena_cell_xy(*SOCCER_TASKS[task_id - 1][2])
        floor = spec.geom("floor")
        floor.pos = np.asarray([10.0, 10.0, 0.0])
        floor.size = np.asarray([16.0, 16.0, 0.2])
        for i, row in enumerate(_ARENA_MAP):
            for j, occupied in enumerate(row):
                if not occupied:
                    continue
                xy = arena_cell_xy(i, j)
                spec.worldbody.add_geom(
                    name=f"wall_{i}_{j}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[float(xy[0]), float(xy[1]), 1.0],
                    size=[2.0, 2.0, 1.0],
                    contype=1,
                    conaffinity=1,
                    material="wall",
                )
        spec.worldbody.add_geom(
            name="target",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.5, 0.05, 0.0],
            pos=[float(goal_xy[0]), float(goal_xy[1]), 0.05],
            contype=0,
            conaffinity=0,
            material="target",
        )
        ball = spec.worldbody.add_body(name="ball", pos=[0.0, 0.0, 0.5])
        ball.add_freejoint(name="ball_root")
        ball.add_geom(
            name="ball",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.25, 0.0, 0.0],
            priority=1,
            conaffinity=1,
            condim=6,
            material="ball",
        )
        ball.add_light(
            name="ball_light",
            pos=[0.0, 0.0, 4.0],
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
        )

    model = spec.compile()
    model.opt.timestep = config.sim_dt
    model.opt.integrator = _integrator(config.integrator)
    return model
