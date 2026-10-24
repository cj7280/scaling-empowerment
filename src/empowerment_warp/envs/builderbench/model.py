"""BuilderBench Corner3 model construction and static array metadata."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import math

import mujoco
import numpy as np

from empowerment_warp.envs.builderbench.config import BuilderBenchConfig


@dataclass(frozen=True, slots=True)
class Dof:
    joint_type: int
    axis: tuple[int, int, int]
    stiffness: float
    limits: tuple[float, float]
    transmission: int


_ARM_DOFS = {
    "arm_tx": Dof(
        mujoco.mjtJoint.mjJNT_SLIDE,
        (0, 1, 0),
        300.0,
        (-0.05, 0.45),
        mujoco.mjtTrn.mjTRN_JOINT,
    ),
    "arm_ty": Dof(
        mujoco.mjtJoint.mjJNT_SLIDE,
        (1, 0, 0),
        300.0,
        (-0.35, 0.35),
        mujoco.mjtTrn.mjTRN_JOINT,
    ),
    "arm_tz": Dof(
        mujoco.mjtJoint.mjJNT_SLIDE,
        (0, 0, -1),
        300.0,
        (0.07, 0.5),
        mujoco.mjtTrn.mjTRN_JOINT,
    ),
    "arm_yaw": Dof(
        mujoco.mjtJoint.mjJNT_HINGE,
        (0, 0, -1),
        300.0,
        (-1.5707, 1.5707),
        mujoco.mjtTrn.mjTRN_JOINT,
    ),
}

_COLORS = np.asarray(
    [
        [1.0, 0.49, 0.43, 1.0],
        [0.0, 0.58, 0.55, 1.0],
        [0.93, 0.78, 0.28, 1.0],
    ]
)


@dataclass(frozen=True, slots=True)
class ModelBundle:
    model: mujoco.MjModel
    init_qpos: np.ndarray
    init_qvel: np.ndarray
    init_ctrl: np.ndarray
    object_qpos_adr: np.ndarray
    object_qvel_adr: np.ndarray
    finger_qpos_adr: np.ndarray
    gripper_pos_slice: slice
    gripper_quat_slice: slice
    gripper_linvel_slice: slice
    mocap_ids: np.ndarray
    starts: np.ndarray
    target_goal_offsets: np.ndarray
    target_mask: np.ndarray
    workspace_bounds: np.ndarray
    ctrl_bounds: np.ndarray


def _integrator(value: str) -> int:
    integrators = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    try:
        return integrators[value]
    except KeyError as exc:
        raise ValueError(f"unsupported MuJoCo integrator: {value}") from exc


def _sensor_slice(model: mujoco.MjModel, name: str) -> slice:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    start = int(model.sensor_adr[sensor_id])
    return slice(start, start + int(model.sensor_dim[sensor_id]))


def _build_spec(config: BuilderBenchConfig) -> tuple[mujoco.MjSpec, list[str]]:
    package = resources.files("empowerment_warp.envs.builderbench")
    xml_path = package.joinpath("xmls", "scene.xml")
    spec = mujoco.MjSpec.from_file(str(xml_path))

    base_body = spec.body("base_mount")
    base_body.pos = np.asarray([0.0, 0.0, 0.12])
    base_body.quat = np.asarray([0.0, 0.707106, 0.707106, 0.0])

    for name, dof in _ARM_DOFS.items():
        base_body.add_joint(
            name=name,
            type=dof.joint_type,
            axis=dof.axis,
            range=dof.limits,
        )
        actuator = spec.add_actuator(
            name=name,
            target=name,
            trntype=dof.transmission,
            ctrlrange=dof.limits,
        )
        actuator.set_to_position(kp=dof.stiffness)

    object_names = []
    for index in range(3):
        body = spec.worldbody.add_body(
            name=f"block_{index}",
            pos=[0.1, 0.0, 0.02 * (2 * index + 1)],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
        body.add_freejoint(name=f"block_joint_{index}")
        body.add_geom(
            name=f"block_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            contype=3,
            conaffinity=1,
            solref=[0.004, 1.0],
            size=[0.02, 0.02, 0.02],
            rgba=_COLORS[index],
            density=config.cube_density,
        )
        target = spec.worldbody.add_body(
            name=f"target_mocap_{index}",
            mocap=True,
            pos=[0.1, 0.0, 0.02 * (2 * index + 1)],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
        target.add_geom(
            name=f"target_mocap_{index}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0.01, 0.01],
            rgba=_COLORS[index] - np.asarray([0.0, 0.0, 0.0, 0.8]),
            contype=0,
            conaffinity=0,
        )
        object_names.append(f"block_{index}")

    spec.stat.center = np.asarray([0.4, 0.0, 0.4])
    spec.stat.extent = 0.95
    spec.visual.global_.elevation = -30.0
    spec.visual.global_.azimuth = 180.0
    return spec, object_names


def build_model(config: BuilderBenchConfig) -> ModelBundle:
    """Compile the exact three-cube task model used by the release recipe."""

    spec, object_names = _build_spec(config)
    model = spec.compile()
    model.opt.timestep = config.sim_dt
    model.opt.integrator = _integrator(config.integrator)
    if config.solver_iterations is not None:
        model.opt.iterations = config.solver_iterations
    if config.solver_ls_iterations is not None:
        model.opt.ls_iterations = config.solver_ls_iterations

    data = mujoco.MjData(model)
    data.qpos[:12] = np.asarray(
        [0, 0, 0.07, 0, 0.0029, 0.00044, 0.0053, -0.0077, 0.0029, 0.00044, 0.0053, -0.0077]
    )
    data.ctrl = np.asarray([0, 0, 0, 0.07, 0])
    mujoco.mj_step(model, data, nstep=config.substeps)

    if config.centered_start:
        # The three slide axes are orthonormal in world coordinates. Place the
        # physical pinch site at the requested reset pose, then hold that pose.
        mujoco.mj_forward(model, data)
        joints = [model.joint(name).id for name in ("arm_tx", "arm_ty", "arm_tz")]
        pinch_pos = data.sensordata[_sensor_slice(model, "pinch_framepos_sensor")]
        delta = np.linalg.solve(data.xaxis[joints].T, np.asarray([0.2, 0.0, 0.12]) - pinch_pos)
        for joint_id, displacement in zip(joints, delta, strict=True):
            data.qpos[model.jnt_qposadr[joint_id]] += displacement
        for name in _ARM_DOFS:
            data.ctrl[model.actuator(name).id] = data.qpos[model.jnt_qposadr[model.joint(name).id]]
        mujoco.mj_forward(model, data)

    for name, dof in _ARM_DOFS.items():
        joint_id = model.joint(name).id
        mass = model.dof_M0[model.jnt_dofadr[joint_id]]
        model.joint(name).damping = 2.0 * math.sqrt(mass * dof.stiffness)

    object_qpos_adr = np.asarray(
        [model.jnt_qposadr[model.body(name).jntadr[0]] for name in object_names],
        dtype=np.int32,
    )
    object_qvel_adr = np.asarray(
        [model.jnt_dofadr[model.body(name).jntadr[0]] for name in object_names],
        dtype=np.int32,
    )
    finger_qpos_adr = np.asarray(
        [model.jnt_qposadr[model.joint(name).id] for name in ("left_driver_joint", "right_driver_joint")],
        dtype=np.int32,
    )
    mocap_ids = np.asarray([model.body(f"target_mocap_{i}").mocapid[0] for i in range(3)], dtype=np.int32)

    task_path = resources.files("empowerment_warp.envs.builderbench").joinpath("tasks", "cube-3.npz")
    with np.load(task_path) as task_data:
        starts = np.asarray(task_data["starts"][0], dtype=np.float32)
        target_mask = np.asarray(task_data["masks"][0], dtype=bool)
        target_goal_offsets = np.asarray(task_data["goals"][0][target_mask], dtype=np.float32)

    expand = config.workspace_expand_xy
    workspace_bounds = np.asarray(
        [[-0.05 - expand, -0.35 - expand, 0.0], [0.45 + expand, 0.35 + expand, 0.5]],
        dtype=np.float32,
    )

    return ModelBundle(
        model=model,
        init_qpos=np.asarray(data.qpos, dtype=np.float32),
        init_qvel=np.zeros(model.nv, dtype=np.float32),
        init_ctrl=np.asarray(data.ctrl, dtype=np.float32),
        object_qpos_adr=object_qpos_adr,
        object_qvel_adr=object_qvel_adr,
        finger_qpos_adr=finger_qpos_adr,
        gripper_pos_slice=_sensor_slice(model, "pinch_framepos_sensor"),
        gripper_quat_slice=_sensor_slice(model, "pinch_framequat_sensor"),
        gripper_linvel_slice=_sensor_slice(model, "pinch_framelinvel_sensor"),
        mocap_ids=mocap_ids,
        starts=starts,
        target_goal_offsets=target_goal_offsets,
        target_mask=target_mask,
        workspace_bounds=workspace_bounds,
        ctrl_bounds=np.asarray(model.actuator_ctrlrange.T, dtype=np.float32),
    )
