"""Jitted environment collection with explicit skills and reset ownership."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any, NamedTuple

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.core.replay import BranchTrajectory, Transition


ActionFn = Callable[[Any, jax.Array, jax.Array,
                     jax.Array, jax.Array], jax.Array]
SkillSampler = Callable[[jax.Array, int], jax.Array]
ObservationFn = Callable[[jax.Array], jax.Array]


def branch_anchor_layout(
    num_envs: int,
    phase_groups: int,
    targets_per_anchor: int,
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """Partition each completion phase into balanced shared-anchor blocks."""

    if min(num_envs, phase_groups, targets_per_anchor) <= 0:
        raise ValueError("branch layout sizes must be positive")
    if phase_groups > num_envs:
        raise ValueError("branch phase groups cannot exceed the environment count")

    anchor_lanes: list[int] = []
    block_indices: list[int] = []
    block_positions: list[int] = []
    block_offset = 0
    for group in range(phase_groups):
        start = (group * num_envs + phase_groups - 1) // phase_groups
        stop = ((group + 1) * num_envs + phase_groups - 1) // phase_groups
        size = stop - start
        block_count = math.ceil(size / targets_per_anchor)
        for local_lane in range(size):
            local_block = local_lane * block_count // size
            block_start = (
                local_block * size + block_count - 1
            ) // block_count
            anchor_lanes.append(start + block_start)
            block_indices.append(block_offset + local_block)
            block_positions.append(local_lane - block_start)
        block_offset += block_count

    return (
        jnp.asarray(anchor_lanes, dtype=jnp.int32),
        jnp.asarray(block_indices, dtype=jnp.int32),
        jnp.asarray(block_positions, dtype=jnp.int32),
        block_offset,
    )


@struct.dataclass
class CollectorState:
    env_state: Any
    anchor_observations: jax.Array
    skills: jax.Array
    episode_ids: jax.Array
    next_episode_id: jax.Array
    episode_genuine: jax.Array
    episode_start_agent_xy: jax.Array
    episode_start_ball_xy: jax.Array
    rng: jax.Array
    # [num_envs] bool; the "separate lanes" (no policy switching) ablation, as in the PQN
    # collector: a potential lane runs z=0 for its WHOLE episode and never activates a skill,
    # redrawn at every reset with probability `potential_lane_fraction`. All-False = the
    # two-phase method. None = built by hand.
    potential_lane: jax.Array | None = None
    # [num_tracks, 2, num_cells] uint8 occupancy of the TRAINING rollout, cumulative since the
    # collector was initialized (it rides in the carry, so it survives chunks and checkpoints):
    # axis 1 = phase the lane was in when it visited the cell (0 = potential z=0, 1 = skill).
    # Tracks: locomotion (agent_xy, ball_xy) or objects (cube_xy); see COVERAGE_TRACKS. None =
    # an env without world-frame positions (toy/test envs) or a state built by hand.
    coverage_grids: jax.Array | None = None


def _draw_potential_lanes(key: jax.Array, num_envs: int, fraction: float) -> jax.Array:
    if float(fraction) <= 0.0:
        return jnp.zeros((num_envs,), dtype=jnp.bool_)
    # fold_in keeps the existing split streams (action/skill/reset keys) byte-identical.
    return jax.random.uniform(jax.random.fold_in(key, 7), (num_envs,)) < float(fraction)


@struct.dataclass
class MarginalSource:
    env_state: Any
    skills: jax.Array


@struct.dataclass
class StaggeredBranchState:
    """Checkpointed full-width counterfactual trajectories at staggered phases."""

    env_state: Any
    policy_anchors: jax.Array
    encoder_anchors: jax.Array
    source_skills: jax.Array
    skills: jax.Array
    outcomes: jax.Array
    valid: jax.Array
    progress: jax.Array
    genuine: jax.Array
    alive: jax.Array
    active: jax.Array
    skill_shift: jax.Array
    # Policy-facing history, kept only so branch rollouts can also train the
    # policy. `observations` are derived by shifting `policy_outcomes`, since
    # the outcome at row r is the state entering row r + 1.
    policy_outcomes: jax.Array
    actions: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    alive_rows: jax.Array
    emit_index: jax.Array
    # World-frame xy at the branch anchor. Displacement cannot be read off the
    # observation (the locomotion obs carries no global position), so it is
    # captured from the env state when the branch starts.
    anchor_xy: jax.Array


# Centre of the walled locomotion arena (floor spans [-2, 22] m on both axes; envs/locomotion/model.py).
ARENA_CENTER_XY = (10.0, 10.0)


class ExplorationMetrics(NamedTuple):
    """Completed-episode sums from the actual training rollout, plus what the POTENTIAL policy achieved
    at the moment each lane switched to a skill (distance from the ball for AntSoccer, from the
    blocks for BuilderBench, from the centre of the room for Ant): the
    `switch_*_sum` fields are summed over lanes at their switch step and divided by `switch_count`."""

    completed_episodes: jax.Array
    agent_disp_xy_sum: jax.Array
    ball_disp_xy_sum: jax.Array
    cube_disp_xyz_sum: jax.Array
    cube_disp_max_xyz_sum: jax.Array
    cube_disp_xy_sum: jax.Array
    cube_disp_max_xy_sum: jax.Array
    obj_moved_sum: jax.Array
    switch_count: jax.Array
    switch_center_dist_sum: jax.Array          # agent x-y to ARENA_CENTER_XY (locomotion)
    switch_agent_ball_dist_sum: jax.Array      # agent x-y to ball x-y (locomotion)
    switch_gripper_cube_dist_sum: jax.Array    # gripper x-y to the NEAREST cube x-y (objects)


def _locomotion_xy(env_state: Any, num_envs: int) -> tuple[jax.Array, jax.Array]:
    metrics = getattr(env_state, "metrics", None)
    if metrics is None or not hasattr(metrics, "agent_x"):
        zeros = jnp.zeros((num_envs, 2), dtype=jnp.float32)
        return zeros, zeros
    agent_xy = jnp.stack((metrics.agent_x, metrics.agent_y), axis=-1)
    ball_xy = jnp.stack((metrics.ball_x, metrics.ball_y), axis=-1)
    return agent_xy, ball_xy


# Training-time state coverage, split by phase. Same cells as evaluation.state_coverage
# (1.0 m for agent/ball, 0.05 m for blocks/gripper); grids span [-bound, bound)^D and
# positions outside it are not counted. Logged by the runner as
# exploration/cells_visited_<track>_{potential_phase,skill_phase,total}.
COVERAGE_TRACKS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "locomotion": (("agent_xy", 1.0, 50.0), ("ball_xy", 1.0, 50.0)),
    "objects": (
        ("cube_xy", 0.05, 1.0),
        ("block_0_xyz", 0.05, 1.0),
        ("block_1_xyz", 0.05, 1.0),
        ("block_2_xyz", 0.05, 1.0),
        ("gripper_xyz", 0.05, 1.0),
    ),
}


def coverage_track_specs(env: Any, env_state: Any | None = None) -> tuple[tuple[str, float, float], ...]:
    if hasattr(env, "_object_positions"):
        return COVERAGE_TRACKS["objects"]
    metrics = None if env_state is None else getattr(env_state, "metrics", None)
    if metrics is not None and hasattr(metrics, "agent_x"):
        return COVERAGE_TRACKS["locomotion"]
    return ()


def coverage_grid_side(cell: float, bound: float) -> int:
    return int(round(2.0 * bound / cell))


def _coverage_cells(xy: jax.Array, cell: float, bound: float) -> tuple[jax.Array, jax.Array]:
    """Positions [M, 2 or 3] -> flat cell index and an in-bounds mask."""
    side = coverage_grid_side(cell, bound)
    raw = jnp.floor((xy + bound) / cell).astype(jnp.int32)
    if xy.shape[-1] == 3:
        # Origin-aligned XYZ bins, matching the evaluation's floor(pos / cell).
        raw = jnp.floor(xy / cell).astype(jnp.int32) + int(round(bound / cell))
    in_bounds = jnp.all((raw >= 0) & (raw < side), axis=-1)
    clipped = jnp.clip(raw, 0, side - 1)
    index = clipped[:, 0]
    for axis in range(1, xy.shape[-1]):
        index = index * side + clipped[:, axis]
    return index, in_bounds


def initial_coverage_grids(specs: tuple[tuple[str, float, float], ...]) -> jax.Array | None:
    if not specs:
        return None
    num_cells = max(
        coverage_grid_side(cell, bound) ** (3 if name.endswith("_xyz") else 2)
        for name, cell, bound in specs
    )
    return jnp.zeros((len(specs), 2, num_cells), dtype=jnp.uint8)


def coverage_floor_masks(
    env: Any,
    specs: tuple[tuple[str, float, float], ...],
) -> dict[str, np.ndarray] | None:
    """Arena-floor masks for bounded soccer coverage; unbounded envs have none."""

    if not bool(getattr(env, "soccer", False)):
        return None
    from empowerment_warp.envs.locomotion.model import arena_floor_cell_indices

    masks: dict[str, np.ndarray] = {}
    for name, cell, bound in specs:
        side = coverage_grid_side(cell, bound)
        cells = arena_floor_cell_indices(cell, origin_xy=(-bound, -bound))
        if np.any(cells < 0) or np.any(cells >= side):
            raise ValueError(f"arena floor lies outside the {name} coverage grid")
        mask = np.zeros((side * side,), dtype=bool)
        mask[cells[:, 0] * side + cells[:, 1]] = True
        masks[name] = mask
    return masks


def _mark_coverage(
    grids: jax.Array,
    specs: tuple[tuple[str, float, float], ...],
    positions: tuple[jax.Array, ...],
    skill_phase: jax.Array,
) -> jax.Array:
    """OR each track's visited cells into the phase row its lane is in. `positions[t]` is
    [num_envs, D] or [num_envs, K, D], with D=2 or 3."""
    for track, ((_, cell, bound), xy) in enumerate(zip(specs, positions, strict=True)):
        phase = skill_phase
        if xy.ndim == 3:
            phase = jnp.repeat(skill_phase, xy.shape[1])
            xy = xy.reshape((-1, xy.shape[-1]))
        idx, ok = _coverage_cells(xy, cell, bound)
        one = jnp.ones_like(idx, dtype=jnp.uint8)
        grids = grids.at[track, 0, idx].max(jnp.where(ok & ~phase, one, 0))
        grids = grids.at[track, 1, idx].max(jnp.where(ok & phase, one, 0))
    return grids


def coverage_counts(
    grids: jax.Array | None,
    specs: tuple[tuple[str, float, float], ...],
    floor_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Cells visited per track: potential phase, skill phase, and their union."""
    if grids is None:
        return {}
    grid = np.asarray(jax.device_get(grids)).astype(bool)
    out: dict[str, float] = {}
    for track, (name, _, _) in enumerate(specs):
        potential, skill = grid[track, 0], grid[track, 1]
        out[f"exploration/cells_visited_{name}_potential_phase"] = float(potential.sum())
        out[f"exploration/cells_visited_{name}_skill_phase"] = float(skill.sum())
        out[f"exploration/cells_visited_{name}_total"] = float((potential | skill).sum())
        if floor_masks is None or name not in floor_masks:
            continue
        floor = floor_masks[name]
        possible = int(floor.sum())
        if possible <= 0:
            raise ValueError(f"coverage floor mask for {name} is empty")
        out[f"exploration/coverage_cells_possible_{name}"] = float(possible)
        for phase_name, visited in (
            ("potential_phase", potential),
            ("skill_phase", skill),
            ("total", potential | skill),
        ):
            in_floor = int((visited[: floor.size] & floor).sum())
            out[f"exploration/coverage_cells_visited_in_floor_{name}_{phase_name}"] = float(in_floor)
            out[f"exploration/coverage_fraction_{name}_{phase_name}"] = in_floor / possible
    return out


def staggered_reset_offsets(episode_length: int, num_envs: int, groups: int) -> jax.Array:
    """Balanced `staggered_reset_steps`: group g starts at g * (episode_length // groups).

    `groups <= 0` disables staggering (all environments start at step 0).
    """

    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if groups <= 0:
        return jnp.zeros((num_envs,), dtype=jnp.int32)
    groups = min(groups, num_envs)
    # Multiplication before integer division partitions every vector width into
    # groups whose occupancies differ by at most one. Dividing by
    # `num_envs // groups` instead dumps the entire remainder into the final
    # group (88 lanes instead of at most 41 for 2,048 lanes / 50 groups).
    group_index = jnp.arange(num_envs, dtype=jnp.int32) * groups // num_envs
    step_size = max(1, int(episode_length) // groups)
    return group_index * jnp.asarray(step_size, dtype=jnp.int32)


def _with_step_count(env_state: Any, step_count: jax.Array) -> Any:
    if hasattr(env_state, "replace"):
        return env_state.replace(step_count=step_count)
    return env_state._replace(step_count=step_count)


def _branch_start_state(env_state: Any) -> Any:
    reset_fields = {
        "reward": jnp.zeros_like(env_state.reward),
        "terminated": jnp.zeros_like(env_state.terminated),
        "truncated": jnp.zeros_like(env_state.truncated),
        "step_count": jnp.zeros_like(env_state.step_count),
    }
    if hasattr(env_state, "replace"):
        return env_state.replace(**reset_fields)
    return env_state._replace(**reset_fields)


def potential_switch_steps(
    episode_ids: jax.Array,
    potential_steps: int,
    random_potential_switch_max_steps: int = 0,
    seed: int = 0,
    geometric: bool = False,
) -> jax.Array:
    """Episode-keyed prefix lengths; replay reconstructs the same boundary."""

    ids = jnp.asarray(episode_ids, dtype=jnp.int32)
    if random_potential_switch_max_steps < 0:
        raise ValueError("random_potential_switch_max_steps must be nonnegative")
    if geometric:
        if potential_steps < 1 or random_potential_switch_max_steps:
            raise ValueError("geometric switching needs a positive mean and no uniform schedule")
        root = jax.random.PRNGKey(seed)
        return jax.vmap(lambda episode_id: jax.random.geometric(
            jax.random.fold_in(root, episode_id), 1.0 / potential_steps,
            dtype=jnp.int32,
        ))(ids.reshape(-1)).reshape(ids.shape)
    if random_potential_switch_max_steps == 0:
        return jnp.full_like(ids, potential_steps)
    root = jax.random.PRNGKey(seed)
    draws = jax.vmap(
        lambda episode_id: jax.random.randint(
            jax.random.fold_in(root, episode_id),
            (),
            0,
            random_potential_switch_max_steps + 1,
            dtype=jnp.int32,
        )
    )(ids.reshape(-1))
    return draws.reshape(ids.shape)


def initialize_collector(
    env: Any,
    key: jax.Array,
    skill_sampler: SkillSampler,
    *,
    potential_steps: int = 0,
    episode_length: int = 0,
    stagger_groups: int = 0,
    potential_lane_fraction: float = 0.0,
    random_potential_switch_max_steps: int = 0,
    potential_switch_seed: int = 0,
    geometric_potential_switch: bool = False,
) -> CollectorState:
    """Reset every environment; with `stagger_groups > 0` seed the elapsed-step
    counters at spread offsets so episode boundaries decorrelate across lanes.
    Environments with a ``stagger_initial_state`` hook instead shorten discarded
    startup episodes while keeping their physical clock and observation intact.
    Otherwise, lanes seeded past the prefix start with an anchored live skill."""

    reset_root, skill_key, next_key = jax.random.split(key, 3)
    reset_keys = jax.random.split(
        reset_root, env.config.num_envs if hasattr(env, "config") else env.num_envs)
    num_envs = reset_keys.shape[0]
    env_state = env.reset(reset_keys)
    if stagger_groups > 0 and episode_length <= 0:
        raise ValueError("staggered resets need the episode length")
    offsets = staggered_reset_offsets(episode_length, num_envs, stagger_groups)
    env_state = (env.stagger_initial_state(env_state, offsets)
                 if hasattr(env, "stagger_initial_state")
                 else _with_step_count(env_state, offsets))
    sampled_skills = skill_sampler(skill_key, num_envs)
    potential_lane = _draw_potential_lanes(skill_key, num_envs, potential_lane_fraction)
    episode_ids = jnp.arange(num_envs, dtype=jnp.int32)
    switch_steps = potential_switch_steps(
        episode_ids, potential_steps, random_potential_switch_max_steps, potential_switch_seed,
        geometric=geometric_potential_switch,
    )
    if geometric_potential_switch:
        offsets = jnp.minimum(offsets, switch_steps + episode_length - potential_steps - 1)
        env_state = _with_step_count(env_state, offsets)
    active_at_start = (env_state.step_count >= switch_steps) & (~potential_lane)
    if getattr(getattr(env, "config", None), "density_choice", False):
        active_at_start &= ~env_state.density_choice_pending
    skills = jnp.where(
        active_at_start[:, None], sampled_skills, jnp.zeros_like(sampled_skills))
    agent_xy, ball_xy = _locomotion_xy(env_state, num_envs)
    return CollectorState(
        env_state=env_state,
        anchor_observations=env_state.obs,
        skills=skills,
        episode_ids=episode_ids,
        next_episode_id=jnp.asarray(num_envs, dtype=jnp.int32),
        episode_genuine=offsets == 0,
        episode_start_agent_xy=agent_xy,
        episode_start_ball_xy=ball_xy,
        rng=next_key,
        potential_lane=potential_lane,
        coverage_grids=initial_coverage_grids(coverage_track_specs(env, env_state)),
    )


def make_rollout_chunk(
    env: Any,
    action_fn: ActionFn,
    skill_sampler: SkillSampler,
    *,
    chunk_steps: int,
    potential_steps: int = 0,
    capture_source: bool = False,
    potential_lane_fraction: float = 0.0,
    random_potential_switch_max_steps: int = 0,
    potential_switch_seed: int = 0,
    geometric_potential_switch: bool = False,
    skill_steps: int = 0,
) -> Callable[..., Any]:
    """Build one compiled chunk; policy and all environment work stay on device."""

    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")
    if potential_steps < 0:
        raise ValueError("potential_steps must be nonnegative")
    if geometric_potential_switch and skill_steps <= 0:
        raise ValueError("geometric switching needs a positive skill suffix")
    num_envs = env.config.num_envs if hasattr(env, "config") else env.num_envs
    density_choice = getattr(getattr(env, "config", None), "density_choice", False)

    tracks_objects = hasattr(env, "_object_positions")

    def one_step(
        params: Any,
        carry: CollectorState,
    ) -> tuple[CollectorState, tuple[Transition, ExplorationMetrics]]:
        next_rng, action_root, reset_root, skill_key = jax.random.split(carry.rng, 4)
        action_keys = jax.random.split(action_root, num_envs)
        actions = action_fn(
            params,
            carry.env_state.obs,
            carry.anchor_observations,
            carry.skills,
            action_keys,
        )
        next_env_state = env.step(carry.env_state, actions)
        if geometric_potential_switch:
            boundary = potential_switch_steps(
                carry.episode_ids, potential_steps, seed=potential_switch_seed, geometric=True,
            )
            next_env_state = next_env_state._replace(
                truncated=next_env_state.step_count >= boundary + skill_steps,
            )
        done = next_env_state.terminated | next_env_state.truncated
        completed = done & carry.episode_genuine
        completed_float = completed.astype(jnp.float32)
        agent_xy, ball_xy = _locomotion_xy(next_env_state, num_envs)
        agent_disp_xy = jnp.linalg.norm(
            agent_xy - carry.episode_start_agent_xy,
            axis=-1,
        )
        ball_disp_xy = jnp.linalg.norm(
            ball_xy - carry.episode_start_ball_xy,
            axis=-1,
        )
        if tracks_objects:
            object_pos = jax.vmap(env._object_positions)(next_env_state.data)
            object_delta = object_pos - next_env_state.initial_object_pos
            cube_disp_xyz = jnp.linalg.norm(object_delta, axis=-1)
            cube_disp_xy = jnp.linalg.norm(object_delta[..., :2], axis=-1)
            cube_disp_xyz_mean = jnp.mean(cube_disp_xyz, axis=-1)
            cube_disp_xyz_max = jnp.max(cube_disp_xyz, axis=-1)
            cube_disp_xy_mean = jnp.mean(cube_disp_xy, axis=-1)
            cube_disp_xy_max = jnp.max(cube_disp_xy, axis=-1)
            obj_moved = next_env_state.metrics.obj_moved
        else:
            cube_disp_xyz_mean = jnp.zeros((num_envs,), dtype=jnp.float32)
            cube_disp_xyz_max = jnp.zeros((num_envs,), dtype=jnp.float32)
            cube_disp_xy_mean = jnp.zeros((num_envs,), dtype=jnp.float32)
            cube_disp_xy_max = jnp.zeros((num_envs,), dtype=jnp.float32)
            obj_moved = jnp.zeros((num_envs,), dtype=jnp.float32)
        coverage_grids = carry.coverage_grids
        if coverage_grids is not None:
            # Phase of the lane DURING this step: the skill vector that produced the action.
            phase_skill = jnp.linalg.norm(carry.skills, axis=-1) > 1e-6
            if tracks_objects:
                coverage_grids = _mark_coverage(
                    coverage_grids, COVERAGE_TRACKS["objects"],
                    (object_pos[..., :2], object_pos[:, 0], object_pos[:, 1],
                     object_pos[:, 2], next_env_state.obs[:, :3]), phase_skill)
            else:
                coverage_grids = _mark_coverage(
                    coverage_grids, COVERAGE_TRACKS["locomotion"], (agent_xy, ball_xy), phase_skill)
        sampled_skills = skill_sampler(skill_key, num_envs)
        skill_on = jnp.linalg.norm(carry.skills, axis=-1) > 1e-6
        potential_lane = (
            jnp.zeros((num_envs,), dtype=jnp.bool_) if carry.potential_lane is None else carry.potential_lane
        )
        switch_steps = potential_switch_steps(
            carry.episode_ids, potential_steps, random_potential_switch_max_steps, potential_switch_seed,
            geometric=geometric_potential_switch,
        )
        activate = (~skill_on) & (next_env_state.step_count >= switch_steps) & (~potential_lane)
        if density_choice:
            activate &= ~next_env_state.density_choice_pending
        # What the potential policy achieved by the switch step (0 where the lane did not switch now).
        switch_float = activate.astype(jnp.float32)
        center_dist = jnp.linalg.norm(agent_xy - jnp.asarray(ARENA_CENTER_XY, agent_xy.dtype), axis=-1)
        agent_ball_dist = jnp.linalg.norm(agent_xy - ball_xy, axis=-1)
        if tracks_objects:
            gripper_cube_dist = jnp.min(
                jnp.linalg.norm(next_env_state.obs[:, None, :2] - object_pos[..., :2], axis=-1), axis=-1)
        else:
            gripper_cube_dist = jnp.zeros((num_envs,), dtype=jnp.float32)
        exploration = ExplorationMetrics(
            completed_episodes=completed_float,
            agent_disp_xy_sum=completed_float * agent_disp_xy,
            ball_disp_xy_sum=completed_float * ball_disp_xy,
            cube_disp_xyz_sum=completed_float * cube_disp_xyz_mean,
            cube_disp_max_xyz_sum=completed_float * cube_disp_xyz_max,
            cube_disp_xy_sum=completed_float * cube_disp_xy_mean,
            cube_disp_max_xy_sum=completed_float * cube_disp_xy_max,
            obj_moved_sum=completed_float * obj_moved,
            switch_count=switch_float,
            switch_center_dist_sum=switch_float * center_dist,
            switch_agent_ball_dist_sum=switch_float * agent_ball_dist,
            switch_gripper_cube_dist_sum=switch_float * gripper_cube_dist,
        )
        next_skills = jnp.where(activate[:, None], sampled_skills, carry.skills)
        next_skill_on = jnp.linalg.norm(next_skills, axis=-1) > 1e-6
        anchor_reset = ((~skill_on) & next_skill_on) | (~next_skill_on)
        anchor_reset_mask = anchor_reset.reshape(
            (num_envs,) + (1,) * (carry.anchor_observations.ndim - 1)
        )
        transition_next_anchors = jnp.where(
            anchor_reset_mask,
            next_env_state.obs,
            carry.anchor_observations,
        )
        transition = Transition(
            observations=carry.env_state.obs,
            anchor_observations=carry.anchor_observations,
            actions=actions,
            rewards=next_env_state.reward,
            masks=(~next_env_state.terminated).astype(jnp.float32),
            next_observations=next_env_state.obs,
            next_anchor_observations=transition_next_anchors,
            dones=done,
            skills=carry.skills,
            next_skills=next_skills,
            loss_weights=(carry.episode_genuine.astype(jnp.float32)
                          if getattr(env, "exclude_startup_fragments", False)
                          else jnp.ones((num_envs,), dtype=jnp.float32)),
            episode_ids=carry.episode_ids,
            # -1 excludes discarded startup fragments from the episode archive.
            # Replay rows include setup; the environment counter counts physics only.
            timesteps=(
                carry.env_state.step_count
                + (~carry.env_state.density_choice_pending).astype(jnp.int32)
                if getattr(getattr(env, "config", None), "density_choice", False)
                else (jnp.where(carry.episode_genuine, carry.env_state.step_count, -1)
                      if getattr(env, "exclude_startup_fragments", False)
                      else carry.env_state.step_count)
            ),
        )

        def reset_finished(state: Any) -> Any:
            reset_keys = jax.random.split(reset_root, num_envs)
            reset_state = env.reset(reset_keys)
            return env.where_done(state, reset_state)

        env_state = jax.lax.cond(
            jnp.any(done),
            reset_finished,
            lambda state: state,
            next_env_state,
        )
        reset_agent_xy, reset_ball_xy = _locomotion_xy(env_state, num_envs)
        # Lanes that reset now redraw their potential-lane flag; base = all False.
        redrawn_lane = jnp.where(done, _draw_potential_lanes(reset_root, num_envs, potential_lane_fraction), potential_lane)
        reset_offset = jnp.cumsum(done.astype(jnp.int32)) - 1
        new_episode_ids = carry.next_episode_id + reset_offset
        episode_ids = jnp.where(done, new_episode_ids, carry.episode_ids)
        reset_switch_steps = potential_switch_steps(
            episode_ids, potential_steps, random_potential_switch_max_steps, potential_switch_seed,
            geometric=geometric_potential_switch,
        )
        reset_skills = (
            jnp.where(reset_switch_steps[:, None] == 0, sampled_skills, jnp.zeros_like(sampled_skills))
            if not density_choice
            else jnp.zeros_like(sampled_skills)
        )
        reset_skills = jnp.where(redrawn_lane[:, None], jnp.zeros_like(reset_skills), reset_skills)
        skill_mask = done.reshape((num_envs,) + (1,) * (carry.skills.ndim - 1))
        skills = jnp.where(skill_mask, reset_skills, next_skills)
        anchor_mask = done.reshape(
            (num_envs,) + (1,) * (carry.anchor_observations.ndim - 1)
        )
        anchor_observations = jnp.where(
            anchor_mask,
            env_state.obs,
            transition_next_anchors,
        )
        next_episode_id = carry.next_episode_id + jnp.sum(done.astype(jnp.int32))
        return (
            CollectorState(
                env_state=env_state,
                anchor_observations=anchor_observations,
                skills=skills,
                episode_ids=episode_ids,
                next_episode_id=next_episode_id,
                episode_genuine=jnp.where(done, True, carry.episode_genuine),
                episode_start_agent_xy=jnp.where(
                    done[:, None],
                    reset_agent_xy,
                    carry.episode_start_agent_xy,
                ),
                episode_start_ball_xy=jnp.where(
                    done[:, None],
                    reset_ball_xy,
                    carry.episode_start_ball_xy,
                ),
                rng=next_rng,
                potential_lane=None if carry.potential_lane is None else redrawn_lane,
                coverage_grids=coverage_grids,
            ),
            (transition, exploration),
        )

    def flatten(
        collector: CollectorState,
        transitions: Transition,
        exploration: ExplorationMetrics,
    ) -> tuple[CollectorState, Transition, ExplorationMetrics]:
        flat = jax.tree_util.tree_map(
            lambda value: value.reshape(
                (chunk_steps * num_envs,) + value.shape[2:]),
            transitions,
        )
        exploration = jax.tree_util.tree_map(jnp.sum, exploration)
        return collector, flat, exploration

    def run(
        params: Any,
        collector: CollectorState,
    ) -> tuple[CollectorState, Transition, ExplorationMetrics]:
        collector, (transitions, exploration) = jax.lax.scan(
            lambda carry, _: one_step(params, carry),
            collector,
            None,
            length=chunk_steps,
        )
        return flatten(collector, transitions, exploration)

    def run_with_source(
        params: Any,
        collector: CollectorState,
        capture_step: jax.Array,
    ) -> tuple[CollectorState, Transition, ExplorationMetrics, MarginalSource]:
        source = MarginalSource(collector.env_state, collector.skills)

        def capture_and_step(
            carry: tuple[CollectorState, MarginalSource],
            step: jax.Array,
        ) -> tuple[
            tuple[CollectorState, MarginalSource],
            tuple[Transition, ExplorationMetrics],
        ]:
            current, captured = carry
            select = step == capture_step
            captured = MarginalSource(
                env_state=jax.tree_util.tree_map(
                    lambda old, new: jnp.where(select, new, old),
                    captured.env_state,
                    current.env_state,
                ),
                skills=jnp.where(select, current.skills, captured.skills),
            )
            next_collector, outputs = one_step(params, current)
            return (next_collector, captured), outputs

        (collector, source), (transitions, exploration) = jax.lax.scan(
            capture_and_step,
            (collector, source),
            jnp.arange(chunk_steps, dtype=jnp.int32),
        )
        collector, flat, exploration = flatten(
            collector, transitions, exploration)
        return collector, flat, exploration, source

    return jax.jit(run_with_source if capture_source else run)


def branch_policy_transitions(
    *,
    policy_anchors: jax.Array,
    skills: jax.Array,
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    alive: jax.Array,
    branch_epoch: jax.Array,
) -> tuple[Transition, jax.Array]:
    """Policy-replay rows for one branch sweep, flattened step-major.

    Branch rollouts are the only data whose anchor is fixed and independent of
    the commanded skill, which is exactly the geometry the potential reward
    assumes; live lanes hold one skill for a whole episode and so contain no
    same-state/different-skill contrast at all.

    Step-major inputs are `(horizon, num_envs, ...)`; anchors and skills are
    `(num_envs, ...)` and are constant along the branch. Episode ids are made
    NEGATIVE so the replay-batch constructor preserves the fixed branch anchor
    instead of applying the live replay's hindsight-anchor replacement.

    Every row is inserted to preserve the replay's layout; `alive` becomes the
    loss weight so dead-lane padding is inert. `alive` is the pre-step liveness,
    NOT the encoder's `valid`: `valid` additionally excludes the terminating
    step, and zeroing that row would discard precisely the transitions that tell
    the critic where to stop bootstrapping. Also returns the alive fraction.
    """

    horizon, num_envs = alive.shape
    per_step = lambda value: jnp.broadcast_to(value, (horizon,) + value.shape)
    episode_ids = -(
        branch_epoch * num_envs + jnp.arange(num_envs, dtype=jnp.int32) + 1
    )
    transitions = Transition(
        observations=observations,
        anchor_observations=per_step(policy_anchors),
        actions=actions,
        # Extrinsic reward is unused; the learner overwrites it with the
        # intrinsic reward for branch rows exactly as it does for live rows.
        rewards=jnp.zeros((horizon, num_envs), dtype=jnp.float32),
        # Termination cuts the bootstrap; truncation does not.
        masks=(~terminated).astype(jnp.float32),
        next_observations=next_observations,
        next_anchor_observations=per_step(policy_anchors),
        dones=terminated | truncated,
        skills=per_step(skills),
        next_skills=per_step(skills),
        # Dead-lane padding rows are inserted so the replay keeps its
        # vector-step-major layout (a future k steps later lives at
        # `index + k * num_envs`); a zero loss weight makes them inert.
        # Compacting them out instead would break that stride invariant.
        loss_weights=alive.astype(jnp.float32),
        episode_ids=per_step(episode_ids),
        timesteps=jnp.broadcast_to(
            jnp.arange(horizon, dtype=jnp.int32)[:, None], (horizon, num_envs)
        ),
    )
    flatten = lambda value: value.reshape((horizon * num_envs,) + value.shape[2:])
    return (
        jax.tree_util.tree_map(flatten, transitions),
        jnp.mean(alive.astype(jnp.float32)),
    )


def make_branch_trajectory_rollout(
    env: Any,
    action_fn: ActionFn,
    skill_sampler: SkillSampler,
    observation_fn: ObservationFn,
    *,
    horizon: int,
    targets_per_anchor: int = 1,
) -> Callable[[Any, MarginalSource, jax.Array], BranchTrajectory]:
    """Build one full-width counterfactual scan from immutable source states."""

    if horizon <= 0:
        raise ValueError("branch horizon must be positive")
    if not hasattr(env, "where"):
        raise ValueError(
            "branch trajectories require an environment with lane selection")
    num_envs = env.config.num_envs if hasattr(env, "config") else env.num_envs
    anchor_lane, _, _, _ = branch_anchor_layout(
        num_envs,
        1,
        targets_per_anchor,
    )
    if targets_per_anchor > 1 and not hasattr(env, "gather"):
        raise ValueError("shared-anchor branches require environment lane gathering")

    def run(
        params: Any,
        source: MarginalSource,
        key: jax.Array,
    ) -> BranchTrajectory:
        skill_key, rollout_key = jax.random.split(key)
        target_skills = skill_sampler(skill_key, num_envs)
        source_state = (
            env.gather(source.env_state, anchor_lane)
            if targets_per_anchor > 1
            else source.env_state
        )
        policy_anchors = source_state.obs
        encoder_anchors = observation_fn(policy_anchors)
        branch_state = _branch_start_state(source_state)
        alive = jnp.ones((num_envs,), dtype=jnp.bool_)

        def one_step(
            carry: tuple[Any, jax.Array, jax.Array],
            _: None,
        ) -> tuple[tuple[Any, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
            current, current_alive, current_key = carry
            next_key, action_root = jax.random.split(current_key)
            actions = action_fn(
                params,
                current.obs,
                policy_anchors,
                target_skills,
                jax.random.split(action_root, num_envs),
            )
            stepped = env.step(current, actions)
            finite = jnp.all(
                jnp.isfinite(stepped.obs).reshape((num_envs, -1)),
                axis=-1,
            )
            valid = current_alive & finite & (
                ~stepped.terminated) & (~stepped.truncated)
            next_state = env.where(valid, current, stepped)
            outcome = jnp.where(
                valid.reshape((num_envs,) + (1,) * (stepped.obs.ndim - 1)),
                stepped.obs,
                current.obs,
            )
            return (next_state, valid, next_key), (observation_fn(outcome), valid)

        _, (outcomes, valid) = jax.lax.scan(
            one_step,
            (branch_state, alive, rollout_key),
            None,
            length=horizon,
        )
        return BranchTrajectory(
            anchors=encoder_anchors,
            skills=target_skills,
            outcomes=outcomes,
            valid=valid,
        )

    return jax.jit(run)


def make_staggered_branch_trajectory_rollout(
    env: Any,
    action_fn: ActionFn,
    observation_fn: ObservationFn,
    *,
    horizon: int,
    chunk_steps: int,
    phase_groups: int,
    targets_per_anchor: int = 1,
    skill_sampler: SkillSampler | None = None,
) -> tuple[
    Callable[[StaggeredBranchState, MarginalSource, jax.Array], StaggeredBranchState],
    Callable[
        [Any, StaggeredBranchState, MarginalSource, jax.Array],
        tuple[StaggeredBranchState, BranchTrajectory, jax.Array, jax.Array, jax.Array],
    ],
]:
    """Build a full-width branch collector with one completing phase per chunk."""

    if min(horizon, chunk_steps, phase_groups) <= 0:
        raise ValueError("staggered branches need positive horizon, chunk, and phases")
    if horizon != chunk_steps * phase_groups:
        raise ValueError("staggered branch horizon must equal chunk_steps * phase_groups")
    num_envs = env.config.num_envs if hasattr(env, "config") else env.num_envs
    max_completions = math.ceil(num_envs / phase_groups)
    # Continuous skills: every relaunched lane draws a fresh random unit skill from
    # `skill_sampler`, which carries no anchor information.
    if skill_sampler is None:
        raise ValueError("staggered branches need a skill_sampler")
    if targets_per_anchor != 1:
        raise ValueError("continuous branch skills support targets_per_anchor == 1 only")
    # The skill-shift draw below treats the continuous prior as one "skill"; it fixes the
    # RNG stream and compiled program.
    num_skills = 1
    lane_indices = jnp.arange(num_envs, dtype=jnp.int32)
    group_indices = lane_indices * phase_groups // num_envs
    phase_offsets = group_indices * chunk_steps
    anchor_lane, block_indices, block_positions, num_blocks = branch_anchor_layout(
        num_envs,
        phase_groups,
        targets_per_anchor,
    )
    if targets_per_anchor > 1 and not hasattr(env, "gather"):
        raise ValueError("shared-anchor branches require environment lane gathering")

    def initialize(
        template: StaggeredBranchState,
        source: MarginalSource,
        target_skills: jax.Array,
    ) -> StaggeredBranchState:
        source_state = (
            env.gather(source.env_state, anchor_lane)
            if targets_per_anchor > 1
            else source.env_state
        )
        return template.replace(
            env_state=_branch_start_state(source_state),
            policy_anchors=source.env_state.obs[anchor_lane],
            encoder_anchors=observation_fn(source.env_state.obs)[anchor_lane],
            source_skills=source.skills[anchor_lane],
            skills=target_skills,
            outcomes=jnp.zeros_like(template.outcomes),
            valid=jnp.zeros_like(template.valid),
            progress=phase_offsets,
            genuine=phase_offsets == 0,
            alive=jnp.ones((num_envs,), dtype=jnp.bool_),
            active=jnp.asarray(True),
            skill_shift=jnp.asarray(0, dtype=jnp.int32),
            policy_outcomes=jnp.zeros_like(template.policy_outcomes),
            actions=jnp.zeros_like(template.actions),
            terminated=jnp.zeros_like(template.terminated),
            truncated=jnp.zeros_like(template.truncated),
            alive_rows=jnp.zeros_like(template.alive_rows),
            emit_index=template.emit_index,
            anchor_xy=_locomotion_xy(source.env_state, num_envs)[0][anchor_lane],
        )

    def run(
        params: Any,
        state: StaggeredBranchState,
        source: MarginalSource,
        key: jax.Array,
    ) -> tuple[
        StaggeredBranchState,
        BranchTrajectory,
        jax.Array,
        jax.Array,
        jax.Array,
        Transition,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        rollout_key, shift_key, menu_key = jax.random.split(key, 3)

        def one_step(
            carry: tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array],
            local_step: jax.Array,
        ) -> tuple[
            tuple[Any, jax.Array, jax.Array, jax.Array, jax.Array],
            None,
        ]:
            (
                current,
                alive,
                outcomes,
                valid_rows,
                policy_outcomes,
                action_rows,
                terminated_rows,
                truncated_rows,
                alive_rows,
                current_key,
            ) = carry
            next_key, action_root = jax.random.split(current_key)
            actions = action_fn(
                params,
                current.obs,
                state.policy_anchors,
                state.skills,
                jax.random.split(action_root, num_envs),
            )
            stepped = env.step(current, actions)
            finite = jnp.all(
                jnp.isfinite(stepped.obs).reshape((num_envs, -1)),
                axis=-1,
            )
            step_valid = alive & finite & (~stepped.terminated) & (~stepped.truncated)
            next_state = env.where(step_valid, current, stepped)
            outcome = jnp.where(
                step_valid.reshape(
                    (num_envs,) + (1,) * (stepped.obs.ndim - 1)
                ),
                stepped.obs,
                current.obs,
            )
            rows = state.progress + local_step
            outcomes = outcomes.at[rows, lane_indices].set(observation_fn(outcome))
            valid_rows = valid_rows.at[rows, lane_indices].set(step_valid)
            # `outcome` is the FULL observation; `observation_fn` above is the
            # encoder's narrower view. `alive` here is the PRE-step liveness, so
            # the terminating transition is kept for the critic.
            policy_outcomes = policy_outcomes.at[rows, lane_indices].set(outcome)
            action_rows = action_rows.at[rows, lane_indices].set(actions)
            terminated_rows = terminated_rows.at[rows, lane_indices].set(
                stepped.terminated
            )
            truncated_rows = truncated_rows.at[rows, lane_indices].set(
                stepped.truncated
            )
            alive_rows = alive_rows.at[rows, lane_indices].set(alive)
            return (
                next_state,
                step_valid,
                outcomes,
                valid_rows,
                policy_outcomes,
                action_rows,
                terminated_rows,
                truncated_rows,
                alive_rows,
                next_key,
            ), None

        (
            env_state,
            alive,
            outcomes,
            valid,
            policy_outcomes,
            action_rows,
            terminated_rows,
            truncated_rows,
            alive_rows,
            _,
        ), _ = jax.lax.scan(
            one_step,
            (
                state.env_state,
                state.alive,
                state.outcomes,
                state.valid,
                state.policy_outcomes,
                state.actions,
                state.terminated,
                state.truncated,
                state.alive_rows,
                rollout_key,
            ),
            jnp.arange(chunk_steps, dtype=jnp.int32),
        )
        next_progress = state.progress + chunk_steps
        finished = next_progress >= horizon
        completed = finished & state.genuine
        completed_count = jnp.sum(completed.astype(jnp.int32))
        completed_indices = jnp.nonzero(
            completed,
            size=max_completions,
            fill_value=0,
        )[0]
        branch = BranchTrajectory(
            anchors=state.encoder_anchors[completed_indices],
            skills=state.skills[completed_indices],
            outcomes=outcomes[:, completed_indices],
            valid=valid[:, completed_indices],
        )
        completed_source_skills = state.source_skills[completed_indices]

        # Policy rows for the lanes that finished this segment. Slots beyond
        # `completed_count` are padding from the fixed-size nonzero(); they are
        # zero-weighted rather than compacted so the replay keeps its
        # vector-step-major layout.
        slot_real = jnp.arange(max_completions, dtype=jnp.int32) < completed_count
        completed_anchors = state.policy_anchors[completed_indices]
        completed_outcomes = policy_outcomes[:, completed_indices]
        # The state entering row r is the outcome of row r - 1; row 0 enters at
        # the anchor. Storing only outcomes halves the resident history.
        completed_observations = jnp.concatenate(
            (completed_anchors[None], completed_outcomes[:-1]), axis=0
        )
        branch_transitions, branch_alive_fraction = branch_policy_transitions(
            policy_anchors=completed_anchors,
            skills=state.skills[completed_indices],
            observations=completed_observations,
            actions=action_rows[:, completed_indices],
            next_observations=completed_outcomes,
            terminated=terminated_rows[:, completed_indices],
            truncated=truncated_rows[:, completed_indices],
            alive=alive_rows[:, completed_indices] & slot_real[None, :],
            branch_epoch=state.emit_index,
        )

        # Endpoint spread contains both skill effects and stochastic action
        # noise. A paired same-anchor/same-skill evaluation is required before
        # interpreting it as skill differentiation.
        end_xy = _locomotion_xy(env_state, num_envs)[0]
        disp_vec = end_xy - state.anchor_xy
        disp_norm = jnp.linalg.norm(disp_vec, axis=-1)
        done_f = completed.astype(jnp.float32)
        skill_disp_mean = jnp.sum(disp_norm * done_f) / jnp.maximum(jnp.sum(done_f), 1.0)
        block_n = jnp.bincount(
            block_indices,
            weights=done_f,
            length=num_blocks,
        )
        block_sum = jnp.zeros((num_blocks, 2), dtype=disp_vec.dtype).at[
            block_indices
        ].add(disp_vec * done_f[:, None])
        block_mean = block_sum / jnp.maximum(block_n[:, None], 1.0)
        squared_residual = jnp.sum(
            jnp.square(disp_vec - block_mean[block_indices]),
            axis=-1,
        )
        block_var = jnp.zeros((num_blocks,), dtype=disp_vec.dtype).at[
            block_indices
        ].add(squared_residual * done_f) / jnp.maximum(block_n, 1.0)
        block_ok = (block_n > 1.0).astype(jnp.float32)
        skill_endpoint_spread = jnp.sum(jnp.sqrt(block_var + 1e-12) * block_ok) / jnp.maximum(
            jnp.sum(block_ok), 1.0
        )

        cycle_boundary = jnp.any(completed & (group_indices == 0))
        sampled_shift = jax.random.randint(shift_key, (), 0, num_skills)
        skill_shift = jnp.where(cycle_boundary, sampled_shift, state.skill_shift)
        rotated_skills = skill_sampler(menu_key, num_envs).astype(state.skills.dtype)
        reset_skills = jnp.where(state.genuine[:, None], rotated_skills, state.skills)
        reset_source = (
            env.gather(source.env_state, anchor_lane)
            if targets_per_anchor > 1
            else source.env_state
        )
        reset_state = _branch_start_state(reset_source)
        env_state = env.where(finished, env_state, reset_state)

        def reset_lanes(old: jax.Array, new: jax.Array) -> jax.Array:
            mask = finished.reshape(
                finished.shape + (1,) * (old.ndim - finished.ndim)
            )
            return jnp.where(mask, new, old)

        clear_outcomes = finished.reshape(
            (1, num_envs) + (1,) * (outcomes.ndim - 2)
        )
        clear_valid = finished.reshape((1, num_envs))
        next_state = state.replace(
            env_state=env_state,
            policy_anchors=reset_lanes(
                state.policy_anchors, source.env_state.obs[anchor_lane]
            ),
            encoder_anchors=reset_lanes(
                state.encoder_anchors,
                observation_fn(source.env_state.obs)[anchor_lane],
            ),
            source_skills=reset_lanes(
                state.source_skills, source.skills[anchor_lane]
            ),
            skills=reset_lanes(state.skills, reset_skills),
            outcomes=jnp.where(clear_outcomes, 0, outcomes),
            valid=jnp.where(clear_valid, False, valid),
            policy_outcomes=jnp.where(
                finished.reshape((1, num_envs) + (1,) * (policy_outcomes.ndim - 2)),
                0,
                policy_outcomes,
            ),
            actions=jnp.where(
                finished.reshape((1, num_envs) + (1,) * (action_rows.ndim - 2)),
                0,
                action_rows,
            ),
            terminated=jnp.where(clear_valid, False, terminated_rows),
            truncated=jnp.where(clear_valid, False, truncated_rows),
            alive_rows=jnp.where(clear_valid, False, alive_rows),
            emit_index=state.emit_index + 1,
            anchor_xy=reset_lanes(
                state.anchor_xy,
                _locomotion_xy(source.env_state, num_envs)[0][anchor_lane],
            ),
            progress=jnp.where(finished, 0, next_progress),
            genuine=jnp.where(finished, True, state.genuine),
            alive=jnp.where(finished, True, alive),
            skill_shift=skill_shift,
        )
        discarded_count = jnp.sum((finished & (~state.genuine)).astype(jnp.int32))
        return (
            next_state,
            branch,
            completed_source_skills,
            completed_count,
            discarded_count,
            branch_transitions,
            branch_alive_fraction,
            skill_disp_mean,
            skill_endpoint_spread,
        )

    return jax.jit(initialize), jax.jit(run)
