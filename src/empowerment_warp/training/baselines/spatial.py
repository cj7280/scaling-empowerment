"""Small spatial evaluation records; no policy, learner, or simulation changes."""
import json

import numpy as np


def spatial_metrics(environment, observations, reached, dones, horizon=100):
    """Return rollout-average distances, first-episode endpoints, and compact positions."""
    obs, after, done = np.asarray(observations), np.asarray(reached), np.asarray(dones, dtype=bool)
    assert after.shape[:2] == done.shape and len(after) >= horizon
    prefix_done = done[:horizon]
    last = np.where(prefix_done.any(axis=0), prefix_done.argmax(axis=0), horizon - 1)
    lanes = np.arange(done.shape[1])
    positions = {}
    if environment == 'builderbench_corner3':
        agent = after[..., :2].astype(float)
        blocks = after[..., 10:19].reshape(*after.shape[:2], 3, 3)[..., :2].astype(float)
        distances = np.linalg.norm(blocks - agent[..., None, :], axis=-1).min(axis=-1)
        row = {'gripper_nearest_block_distance_xy_mean': float(distances.mean())}
        positions['gripper_xyz'] = np.concatenate([obs[:1, :, :3], after[:horizon, :, :3]])
        positions['block_xyz'] = np.concatenate([obs[:1, :, 10:19], after[:horizon, :, 10:19]]).reshape(horizon + 1, len(lanes), 3, 3)
    else:
        assert environment in ('ant', 'ant_soccer')
        agent = after[..., :2].astype(float)
        center = np.linalg.norm(agent - [10., 10.], axis=-1)
        row = {'agent_center_distance_xy_mean': float(center.mean())}
        positions['agent_xy'] = np.concatenate([obs[:1, :, :2], after[:horizon, :, :2]])
        if environment == 'ant_soccer':
            distances = np.linalg.norm(agent - after[..., 15:17], axis=-1)
            row['agent_ball_distance_xy_mean'] = float(distances.mean())
            positions['ball_xy'] = np.concatenate([obs[:1, :, 15:17], after[:horizon, :, 15:17]])
        else:
            distances = center
    endpoint = distances[last, lanes]
    assert np.isfinite(distances).all() and np.isfinite(endpoint).all()
    row.update(potential_endpoint_distance_xy_mean=float(endpoint.mean()),
               potential_endpoint_horizon_actions=horizon,
               potential_endpoint_early_termination_count=int((last < horizon - 1).sum()))
    positions['dones'] = prefix_done
    positions['endpoint_distances_xy'] = endpoint
    positions['endpoint_actions'] = last + 1
    return row, positions


def save_spatial_evaluation(output, environment, mode, step, observations, reached, dones):
    row, positions = spatial_metrics(environment, observations, reached, dones)
    directory = output / 'spatial'
    directory.mkdir(exist_ok=True)
    positions['metadata'] = json.dumps(dict(environment=environment, policy_mode=mode,
        env_steps=step, source_run=str(output), horizon_actions=100, units='meters',
        endpoint='action 100 or first termination; XY distance to center, ball, or nearest block'))
    np.savez_compressed(directory / f'{mode}_step_{step:012d}.npz', **positions)
    return {f'{mode}/spatial/{key}': value for key, value in row.items()}
