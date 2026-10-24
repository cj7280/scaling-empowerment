"""Matched continuous evaluations: fresh worlds per checkpoint, saved before rendering."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

PROTOCOL = 'continuous-128-fresh-v1'
NUM_WORLDS = 128
POTENTIAL_STEPS = 100
PRESETS = {'ant': 'past_center_ball', 'ant_soccer': 'closer_ball',
           'builderbench_corner3': 'standard'}


def evaluation_cadence(environment):
    return 5_000_000 if environment == 'builderbench_corner3' else 50_000_000


def evaluation_keys(environment, seed, target_step):
    """Stable across methods; independent across environments, seeds and checkpoints."""
    index = (tuple(PRESETS) + ('craftax:classic-symbolic',)).index(environment)
    key = jax.random.PRNGKey(20261004)
    for value in (index, int(seed), int(target_step) & 0xffffffff, int(target_step) >> 32):
        key = jax.random.fold_in(key, value)
    return tuple(jax.random.split(key, 3))


def make_rollout(env, action_fn, *, horizon=100):
    """Return first-episode transitions; terminal actions count, subsequent actions do not."""
    if horizon not in (100, 400):
        raise ValueError('evaluation requires 100 potential actions or 100 + 300 skill actions')
    if env.config.num_envs != NUM_WORLDS:
        raise ValueError('continuous evaluation requires exactly 128 worlds')
    builder = hasattr(env, '_object_positions')

    def positions(state):
        if builder:
            blocks = jax.vmap(env._object_positions)(state.data)
            return dict(gripper_xyz=state.obs[:, :3],
                        **{f'block_{i}_xyz': blocks[:, i] for i in range(3)})
        result = dict(agent_xy=state.data.qpos[:, :2])
        if env.soccer:
            result['ball_xy'] = state.data.qpos[:, -7:-5]
        return result

    @jax.jit
    def rollout(params, reset_key, action_key, skills):
        initial = env.reset(jax.random.split(reset_key, NUM_WORLDS))
        def step(carry, index):
            state, anchors, alive, rng = carry
            rng, key = jax.random.split(rng)
            marked = state._replace(terminated=~alive, truncated=jnp.zeros_like(alive))
            safe = env.where_done(marked, initial)
            z = jnp.where(index < POTENTIAL_STEPS, jnp.zeros_like(skills), skills)
            actions = action_fn(params, safe.obs, anchors, z,
                                jax.random.split(key, NUM_WORLDS))
            reached = env.step(safe, actions)
            done = reached.terminated | reached.truncated
            anchors = jnp.where(index < POTENTIAL_STEPS, reached.obs, anchors)
            row = dict(alive=alive, dones=done, observations=safe.obs,
                       reached_observations=reached.obs, actions=actions,
                       skill_on=jnp.linalg.norm(z, axis=-1) > 0,
                       rewards=reached.reward, **positions(reached))
            row.update({name: getattr(reached.data, name)[0] for name in
                        ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat')})
            return (reached, anchors, alive & ~done, rng), row
        _, arrays = jax.lax.scan(step, (initial, initial.obs,
            jnp.ones(NUM_WORLDS, bool), action_key), jnp.arange(horizon))
        arrays.update({f'initial_{name}': value for name, value in positions(initial).items()})
        arrays.update({f'initial_{name}': getattr(initial.data, name) for name in
                       ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat')})
        arrays['skills'] = skills
        arrays['control_dt'] = jnp.asarray(env.config.ctrl_dt)
        return arrays
    return rollout


def save_evaluation(output_directory, arrays, *, environment, seed, target_step,
                    checkpoint_step, reset_preset, source_run, skill_phase=False):
    from empowerment_warp.utils.evaluation_metrics import (
        COVERAGE_CELL, save_potential_policy_probe, state_coverage,
        builderbench_state_coverage, phase_displacement, _metric_statistics)
    from empowerment_warp.media.evaluation import render_evaluation
    from empowerment_warp.evaluation_checks import source_hashes
    if reset_preset != PRESETS[environment]:
        raise ValueError(f'{environment}: reset {reset_preset!r} differs from reference {PRESETS[environment]!r}')
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in jax.device_get(arrays).items()}
    alive = arrays['alive'].astype(bool)
    expected = (400 if skill_phase else 100, NUM_WORLDS)
    if alive.shape != expected or not alive[0].all():
        raise ValueError(f'invalid first-episode support: {alive.shape}, expected {expected}')
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise FloatingPointError('nonfinite evaluation trajectory')
    names = (['gripper_xyz', 'block_0_xyz', 'block_1_xyz', 'block_2_xyz']
             if environment == 'builderbench_corner3' else ['agent_xy', 'ball_xy'])
    names = [name for name in names if name in arrays]
    metadata = dict(protocol=PROTOCOL, environment=environment, seed=int(seed),
        engine='sac' if skill_phase else 'baseline', backend=jax.devices()[0].platform,
        sampling='fresh_prior_and_resets_per_checkpoint',
        evaluation_source=source_hashes('sac' if skill_phase else 'baseline', environment),
        target_env_steps=int(target_step), checkpoint_step=int(checkpoint_step),
        reset_preset=reset_preset, source_run=str(source_run), num_envs=NUM_WORLDS,
        horizon_steps=expected[0], potential_steps=100, policy_mode='stochastic',
        control_dt=float(arrays['control_dt']), render_lane=0,
        rng_keys=[np.asarray(k).tolist() for k in evaluation_keys(environment, seed, target_step)])
    source = output / 'trajectories.npz'
    np.savez_compressed(source, **arrays, metadata=json.dumps(metadata))
    potential = save_potential_policy_probe(output / 'metrics.json',
        {name: arrays[name][:100] for name in names},
        {name: arrays['initial_' + name] for name in names}, alive[:100],
        {name: COVERAGE_CELL[name] for name in names},
        checkpoint_step=checkpoint_step, seed=seed)
    potential.update(protocol=PROTOCOL, policy='potential policy', target_env_steps=int(target_step))
    summary = {}
    if environment == 'builderbench_corner3':
        blocks = np.stack([arrays[f'block_{i}_xyz'] for i in range(3)], axis=2)
        coverage = builderbench_state_coverage(blocks, arrays['gripper_xyz'], alive, onset=100)
        starts = np.stack([arrays[f'initial_block_{i}_xyz'] for i in range(3)], axis=1)
        phases = phase_displacement(blocks[..., :2], alive, starts[..., :2], onset=100)
        for phase, value in phases.items():
            summary[f'{phase}_cube_disp_xy'] = _metric_statistics(value.mean(-1))
    else:
        coverage = {name: state_coverage(arrays[name], alive, COVERAGE_CELL[name], onset=100)
                    for name in names}
        for name in names:
            for phase, value in phase_displacement(arrays[name], alive,
                    arrays['initial_' + name], onset=100).items():
                summary[f'{phase}_{name[:-3]}_disp_xy'] = _metric_statistics(value)
    if not skill_phase:
        # Standalone policies have no skill phase.
        summary = {k: v for k, v in summary.items() if not k.startswith('skill_')}
        coverage = {name: {k: v for k, v in values.items() if 'skill_phase' not in k}
                    for name, values in coverage.items()}
    payload = dict(metadata, status='passed', summary=summary,
        state_coverage=coverage, potential_policy=potential, per_lane_artifact=str(source),
        evaluation_transitions=int(alive.size), valid_transitions=int(alive.sum()))
    # Write metrics before rendering; a media failure must still fail the caller.
    path = output / 'metrics.json'
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    payload['media'] = render_evaluation(source, output / 'media')
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return payload


def evaluate_sac(run_dir, output, *, target_step=None, checkpoint=None):
    from dataclasses import replace
    from empowerment_warp.training.sac.evaluation import load_run
    from empowerment_warp.envs.factory import make_environment
    config, experiment, state = load_run(run_dir)
    if checkpoint is not None:
        from empowerment_warp.checkpointing import CheckpointStore
        path = Path(checkpoint).resolve()
        if not path.is_dir() or not path.name.startswith('step_'):
            raise ValueError('checkpoint must name a retained step directory')
        state = CheckpointStore(path.parent).restore(state, int(path.name.removeprefix('step_')))
    step = int(state.env_steps)
    target = step if target_step is None else int(target_step)
    env = make_environment(replace(config, num_envs=NUM_WORLDS, allow_reference_deviation=True))
    keys = evaluation_keys(config.environment, config.seed, target)
    skills = experiment.skill.sample_skills(keys[2], NUM_WORLDS)
    def act(params, obs, anchors, z, action_keys):
        return experiment.sac.sample_actions(params, obs, anchors, z, action_keys, deterministic=False)
    arrays = make_rollout(env, act, horizon=400)(state.sac, keys[0], keys[1], skills)
    return save_evaluation(output, arrays, environment=config.environment, seed=config.seed,
        target_step=target, checkpoint_step=step, reset_preset=config.antsoccer_preset,
        source_run=run_dir, skill_phase=True)
