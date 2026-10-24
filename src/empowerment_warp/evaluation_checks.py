"""Fail-closed checks for the standard evaluations and their saved evidence.

This module contains the expected protocol independently of the rollout builders.
Bounded smokes may collect missing evidence; full submissions must supply it.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

CONTINUOUS = {
    'ant': ('past_center_ball', 50_000_000, 'agent_to_center_xy/distance_mean'),
    'ant_soccer': ('closer_ball', 50_000_000, 'agent_to_ball_xy/distance_mean'),
    'builderbench_corner3': ('standard', 5_000_000, 'gripper_to_nearest_block_xy/distance_mean'),
}
CRAFTAX = 'craftax:classic-symbolic'


def require(label, actual, expected):
    if actual != expected:
        raise ValueError(f'evaluation contract: {label}: expected {expected!r}, got {actual!r}')


def field(mapping, key):
    if key not in mapping:
        raise ValueError(f'evaluation contract: missing required field {key!r}')
    return mapping[key]


def close(label, actual, expected):
    if not np.allclose(actual, expected, rtol=2e-6, atol=2e-6, equal_nan=False):
        raise ValueError(f'evaluation contract: {label}: values differ from saved-trajectory recomputation')


def source_hashes(engine, environment, *, root=None):
    """Hash the consumer code, including this checker; never certify old output afterward."""
    root = Path(__file__).resolve().parent if root is None else Path(root)
    paths = ['evaluation_checks.py', 'evaluation_protocol.py', 'agents/skill_classifier.py',
             'utils/networks.py', 'utils/objectives.py']
    if engine == 'pqn':
        paths += ['training/pqn/branch_eval.py', 'training/pqn/branch_protocol.py',
                  'training/pqn/env.py', 'training/pqn/runner.py', 'training/pqn/config.py',
                  'training/pqn/checkpoint.py', 'training/pqn/logging.py', 'envs/craftax_view.py',
                  'agents/see_pqn.py', 'utils/flax_utils.py', 'utils/sampling.py',
                  'envs/craftax_env.py', 'envs/wrappers.py']
    else:
        paths += ['training/sac/evaluation.py', 'utils/evaluation_metrics.py', 'envs/factory.py',
                  'media/evaluation.py',
                  'envs/builderbench/env.py' if environment == 'builderbench_corner3' else 'envs/locomotion/env.py']
        paths += (['agents/see_sac.py', 'utils/sampling.py', 'training/sac/runner.py',
                   'training/sac/config.py', 'training/sac/logging.py', 'training/sac/progress.py']
                  if engine == 'sac' else ['training/baselines/runner.py', 'training/baselines/config.py',
                                          'training/baselines/collector.py', 'training/baselines/replay.py',
                                          'training/baselines/probes.py', 'agents/ppo.py', 'agents/vim.py',
                                          'agents/latent_gce.py', 'checkpoint_evaluation.py'])
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def check_recipe(recipe):
    engine, cfg = field(recipe, 'engine'), field(recipe, 'settings')
    if engine not in ('sac', 'pqn', 'baseline'):
        return None  # Toy methods have their own benchmark protocol.
    environment = field(cfg, 'environment')
    if engine == 'pqn':
        require('environment', environment, CRAFTAX)
        for key, value in {'eval_contexts':128, 'eval_skills_per_context':1,
                           'eval_at_start':True, 'eval_media_skill_count':4,
                           'eval_video_episodes':1, 'eval_fixed_reset_seed':0,
                           'discrete_skills':False}.items():
            require(key, field(cfg,key), value)
        require('evaluation horizon', field(cfg,'eval_episode_length') or field(cfg,'episode_length'), 1000)
        cadence = field(cfg,'eval_every_segments') * field(cfg,'num_envs') * field(cfg,'rollout_steps')
        require('metric cadence (live transitions)', cadence, 10_092_544)
        require('media cadence target (live transitions)', field(cfg,'eval_media_interval'), 50_000_000)
        return dict(engine=engine,environment=environment,worlds=128,horizon=1000,
                    prefix=100,cadence=cadence,media_cadence=math.ceil(50_000_000/cadence)*cadence,
                    protocol='craftax-128-fresh-switching-v2')
    if environment not in CONTINUOUS:
        raise ValueError(f'evaluation contract: unsupported environment {environment!r}')
    preset, cadence, _ = CONTINUOUS[environment]
    require('reset preset', field(cfg,'antsoccer_preset' if engine == 'sac' else 'reset_preset'),preset)
    require('metric cadence (recorded transitions)',
            field(cfg,'progress_eval_every_env_steps' if engine == 'sac' else 'eval_every_env_steps'),cadence)
    if engine == 'sac':
        require('episode_length',field(cfg,'episode_length'),400)
        require('discrete_skills',field(cfg,'discrete_skills'),False)
    return dict(engine=engine,environment=environment,worlds=128,
                horizon=400 if engine == 'sac' else 100,prefix=100,cadence=cadence,
                media_cadence=cadence,protocol='continuous-128-fresh-v1')


def _array(data, key):
    value = np.asarray(field(data,key))
    if value.dtype.kind in 'fc' and not np.isfinite(value).all():
        raise ValueError(f'evaluation contract: nonfinite {key}')
    return value


def _file(path, label):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f'evaluation contract: missing or empty {label}: {path}')
    return path


def check_evaluation(path, recipe):
    """Read back real metric files and independently recompute their headline values."""
    spec = check_recipe(recipe)
    if spec is None:
        raise ValueError('evaluation evidence is only supported for the benchmark environments')
    path = _file(path,'evaluation JSON')
    row = json.loads(path.read_text())
    pqn = spec['engine'] == 'pqn'
    meta = field(row,'protocol') if pqn else row
    require('protocol version',field(meta,'name' if pqn else 'protocol'),spec['protocol'])
    require('engine',field(meta,'engine'),spec['engine'])
    require('environment',field(meta,'environment'),spec['environment'])
    require('worlds',field(meta,'num_envs'),128)
    require('potential actions',field(meta,'prefix_steps' if pqn else 'potential_steps'),100)
    require('horizon',field(meta,'episode_length' if pqn else 'horizon_steps'),spec['horizon'])
    require('sampling rule',field(meta,'sampling'),'fresh_prior_and_resets_per_checkpoint')
    expected_source = source_hashes(spec['engine'],spec['environment'])
    recorded_source = field(meta,'evaluation_source')
    require('evaluation source file set',sorted(recorded_source),sorted(expected_source))
    for name, digest in expected_source.items():
        require('evaluation source '+name,recorded_source[name],digest)
    require('GPU execution',field(meta,'backend'),'gpu')
    if pqn:
        # Configs written before this flag existed reset memory at the switch.
        require('switch memory', meta.get('pqn_preserve_memory_at_switch', False),
                recipe['settings'].get('pqn_preserve_memory_at_switch', False))
        require('deterministic policy',field(meta,'deterministic'),False)
        require('independent worlds',field(meta,'contexts'),128)
        require('skills per world',field(meta,'skills_per_context'),1)
        target = int(field(row,'env_steps'))
    else:
        require('policy mode',field(meta,'policy_mode'),'stochastic')
        require('reset preset',field(meta,'reset_preset'),CONTINUOUS[spec['environment']][0])
        target = int(field(meta,'target_env_steps'))
    seed = int(field(meta,'seed'))
    checkpoint_step = int(field(row,'checkpoint_step'))
    if checkpoint_step < target:
        raise ValueError('evaluation contract: checkpoint precedes evaluation target')
    if target == 0:
        require('pretraining checkpoint step',checkpoint_step,0)
        if pqn:
            require('pretraining learner updates',field(row,'learner_updates'),0)
    from .evaluation_protocol import evaluation_keys
    import jax
    import jax.numpy as jnp
    keys = np.asarray(evaluation_keys(spec['environment'],seed,target))
    require('RNG keys',field(meta,'rng_keys'),keys.tolist())
    trace = _file(field(meta,'trace' if pqn else 'per_lane_artifact'),'trajectory archive')
    with np.load(trace,allow_pickle=False) as data:
        alive = _array(data,'valid')[1:] if pqn else _array(data,'alive')
        require('trajectory shape',alive.shape,(spec['horizon'],128))
        require('validity dtype',alive.dtype.kind,'b')
        require('initial live worlds',bool(alive[0].all()),True)
        done = _array(data,'dones')
        require('termination shape',done.shape,alive.shape)
        require('first-episode masking',bool(np.array_equal(alive[1:],alive[:-1] & ~done[:-1])),True)
        skill_on = _array(data,'skill_on')
        require('command phase shape',skill_on.shape,alive.shape)
        require('potential phase commands',bool(skill_on[:100].any()),False)
        require('skill phase commands',bool(skill_on[100:].all()),True)
        if spec['engine'] != 'baseline':
            skills = _array(data,'skills')
            require('skill population',skills.shape[0],128)
            expected = jax.random.normal(jnp.asarray(keys[2]),skills.shape)
            expected /= jnp.sqrt(jnp.square(expected).sum(-1,keepdims=True)+1e-8)
            close('fresh prior skills',skills,np.asarray(expected))
        if pqn:
            require('reset keys',_array(data,'reset_keys').tolist(),
                    np.asarray(jax.random.split(jnp.asarray(keys[0]),128)).tolist())
            initial,prefix,final = (_array(data,k).astype(bool) for k in
                ('initial_achievements','prefix_achievements','final_achievements'))
            require('achievement shape',final.shape,(128,22))
            require('persistent achievements',bool(np.any(initial & ~prefix) or np.any(prefix & ~final)),False)
            counts={'total':(final & ~initial).sum(-1),'potential':(prefix & ~initial).sum(-1),
                    'skill':(final & ~prefix).sum(-1)}
            for phase,key in [('total','evaluation/achievement_count'),
                              ('potential','evaluation/potential_policy/achievement_count'),
                              ('skill','evaluation/skill_policy/new_achievement_count')]:
                close(key,field(row,key),counts[phase].mean())
                close(phase+' saved counts',_array(data,phase+'_achievement_counts'),counts[phase])
            close('episode lengths',_array(data,'episode_lengths'),alive.sum(0))
            require('scored episodes',field(row,'evaluation/completed_episodes'),128.)
        else:
            last = alive[:100].sum(0).astype(int)-1
            lanes = np.arange(128)
            endpoint=lambda name:_array(data,name)[last,lanes]
            if spec['environment']=='ant':
                distances=np.linalg.norm(endpoint('agent_xy')-np.array([10.,10.]),axis=-1)
            elif spec['environment']=='ant_soccer':
                distances=np.linalg.norm(endpoint('agent_xy')-endpoint('ball_xy'),axis=-1)
            else:
                blocks=np.stack([endpoint(f'block_{i}_xyz')[:,:2] for i in range(3)],axis=1)
                distances=np.linalg.norm(blocks-endpoint('gripper_xyz')[:,None,:2],axis=-1).min(1)
            metric=CONTINUOUS[spec['environment']][2]
            close(metric,field(field(field(row,'potential_policy'),'metrics'),metric),distances.mean())
            if spec['engine']=='sac':
                summary=field(row,'summary')
                for phase in ('potential','skill'):
                    if not any(k.startswith(phase+'_') for k in summary):
                        raise ValueError(f'evaluation contract: missing {phase} displacement summary')
                if not field(row,'state_coverage'):
                    raise ValueError('evaluation contract: missing state coverage')
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                trace=str(trace.resolve()),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                target=target,seed=seed,spec=spec,evaluation_source=expected_source)


def check_evidence(evidence, recipe):
    """Require two current-source metric events plus locally verifiable review media."""
    evidence = _file(evidence,'evaluation evidence manifest')
    manifest=json.loads(evidence.read_text())
    reports=[check_evaluation(p,recipe) for p in field(manifest,'evaluations')]
    if len(reports)<2 or len({r['target'] for r in reports})<2:
        raise ValueError('evaluation contract: require at least two distinct evaluation checkpoints')
    if len({r['seed'] for r in reports})!=1:
        raise ValueError('evaluation contract: evidence checkpoints must share the training seed')
    if min(r['target'] for r in reports)!=0:
        raise ValueError('evaluation contract: missing pretraining evaluation at step zero')
    media=field(manifest,'media')
    if not media:
        raise ValueError('evaluation contract: missing review media')
    by_path = {r['path']: r for r in reports}
    roles = set()
    for item in media:
        p=_file(field(item,'path'),'review media')
        require('media hash',hashlib.sha256(p.read_bytes()).hexdigest(),field(item,'sha256'))
        role = field(item,'role')
        if role not in ('fixed_diagnostic','scored_trajectory'):
            raise ValueError('evaluation contract: unknown media role')
        evaluation = str(Path(field(item,'evaluation')).resolve())
        if evaluation not in by_path:
            raise ValueError('evaluation contract: media must refer to a checked evaluation')
        row = json.loads(Path(evaluation).read_text())
        if role == 'scored_trajectory':
            saved = field(row,'media')
            require('scored video',str(p.resolve()),str(Path(field(saved,'video')).resolve()))
            provenance = json.loads(_file(field(saved,'metadata'),'media provenance').read_text())
            require('scored media trace hash',field(provenance,'source_sha256'),by_path[evaluation]['trace_sha256'])
            require('scored media resimulation',field(provenance,'new_simulation'),False)
        else:
            figures = field(row,'figures')
            require('diagnostic metric contribution',field(field(figures,'protocol'),'contributes_to_metrics'),False)
            def paths(value):
                if isinstance(value,dict):
                    return [p for v in value.values() for p in paths(v)]
                if isinstance(value,list):
                    return [p for v in value for p in paths(v)]
                return [str(Path(value).resolve())] if isinstance(value,str) else []
            if str(p.resolve()) not in paths(field(figures,'saved_paths')):
                raise ValueError('evaluation contract: video not in evaluation media manifest')
        roles.add(role)
    required_role = 'fixed_diagnostic' if reports[0]['spec']['engine']=='pqn' else 'scored_trajectory'
    if required_role not in roles:
        raise ValueError(f'evaluation contract: missing {required_role} media')
    return dict(status='passed',manifest=str(evidence.resolve()),
                manifest_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),evaluations=reports)
