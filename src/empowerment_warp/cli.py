"""Train, inspect recipes, and evaluate checkpoints."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

from empowerment_warp.recipes import resolve


def _train(args):
    resolved, config = resolve(args.config, overrides=args.set, seed=args.seed, run_dir=args.run_dir)
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(p.name not in {"slurm_metadata.txt", "source_snapshot.json"} for p in run_dir.iterdir()) and not args.resume:
        raise ValueError(f'run already exists; choose another directory or --resume: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        (run_dir / 'recipe.json').write_text(json.dumps(resolved, indent=2) + '\n')
    engine = resolved['engine']
    if engine == 'sac':
        from empowerment_warp.training.sac.runner import train
        train(config, resume=args.resume, stop_after_segments=args.max_chunks)
    elif engine == 'pqn':
        from empowerment_warp.training.pqn.runner import run_potential_pqn_training
        if args.max_chunks is not None:
            config = replace(config, max_chunks=args.max_chunks)
        run_potential_pqn_training(config, resume=args.resume)
    elif engine == 'baseline':
        from empowerment_warp.training.baselines.runner import run
        config = replace(config, resume=args.resume, max_updates=args.max_chunks or config.max_updates)
        run(config)
    elif engine == 'toy_baseline':
        from empowerment_warp.baselines.toy_baselines import run
        config = replace(config, wandb_mode=os.environ.get('WANDB_MODE', 'disabled'),
                         wandb_project=os.environ.get('WANDB_PROJECT', 'empowerment-warp'),
                         wandb_group='', wandb_name=run_dir.name)
        if args.resume:
            config = replace(config, resume=str(run_dir))
        resolved['settings'] = asdict(config)
        if not args.resume:
            (run_dir / 'recipe.json').write_text(json.dumps(resolved, indent=2) + '\n')
        run(config, max_updates=args.max_chunks)


def _evaluation_options(args, supported):
    for name in ('checkpoint', 'eval_seed', 'num_envs', 'steps', 'contexts',
                 'skills_per_context', 'policy_mode', 'media', 'target_step'):
        if getattr(args, name) is not None and getattr(args, name) is not False and name not in supported:
            raise ValueError(f'--{name.replace("_", "-")} is not supported by this checkpoint evaluator')


def _output_directory(run_dir, output):
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    progress = run_dir / 'progress'
    if output == run_dir or output in run_dir.parents or (
        run_dir in output.parents and progress not in output.parents
    ):
        raise ValueError('output must be outside the source run or inside its progress directory')
    return run_dir, output


def _branch_media(args):
    run_dir, output = _output_directory(args.run_dir, args.output)
    saved = json.loads((run_dir / 'config.json').read_text())
    options = {'upload_wandb': False}
    for name in ('prefix_steps', 'horizon_steps', 'skill_count', 'checkpoint'):
        if getattr(args, name) is not None:
            options[name] = getattr(args, name)
    if saved['environment'] == 'builderbench_corner3':
        from empowerment_warp.media.builderbench import create_builderbench_branch_media
        result = create_builderbench_branch_media(run_dir, output, policy_mode=args.policy_mode, **options)
    elif saved['environment'] in {'ant', 'ant_soccer'}:
        from empowerment_warp.media.locomotion import create_locomotion_branch_media
        result = create_locomotion_branch_media(
            run_dir, output, deterministic=args.policy_mode == 'deterministic', **options)
    else:
        raise ValueError('branch-media supports Ant, AntSoccer, and BuilderBench checkpoints')
    print(json.dumps(result))


def _evaluate(args):
    run_dir, output = _output_directory(args.run_dir, args.output)
    config_path = run_dir / 'config.json'
    if not config_path.exists():
        config_path = run_dir / 'resolved_config.json'
    saved = json.loads(config_path.read_text())
    if isinstance(saved.get('environment'), dict):
        _evaluation_options(args, {'checkpoint'})
        from empowerment_warp.baselines.toy_baselines import config_from_dict, evaluate_checkpoint
        saved['output_dir'] = str(output)
        config = config_from_dict(saved)
        result = evaluate_checkpoint(config, args.checkpoint or run_dir)
        (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    elif saved.get('rl_algorithm') == 'potential-pqn':
        _evaluation_options(args, {'checkpoint', 'eval_seed', 'contexts', 'skills_per_context', 'media'})
        from empowerment_warp.training.pqn.branch_eval import evaluate_checkpoint
        candidates = sorted((run_dir / 'checkpoints').glob('params_*.pkl'))
        if not args.checkpoint and not candidates:
            raise FileNotFoundError(f'no parameter checkpoint under {run_dir}')
        checkpoint = Path(args.checkpoint) if args.checkpoint else candidates[-1]
        output.mkdir(parents=True, exist_ok=True)
        result = evaluate_checkpoint(run_dir, checkpoint, seed=args.eval_seed,
                                    contexts=args.contexts, skills_per_context=args.skills_per_context,
                                    media=args.media, output_dir=output)
        (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    elif saved.get('algorithm') == 'potential':
        _evaluation_options(args, {'target_step', 'checkpoint'})
        from empowerment_warp.evaluation_protocol import evaluate_sac
        print(json.dumps(evaluate_sac(run_dir, output, target_step=args.target_step, checkpoint=args.checkpoint)))
    elif saved.get('method') in {'vim', 'gce_vector'}:
        _evaluation_options(args, {'checkpoint', 'target_step'})
        from empowerment_warp.checkpoint_evaluation import evaluate_baseline
        print(json.dumps(evaluate_baseline(run_dir, output, checkpoint=args.checkpoint, target_step=args.target_step)))
    else:
        raise ValueError(f'unsupported checkpoint configuration: {run_dir}')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='empowerment-warp')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('resolve', 'train'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True)
        command.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
        command.add_argument('--seed', type=int, default=None if name == 'resolve' else 0)
        command.add_argument('--run-dir', required=name == 'train')
        if name == 'train':
            command.add_argument('--resume', action='store_true')
            command.add_argument('--max-chunks', type=int, help='stop early without shortening the training schedule')
    commands.add_parser('doctor')
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('run_dir')
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--target-step', type=int, help='evaluation target used for reproducible matched sampling')
    evaluate.add_argument('--checkpoint', help='PQN or toy-baseline checkpoint; continuous SAC/baselines use their latest checkpoint')
    evaluate.add_argument('--eval-seed', type=int)
    evaluate.add_argument('--num-envs', type=int)
    evaluate.add_argument('--steps', type=int)
    evaluate.add_argument('--contexts', type=int)
    evaluate.add_argument('--skills-per-context', type=int)
    evaluate.add_argument('--policy-mode', choices=('stochastic', 'deterministic'))
    evaluate.add_argument('--media', action='store_true')
    media = commands.add_parser('branch-media', help='render shared-prefix skill branches from a SAC checkpoint')
    media.add_argument('run_dir')
    media.add_argument('--output', required=True)
    media.add_argument('--policy-mode', choices=('stochastic', 'deterministic'), default='stochastic')
    media.add_argument('--prefix-steps', type=int)
    media.add_argument('--horizon-steps', type=int)
    media.add_argument('--skill-count', type=int)
    media.add_argument('--checkpoint', help='retained SAC step directory; defaults to latest')
    record = commands.add_parser('record')
    record.add_argument('run_dir')
    record.add_argument('--output', required=True)
    record.add_argument('--steps', type=int, required=True)
    record.add_argument('--lane', type=int, default=0)
    args = parser.parse_args(argv)
    if getattr(args, 'max_chunks', None) is not None and args.max_chunks < 1:
        parser.error('--max-chunks must be positive')
    if args.command == 'resolve':
        recipe, _ = resolve(args.config, overrides=args.set, seed=args.seed, run_dir=args.run_dir)
        print(json.dumps(recipe, indent=2))
    elif args.command == 'train':
        _train(args)
    elif args.command == 'evaluate':
        _evaluate(args)
    elif args.command == 'branch-media':
        _branch_media(args)
    elif args.command == 'record':
        from empowerment_warp.training.sac.evaluation import record_trace
        print(record_trace(args.run_dir, args.output, steps=args.steps, lane=args.lane))
    else:
        import jax
        from importlib.metadata import version
        print(json.dumps({'devices': [str(d) for d in jax.devices()],
                          'versions': {k: version(k) for k in ('jax', 'flax', 'mujoco', 'mujoco-mjx')}}, indent=2))


if __name__ == '__main__':
    main()
