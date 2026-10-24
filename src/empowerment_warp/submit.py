"""Submit a seed array using an immutable copy of the current source and recipes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .recipes import resolve
from .evaluation_checks import check_recipe, check_evidence, source_hashes, require


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--mem', default=os.environ.get('SLURM_MEM'))
    parser.add_argument('--time', default=os.environ.get('SLURM_TIME'))
    parser.add_argument('--exclude', default=os.environ.get('SBATCH_EXCLUDE'),
                        help='excluded Slurm nodes, passed explicitly to sbatch')
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--max-chunks', type=int)
    parser.add_argument('--seed', type=int, help='submit one selected seed, e.g. for a bounded smoke')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--evaluation-evidence', type=Path,
                        help='manifest of current evaluation outputs required for a full run')
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    config = args.config.resolve()
    relative = config.relative_to(root)
    recipe, _ = resolve(config, overrides=args.set)
    evaluation = check_recipe(recipe)
    evidence = None
    if evaluation is not None and args.max_chunks is None:
        if args.evaluation_evidence is None:
            parser.error('full runs require --evaluation-evidence; use a bounded --max-chunks smoke to collect it')
        evidence = check_evidence(args.evaluation_evidence, recipe)
    seeds = recipe['seeds'] if args.seed is None else [args.seed]
    if any(seed not in recipe['seeds'] for seed in seeds):
        parser.error('--seed must be one of the recipe seeds')
    worker = root / 'scripts/train.sbatch'
    subprocess.run(['bash', '-n', str(worker)], check=True)
    preflight_env = dict(os.environ, PREFLIGHT_ONLY='1', SLURM_ARRAY_TASK_ID=str(seeds[0]))
    subprocess.run(['bash', '-c', 'bash /dev/fd/3 "$1" "$2" "$3" 3<"$4"', '_',
                    str(root), str(config), str(root / 'unused-preflight-output'), str(worker)],
                   env=preflight_env, check=True)
    if not args.mem or not args.time:
        parser.error('supply --mem and --time from measured runs, or set them in .env')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.run_root:
        destination = args.run_root.resolve()
    else:
        if not os.environ.get('RUN_ROOT'):
            parser.error('set RUN_ROOT in .env or supply --run-root')
        destination = Path(os.environ['RUN_ROOT']).resolve() / f'{config.parent.name}-{config.stem}-{stamp}'
    source = destination / 'source'
    command = ['sbatch', '--parsable', f"--partition={os.environ.get('SLURM_PARTITION', 'all')}",
               f"--gres={os.environ.get('SLURM_GRES', 'gpu:1')}",
               f"--cpus-per-task={os.environ.get('SLURM_CPUS', '4')}",
               f'--mem={args.mem}', f'--time={args.time}',
               '--array=' + ','.join(map(str, seeds)),
               '--job-name=' + config.parent.name + '-' + config.stem,
               f'--output={destination}/slurm/%A_%a.out', '--export=ALL']
    if os.environ.get('SLURM_ACCOUNT'):
        command.append('--account=' + os.environ['SLURM_ACCOUNT'])
    if args.exclude:
        command.append('--exclude=' + args.exclude)
    command += [str(source / 'scripts/train.sbatch'), str(source), str(source / relative), str(destination)]
    for item in args.set:
        command += ['--set', item]
    if args.max_chunks is not None:
        if args.max_chunks < 1:
            parser.error('--max-chunks must be positive')
        command += ['--max-chunks', str(args.max_chunks)]
    receipt = dict(command=command, recipe=recipe, output=str(destination),
                   evaluation_contract=evaluation, evaluation_evidence=evidence,
                   bounded_smoke=args.max_chunks is not None)
    if args.dry_run:
        print(json.dumps(receipt, indent=2))
        return
    if destination.exists():
        raise FileExistsError(f'will not overwrite a submission: {destination}')
    (destination / 'slurm').mkdir(parents=True)
    for directory in ('src', 'configs', 'scripts'):
        shutil.copytree(root / directory, source / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'texture_cache*.pbz2'))
    for filename in ('pyproject.toml', 'uv.lock'):
        shutil.copy2(root / filename, source / filename)
    if evidence is not None:
        require('submitted evaluation source',
                source_hashes(evaluation['engine'], evaluation['environment'],
                              root=source / 'src/empowerment_warp'),
                evidence['evaluations'][0]['evaluation_source'])
        frozen_recipe, _ = resolve(source / relative, overrides=args.set)
        require('submitted resolved recipe', frozen_recipe, recipe)
    receipt['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    (source / 'working.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root))
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source.rglob('*')) if p.is_file()}
    (source / 'manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')
    # Keep a complete receipt even if sbatch is rejected.
    (destination / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
    launch_env = dict(os.environ)
    launch_env.pop('PREFLIGHT_ONLY', None)
    receipt['job_id'] = subprocess.check_output(command, text=True, env=launch_env).strip()
    (destination / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
