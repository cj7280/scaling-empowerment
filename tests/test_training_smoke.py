"""One bounded GPU training run per engine: two chunks, then metrics and a checkpoint exist."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

SMOKES = {
    'sac_ant': ('configs/ant/see.yaml', [
        'num_envs=128', 'batch_size=256', 'hidden_dims=[64,64]', 'replay_capacity=100000',
        'updates_per_chunk=4', 'encoder_updates_per_chunk=4', 'checkpoint_every_segments=1',
        'log_every_segments=1']),
    'pqn_craftax': ('configs/craftax_classic/see.yaml', [
        'num_envs=64', 'pqn_hidden_size=64', 'encoder_hidden_dims=[64,64]', 'encoder_batch_size=256',
        'representation_archive_size=100000', 'rollout_steps=16', 'eval_at_start=false',
        'checkpoint_every_segments=1', 'log_every_segments=1']),
    'vim_ant': ('configs/ant/vim.yaml', ['num_envs=32', 'rollout_steps=16']),
}


@pytest.mark.parametrize('name', SMOKES)
def test_two_chunks_write_metrics_and_checkpoint(tmp_path, name):
    config, overrides = SMOKES[name]
    run_dir = tmp_path / name
    command = [sys.executable, '-m', 'empowerment_warp.cli', 'train', '--config', str(ROOT / config),
               '--run-dir', str(run_dir), '--max-chunks', '2']
    for item in overrides:
        command += ['--set', item]
    env = {**os.environ, 'WANDB_MODE': 'offline', 'WANDB_DIR': str(tmp_path)}
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text().splitlines() if line]
    assert rows
    assert any((run_dir / 'checkpoints').iterdir())
