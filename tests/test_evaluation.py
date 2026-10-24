"""Matched evaluation keys and the evaluation-evidence gate."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from empowerment_warp.evaluation_checks import check_evaluation, check_recipe, source_hashes
from empowerment_warp.evaluation_protocol import evaluation_keys
from empowerment_warp.recipes import resolve

ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_keys_are_shared_across_methods_and_fresh_per_checkpoint():
    """Keys depend only on (environment, seed, step), so every method sees the same worlds."""
    first, again = evaluation_keys('ant', 0, 50_000_000), evaluation_keys('ant', 0, 50_000_000)
    later = evaluation_keys('ant', 0, 100_000_000)
    for a, b, c in zip(first, again, later):
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)
    assert not np.array_equal(first[0], evaluation_keys('ant', 1, 50_000_000)[0])


@pytest.fixture
def recipe():
    return resolve(ROOT / 'configs/craftax_classic/see_reset_memory.yaml')[0]


def _evidence(directory, step=0):
    """A valid Craftax evaluation record and its trace."""
    directory.mkdir(parents=True, exist_ok=True)
    keys = np.asarray(evaluation_keys('craftax:classic-symbolic', 0, step))
    skills = jax.random.normal(jnp.asarray(keys[2]), (128, 16))
    skills /= jnp.sqrt(jnp.square(skills).sum(-1, keepdims=True) + 1e-8)
    prefix = np.zeros((128, 22), bool); prefix[:, 0] = True
    final = prefix.copy(); final[:, 1] = True
    done = np.zeros((1000, 128), bool); done[-1] = True
    on = np.ones((1000, 128), bool); on[:100] = False
    data = dict(valid=np.ones((1001, 128), bool), dones=done, skill_on=on, skills=np.array(skills),
                reset_keys=np.asarray(jax.random.split(jnp.asarray(keys[0]), 128)),
                initial_achievements=np.zeros((128, 22), bool), prefix_achievements=prefix,
                final_achievements=final, total_achievement_counts=np.full(128, 2),
                potential_achievement_counts=np.ones(128), skill_achievement_counts=np.ones(128),
                episode_lengths=np.full(128, 1000))
    np.savez_compressed(directory / 'trace.npz', **data)
    row = {'env_steps': step, 'checkpoint_step': step, 'learner_updates': step // 131072,
           'evaluation/achievement_count': 2., 'evaluation/potential_policy/achievement_count': 1.,
           'evaluation/skill_policy/new_achievement_count': 1., 'evaluation/completed_episodes': 128.,
           'protocol': dict(name='craftax-128-fresh-switching-v2', engine='pqn',
                            environment='craftax:classic-symbolic', num_envs=128, prefix_steps=100,
                            episode_length=1000, contexts=128, skills_per_context=1,
                            sampling='fresh_prior_and_resets_per_checkpoint', backend='gpu',
                            deterministic=False, seed=0, rng_keys=keys.tolist(),
                            evaluation_source=source_hashes('pqn', 'craftax:classic-symbolic'),
                            trace=str(directory / 'trace.npz'))}
    path = directory / 'evaluation.json'
    path.write_text(json.dumps(row))
    return path, row, data


def test_main_recipes_satisfy_the_evaluation_contract():
    for env in ('ant', 'ant_soccer', 'builderbench_corner3', 'craftax_classic'):
        check_recipe(resolve(ROOT / 'configs' / env / 'see.yaml')[0])


def test_gate_accepts_valid_evidence(tmp_path, recipe):
    path, _, _ = _evidence(tmp_path)
    assert check_evaluation(path, recipe)['target'] == 0


@pytest.mark.parametrize('change,match', [
    ('wrong_total', 'recomputation'), ('fixed_skills', 'fresh prior skills'),
    ('wrong_phase', 'potential phase commands'), ('source', 'evaluation source')])
def test_gate_rejects_invalid_evidence(tmp_path, recipe, change, match):
    path, row, data = _evidence(tmp_path)
    if change == 'wrong_total':
        row['evaluation/achievement_count'] = 99
    elif change == 'fixed_skills':
        data['skills'][:] = data['skills'][0]
    elif change == 'wrong_phase':
        data['skill_on'][:100] = True
    elif change == 'source':
        row['protocol']['evaluation_source']['training/pqn/branch_eval.py'] = 'old-code'
    path.write_text(json.dumps(row))
    np.savez_compressed(tmp_path / 'trace.npz', **data)
    with pytest.raises(ValueError, match=match):
        check_evaluation(path, recipe)
