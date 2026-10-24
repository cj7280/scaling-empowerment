"""Every shipped recipe resolves, and unknown keys are rejected."""
from pathlib import Path

import pytest
import yaml

from empowerment_warp.recipes import apply_overrides, read_recipe, resolve

ROOT = Path(__file__).resolve().parents[1]
RECIPES = sorted((ROOT / 'configs').glob('*/*.yaml'))


@pytest.mark.parametrize('path', RECIPES, ids=lambda p: str(p.relative_to(ROOT / 'configs')))
def test_all_recipes_resolve(path):
    recipe, config = resolve(path)
    assert config is not None
    assert recipe['seeds'] and recipe['settings']


def test_unknown_override_is_rejected():
    settings = {'seed': 0, 'estimator': {'learning_rate': 0.001}}
    assert apply_overrides(settings, ['estimator.learning_rate=0.002'])['estimator']['learning_rate'] == 0.002
    assert settings['estimator']['learning_rate'] == 0.001
    with pytest.raises(ValueError, match='unknown override'):
        apply_overrides(settings, ['learning_rate=0.1'])


def test_unknown_recipe_and_setting_keys_are_rejected(tmp_path):
    base = yaml.safe_load((ROOT / 'configs/ant/see.yaml').read_text())
    (tmp_path / 'extra.yaml').write_text(yaml.safe_dump({**base, 'typo': 1}))
    with pytest.raises(ValueError, match='unknown recipe keys'):
        read_recipe(tmp_path / 'extra.yaml')
    base['settings']['not_a_setting'] = 1
    (tmp_path / 'setting.yaml').write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError, match='unknown ExperimentConfig settings'):
        resolve(tmp_path / 'setting.yaml')
