"""Named scientific recipes, resolved before constructing a simulator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
import json
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict, changes: dict) -> dict:
    result = deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def read_recipe(path: str | Path) -> dict:
    """Read a complete recipe or one override of a complete base."""
    path = Path(path).resolve()
    with path.open() as handle:
        recipe = yaml.safe_load(handle)
    if not isinstance(recipe, dict):
        raise ValueError(f"recipe must be a mapping: {path}")
    allowed = {"engine", "seeds", "settings", "description", "extends"}
    if extra := recipe.keys() - allowed:
        raise ValueError(f"unknown recipe keys: {sorted(extra)}")
    if "extends" in recipe:
        base_path = path.parent / recipe.pop("extends")
        with base_path.open() as handle:
            base = yaml.safe_load(handle)
        if not isinstance(base, dict) or "extends" in base:
            raise ValueError("a recipe may extend one complete base, without further inheritance")
        # Reuse validation of the complete base.
        base = read_recipe(base_path)
        recipe = _merge(base, recipe)
    if recipe.get("engine") not in {"sac", "pqn", "baseline", "toy_baseline"}:
        raise ValueError("engine must be sac, pqn, baseline, or toy_baseline")
    if not isinstance(recipe.get("seeds"), list) or not recipe["seeds"]:
        raise ValueError("seeds must be a nonempty list")
    if any(type(seed) is not int or seed < 0 for seed in recipe["seeds"]):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(recipe["seeds"])) != len(recipe["seeds"]):
        raise ValueError("seeds must be distinct")
    if not isinstance(recipe.get("settings"), dict):
        raise ValueError("settings must be a mapping")
    return recipe


def apply_overrides(settings: dict, overrides: list[str]) -> dict:
    result = deepcopy(settings)
    for item in overrides:
        key, separator, raw = item.partition("=")
        if not separator:
            raise ValueError(f"override must be key=value: {item}")
        node = result
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise ValueError(f"unknown override: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"unknown override: {key}")
        node[parts[-1]] = yaml.safe_load(raw)
    return result


def _construct(cls, settings):
    unknown = settings.keys() - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**settings)


def build_config(engine: str, settings: dict):
    """Construct the existing engine's typed config; do not create an environment."""
    settings = deepcopy(settings)
    if engine == "sac":
        from .training.sac.config import ExperimentConfig
        return _construct(ExperimentConfig, settings)
    if engine == "pqn":
        from .training.pqn.config import PotentialPQNRunConfig
        return _construct(PotentialPQNRunConfig, settings)
    if engine == "baseline":
        from .training.baselines.config import EstimatorConfig
        from .training.baselines.recipe import Recipe
        if "estimator" in settings:
            settings["estimator"] = _construct(EstimatorConfig, settings["estimator"])
        return _construct(Recipe, settings)
    if engine == "toy_baseline":
        from .baselines.toy_baselines import config_from_dict
        return config_from_dict(settings)
    raise ValueError(f"unknown engine: {engine}")


def resolve(path, *, overrides=(), seed=None, run_dir=None) -> tuple[dict, Any]:
    recipe = read_recipe(path)
    settings = apply_overrides(recipe["settings"], list(overrides))
    if seed is not None:
        settings["seed"] = seed
    if run_dir is not None:
        settings["output_dir" if recipe["engine"] == "toy_baseline" else "run_dir"] = str(Path(run_dir).resolve())
    config = build_config(recipe["engine"], settings)
    resolved = asdict(config) if is_dataclass(config) else dict(config)
    if recipe['engine'] == 'baseline':
        resolved['ppo'] = config.resolved()['ppo']
    # JSON types also normalize tuples before comparison with saved run configs.
    recipe["settings"] = json.loads(json.dumps(resolved))
    return recipe, config
