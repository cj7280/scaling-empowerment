"""Evaluate saved baseline checkpoints without changing their run directories."""
from dataclasses import fields
import json
from pathlib import Path


def evaluate_baseline(run_dir, output, *, checkpoint=None, target_step=None):
    from .training.baselines.config import EstimatorConfig
    from .training.baselines.recipe import Recipe
    from .training.baselines.runner import build, evaluate, evaluation_checkpoint
    from .checkpointing import CheckpointStore

    saved = json.loads((run_dir / 'config.json').read_text())
    names = {f.name for f in fields(Recipe)}
    values = {k: v for k, v in saved.items() if k in names}
    values['estimator'] = EstimatorConfig(**values['estimator'])
    recipe = Recipe(**values)
    state, functions = build(recipe)
    if checkpoint is None:
        state = CheckpointStore(run_dir / 'checkpoints').restore(state)
    else:
        import jax.numpy as jnp
        path = Path(checkpoint).resolve()
        if not path.is_dir() or not path.name.startswith('step_'):
            raise ValueError('checkpoint must name a retained evaluation step directory')
        model = CheckpointStore(path.parent).restore(evaluation_checkpoint(recipe, state),
                                                   int(path.name.removeprefix('step_')))
        if recipe.vector_gce_normalize_inputs:
            if 'estimator_norm' not in model:
                raise ValueError('normalized baseline checkpoint is missing estimator_norm')
            state['estimator_norm'] = model['estimator_norm']
        state['ppo'] = state['ppo'].replace(params=model['ppo_params'])
        state['controller'] = state['controller']._replace(observation_norm=model['observation_norm'])
        if recipe.method == 'gce_vector':
            state['estimator'] = state['estimator'].replace(params=model['estimator_params'])
        else:
            state['estimator'] = tuple(s.replace(params=p) for s,p in
                                       zip(state['estimator'],model['estimator_params']))
        state['update'] = jnp.asarray(int(model['env_steps']) // recipe.steps_per_update)
    output.mkdir(parents=True, exist_ok=True)
    return evaluate(recipe, state, functions, output, source_run=run_dir, target_step=target_step)
