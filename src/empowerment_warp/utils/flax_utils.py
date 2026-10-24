"""Flax train-state helpers for the recurrent PQN controller.

``nonpytree_field``, ``ModuleDict`` and ``TrainState`` give the PQN agent its
optimizer/param-group semantics. The SAC agent uses a different ``flax.struct``
pattern. Checkpointing is handled by the PQN runner, not here.
"""

import functools
from typing import Any, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import optax


nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
        param_labels: Optional parameter-group labels matching an
            ``optax.multi_transform`` tree. This is static metadata and is not
            serialized in checkpoints.
        step_group: Optional group whose scoped updates advance ``step``.
            ``None`` makes every update advance the shared step.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any
    param_labels: Any = nonpytree_field(default=None)
    step_group: Any = nonpytree_field(default=None)

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, *, active_groups=None, **kwargs):
        """Apply gradients, optionally isolating named optimizer groups.

        ``active_groups=None`` is the backward-compatible full update. A scoped
        update requires ``param_labels`` and a root
        ``optax.multi_transform``/``optax.partition`` state. Inactive parameters
        and their complete optimizer states are reused verbatim, so momentum,
        variance estimates, and group-local schedule counters do not advance.

        ``step_group`` controls the shared ``step`` clock for scoped updates.
        For example, ``step_group='policy'`` keeps encoder-only updates from
        advancing a policy annealing schedule. Unscoped updates always advance
        the step exactly once.
        """
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        if active_groups is None:
            new_params = optax.apply_updates(self.params, updates)
            new_step = self.step + 1
        else:
            if self.param_labels is None:
                raise ValueError(
                    'Scoped optimizer updates require TrainState.param_labels.'
                )
            groups = (
                frozenset((active_groups,))
                if isinstance(active_groups, (str, bytes))
                else frozenset(active_groups)
            )
            if not groups:
                raise ValueError('active_groups must contain at least one group.')
            if jax.tree_util.tree_structure(self.param_labels) != jax.tree_util.tree_structure(
                self.params
            ):
                raise ValueError('TrainState.param_labels must match the parameter tree.')

            available_groups = frozenset(jax.tree_util.tree_leaves(self.param_labels))
            if self.step_group is not None and self.step_group not in available_groups:
                raise ValueError(
                    f'Unknown step_group {self.step_group!r}; available groups are '
                    f'{sorted(available_groups, key=str)}.'
                )
            unknown_groups = groups - available_groups
            if unknown_groups:
                raise ValueError(
                    f'Unknown active optimizer groups {sorted(unknown_groups, key=str)}; '
                    f'available groups are {sorted(available_groups, key=str)}.'
                )

            old_inner_states = getattr(self.opt_state, 'inner_states', None)
            new_inner_states = getattr(new_opt_state, 'inner_states', None)
            if not isinstance(old_inner_states, Mapping) or not isinstance(
                new_inner_states, Mapping
            ):
                raise ValueError(
                    'Scoped optimizer updates require a root '
                    'optax.multi_transform/optax.partition optimizer state.'
                )
            if set(old_inner_states) != set(new_inner_states):
                raise ValueError('Optimizer group state keys changed during the update.')
            missing_states = groups - set(old_inner_states)
            if missing_states:
                raise ValueError(
                    f'Optimizer state is missing active groups '
                    f'{sorted(missing_states, key=str)}.'
                )

            candidate_params = optax.apply_updates(self.params, updates)
            new_params = jax.tree_util.tree_map(
                lambda old, new, label: new if label in groups else old,
                self.params,
                candidate_params,
                self.param_labels,
            )
            isolated_inner_states = {
                group: (
                    new_inner_states[group]
                    if group in groups
                    else old_inner_states[group]
                )
                for group in old_inner_states
            }
            # Rebuild from the old root state as well as the old inactive
            # substates. Current optax partition states contain only
            # ``inner_states``; starting from the old value also keeps this
            # isolation correct if optax later adds a root-level clock.
            replace_state = getattr(self.opt_state, '_replace', None)
            if replace_state is None:
                replace_state = getattr(self.opt_state, 'replace', None)
            if replace_state is None:
                raise ValueError(
                    'Partitioned optimizer state does not support state replacement.'
                )
            new_opt_state = replace_state(inner_states=isolated_inner_states)
            advance_step = self.step_group is None or self.step_group in groups
            new_step = self.step + int(advance_step)

        return self.replace(
            step=new_step,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn, *, active_groups=None):
        """Apply a loss, optionally restricting the update to named groups."""
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
        return self.apply_gradients(grads=grads, active_groups=active_groups), info
