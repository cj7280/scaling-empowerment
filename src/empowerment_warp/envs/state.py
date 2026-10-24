"""Lane selection for batched MJX-Warp environment states."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import jax
import jax.numpy as jnp


def gather_env_state_lanes(
    source_state: Any,
    target_template: Any,
    lane_indices: jax.Array,
) -> Any:
    """Gather complete source lanes into a target-width MJX-Warp state."""

    from mujoco.mjx.warp import types as warp_types

    indices = jnp.asarray(lane_indices, dtype=jnp.int32)

    def gather_data_field(field: str, source: jax.Array, template: jax.Array) -> jax.Array:
        if warp_types._BATCH_DIM["Data"].get(field, True):
            return source[indices]
        return template

    source_impl = source_state.data._impl
    template_impl = target_template.data._impl
    gathered_impl = template_impl.replace(
        **{
            field.name: gather_data_field(
                field.name,
                getattr(source_impl, field.name),
                getattr(template_impl, field.name),
            )
            for field in fields(type(template_impl))
            if field.type is not int
        }
    )
    gathered_data = target_template.data.replace(
        **{
            field.name: gather_data_field(
                field.name,
                getattr(source_state.data, field.name),
                getattr(target_template.data, field.name),
            )
            for field in fields(type(target_template.data))
            if field.name != "_impl"
        },
        _impl=gathered_impl,
    )
    target_data = jax.vmap(
        lambda current, gathered: current.where(jnp.asarray(True), gathered)
    )(target_template.data, gathered_data)

    def gather_lanes(value: Any) -> Any:
        return jax.tree_util.tree_map(lambda leaf: leaf[indices], value)

    return target_template._replace(
        **{
            name: target_data if name == "data" else gather_lanes(getattr(source_state, name))
            for name in source_state._fields
        }
    )
