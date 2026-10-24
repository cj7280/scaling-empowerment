"""Craftax symbolic-observation slices that the potential-PQN encoder reads."""

from __future__ import annotations


# Craftax symbolic obs layout (empowerment/envs/craftax/{encoder_views,replay_storage}.py):
# obs = [ego-centric map | symbolic tail]; the tail's first 12 columns are inventory
# (player status excluded), the full tail is `inventory_status`.
_CRAFTAX_TAIL_DIMS_BY_OBS_DIM = {8268: 51, 1345: 22}
_CRAFTAX_INVENTORY_TAIL_INDICES = tuple(range(0, 12))


def _craftax_encoder_view(observation_dim: int, mode: str):
    """Resolve the encoder obs view -> (observation_indices | None, view_dim).

    Mirrors empowerment/envs/craftax/encoder_views.py::_gather_indices so warp's PQN
    encoder sees the same slice as the reference. Unknown layouts / full|auto keep the
    whole obs (indices None).
    """
    observation_dim = int(observation_dim)
    if mode in ("full", "auto"):
        return None, observation_dim
    tail_dim = _CRAFTAX_TAIL_DIMS_BY_OBS_DIM.get(observation_dim)
    if tail_dim is None or tail_dim >= observation_dim:
        return None, observation_dim
    tail_start = observation_dim - tail_dim
    if mode == "inventory_status":
        indices = tuple(range(tail_start, observation_dim))
    elif mode == "inventory":
        indices = tuple(tail_start + i for i in _CRAFTAX_INVENTORY_TAIL_INDICES)
    else:
        raise ValueError(f"encoder_obs_mode must be inventory|inventory_status|full|auto, got {mode!r}")
    return indices, len(indices)
