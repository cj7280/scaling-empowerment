"""Lossless Craftax observation storage for replay and encoder archives.

Binary map channels use eight bits per byte; the float32 tail is stored verbatim.
Adapted from empowerment's utils/replay_codecs.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


# Symbolic Craftax observation dim -> float tail dim (the rest is bit-packed map).
# Covers the only symbolic layouts: full (8268/51) and classic (1345/22).
SYMBOLIC_TAIL_DIMS_BY_OBS_DIM = {
    8268: 51,
    1345: 22,
}


@dataclass(frozen=True)
class CraftaxSymbolicObservationCodec:
    """Store Craftax symbolic observations as a bit-packed binary map plus float tail.

    The symbolic map channels are binary (block/item one-hots, multi-hot mob
    channels, and a thresholded light bit), so the `map_dim`-wide map portion is
    bit-packed 8 values per byte with `packbits` (lossless, structure-agnostic). The
    float tail (inventory/intrinsics/etc.) is kept as float32. Decoding unpacks back
    to a float32 vector of width `full_dim`.
    """

    full_dim: int
    tail_dim: int
    map_dtype: object = np.uint8
    tail_dtype: object = np.float32
    sample_dtype: object = np.float32

    @property
    def map_dim(self) -> int:
        return int(self.full_dim) - int(self.tail_dim)

    @property
    def packed_map_dim(self) -> int:
        return (int(self.map_dim) + 7) // 8

    def _check_dim(self, value) -> None:
        if int(value.shape[-1]) != int(self.full_dim):
            raise ValueError(
                "Craftax symbolic observation codec expected last dimension "
                f"{self.full_dim}, got {value.shape[-1]}."
            )

    def encode_jax(self, value):
        arr = jnp.asarray(value)
        self._check_dim(arr)
        binary_map = jnp.rint(jnp.clip(arr[..., : self.map_dim], 0.0, 1.0)).astype(jnp.uint8)
        packed_map = jnp.packbits(binary_map, axis=-1)
        tail = arr[..., self.map_dim :].astype(jnp.float32)
        return packed_map, tail

    def decode_jax(self, packed_map, tail):
        binary_map = jnp.unpackbits(
            jnp.asarray(packed_map, dtype=jnp.uint8), axis=-1, count=self.map_dim
        )
        return jnp.concatenate(
            [binary_map.astype(jnp.float32), jnp.asarray(tail, dtype=jnp.float32)],
            axis=-1,
        ).astype(jnp.float32)

    def encode_numpy(self, value):
        arr = np.asarray(value)
        self._check_dim(arr)
        binary_map = np.rint(np.clip(arr[..., : self.map_dim], 0.0, 1.0)).astype(np.uint8)
        packed_map = np.packbits(binary_map, axis=-1)
        tail = arr[..., self.map_dim :].astype(self.tail_dtype)
        return packed_map, tail

    def decode_numpy(self, packed_map, tail):
        binary_map = np.unpackbits(
            np.asarray(packed_map, dtype=np.uint8), axis=-1, count=self.map_dim
        )
        return np.concatenate(
            [binary_map.astype(self.sample_dtype), np.asarray(tail, dtype=self.sample_dtype)],
            axis=-1,
        )


def symbolic_codec_for_obs_dim(obs_dim: int) -> CraftaxSymbolicObservationCodec | None:
    """Codec for a known symbolic Craftax obs dim, or None if the layout is unknown."""
    tail_dim = SYMBOLIC_TAIL_DIMS_BY_OBS_DIM.get(int(obs_dim))
    if tail_dim is None or tail_dim <= 0 or tail_dim >= int(obs_dim):
        return None
    return CraftaxSymbolicObservationCodec(full_dim=int(obs_dim), tail_dim=int(tail_dim))
