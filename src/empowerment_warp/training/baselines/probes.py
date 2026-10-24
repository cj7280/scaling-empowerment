# Extracted from empowerment; see reference_manifest.json and LICENSE.
import jax

import jax.numpy as jnp

from empowerment_warp.training.baselines.replay import ProbeBatch

def uniform_actions(key, shape, dtype=jnp.float32):
    interior = jnp.nextafter(jnp.asarray(1., dtype), jnp.asarray(0., dtype))
    return jnp.clip(jax.random.uniform(key, shape, dtype=dtype, minval=-1., maxval=1.), -interior, interior)
