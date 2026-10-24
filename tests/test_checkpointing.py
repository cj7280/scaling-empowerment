"""Checkpoint save/restore round trip."""
import json

import jax.numpy as jnp
import numpy as np

from empowerment_warp.checkpointing import CheckpointStore, model_checkpoint


def test_full_checkpoint_round_trip(tmp_path):
    store = CheckpointStore(tmp_path / 'checkpoints')
    state = {'step': jnp.asarray(3), 'weights': jnp.arange(4, dtype=jnp.float32),
             'empty': jnp.zeros((2, 0), dtype=jnp.float32)}
    assert store.save(3, state, {'environment': 'test'}).is_dir()
    assert store.latest_step() == 3
    restored = store.restore(state)
    np.testing.assert_array_equal(restored['weights'], state['weights'])
    assert int(restored['step']) == 3 and restored['empty'].shape == (2, 0)
    manifest = json.loads((tmp_path / 'checkpoints' / 'latest.json').read_text())
    assert manifest['payload_kind'] == 'full'


def test_model_checkpoint_excludes_collector_and_replay(tmp_path):
    class State:
        sac = {'weights': jnp.ones((2,), dtype=jnp.float32)}
        skill = {'weights': jnp.ones((3,), dtype=jnp.float32)}
        env_steps = jnp.asarray(7, dtype=jnp.int32)
        collector = {'large': jnp.ones((16,), dtype=jnp.float32)}
        replay = {'large': jnp.ones((16,), dtype=jnp.float32)}

    payload = model_checkpoint(State())
    assert not hasattr(payload, 'collector') and not hasattr(payload, 'replay')
    store = CheckpointStore(tmp_path / 'checkpoints')
    store.save(7, payload, {'environment': 'test'}, payload_kind='model')
    restored = store.restore(payload)
    np.testing.assert_array_equal(restored.sac['weights'], payload.sac['weights'])
    assert store.latest_manifest()['payload_kind'] == 'model'
