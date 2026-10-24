"""Replay and hindsight-relabel sampling invariants."""
from typing import NamedTuple

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.agents.see_sac import ContinuousSAC, SACConfig
from empowerment_warp.agents.skill_classifier import SkillConfig, SkillLearner
from empowerment_warp.core import ReplayBuffer, Transition
from empowerment_warp.training.sac.config import OffPolicyConfig
from empowerment_warp.training.sac.runner import OffPolicyTrainer
from empowerment_warp.utils.sampling import policy_hindsight_batch


@struct.dataclass
class _State:
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    step_count: jax.Array


class _Config(NamedTuple):
    num_envs: int


class _Env:
    """Four lanes; every episode lasts exactly four steps."""
    config = _Config(num_envs=4)

    def reset(self, keys):
        values = jax.vmap(lambda key: jax.random.uniform(key, (3,), minval=-0.1, maxval=0.1))(keys)
        zeros = jnp.zeros((4,), dtype=jnp.bool_)
        return _State(values, jnp.zeros((4,)), zeros, zeros, jnp.zeros((4,), dtype=jnp.int32))

    def step(self, state, actions):
        step_count = state.step_count + 1
        return _State(state.obs + 0.05 * jnp.pad(actions, ((0, 0), (0, 1))), jnp.zeros((4,)),
                      jnp.zeros((4,), dtype=jnp.bool_), step_count >= 4, step_count)

    @staticmethod
    def where_done(state, reset_state):
        mask = state.terminated | state.truncated
        return jax.tree_util.tree_map(
            lambda a, b: jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - 1)), b, a),
            state, reset_state)


def _trainer(outcome: str, samples: int = 8) -> OffPolicyTrainer:
    sac = ContinuousSAC(SACConfig(3, 2, 2, hidden_dims=(8,), actor_layer_norm=False,
                                  critic_layer_norm=False, separate_skill0_actor=True,
                                  separate_skill0_critic=True,
                                  replace_skill_zero_reward_with_random_value=True))
    skill = SkillLearner(SkillConfig(method="potential", observation_dim=3, skill_dim=2,
                                     discrete_skills=False, hidden_dims=(8,), layer_norm=False))
    return OffPolicyTrainer(_Env(), sac, skill, OffPolicyConfig(
        rollout_steps=4, updates_per_chunk=1, batch_size=4, replay_capacity=32,
        episode_length=4, representation_episodes=4, max_future_steps=4,
        sample_hindsight_anchors=True, achieved_skill_relabel_outcome=outcome,
        achieved_skill_relabel_samples=samples))


def _rollout(trainer):
    state = trainer.initialize(jax.random.PRNGKey(20))
    _, transitions, _ = trainer._rollout(state.sac, state.collector)
    return state, transitions, trainer.replay.insert(state.replay, transitions)


def test_replay_insert_wraps_and_samples():
    x = jnp.arange(4, dtype=jnp.float32)
    obs = jnp.stack((x, x + 1), axis=-1)
    rows = Transition(obs, obs, x[:, None], x, jnp.ones_like(x), obs + 1, obs,
                      jnp.zeros((4,), bool), jnp.zeros((4, 2)), jnp.zeros((4, 2)), jnp.ones_like(x),
                      jnp.arange(4, dtype=jnp.int32), jnp.zeros((4,), dtype=jnp.int32))
    replay = ReplayBuffer(capacity=6)
    state = replay.insert(replay.insert(replay.initialize(rows), rows), rows)
    assert int(state.size) == 6 and int(state.cursor) == 2
    sample = jax.jit(lambda s, k: replay.sample(s, k, 3))(state, jax.random.PRNGKey(0))
    assert sample.observations.shape == (3, 2)


def test_discounted_relabel_futures_stay_inside_the_episode():
    """Each sampled future is a later state of the row's own episode (rows at t=1 -> t=2 or t=3)."""
    trainer = _trainer("discounted", samples=4)
    state, transitions, replay_state = _rollout(trainer)
    indices = jnp.arange(4, 8, dtype=jnp.int32)
    _, _, allowed, outcomes, metrics = policy_hindsight_batch(
        trainer.agent.config, trainer.replay, replay_state, indices, jax.random.PRNGKey(21),
        hindsight=True)
    outcomes = np.asarray(outcomes)
    nexts = np.asarray(transitions.next_observations).reshape(4, 4, 3)  # (timestep, lane, obs)
    for lane in range(4):
        for sample in range(4):
            assert any(np.allclose(outcomes[lane, sample], c) for c in nexts[2:4, lane])
    assert float(metrics.segment_end_available_fraction) == 1.0
    rows = jax.tree_util.tree_map(lambda v: v[4:8], transitions)
    relabeled, _ = trainer.skill.relabel_achieved_skills(
        state.skill, rows, allowed, outcome_observations=jnp.asarray(outcomes))
    assert np.asarray(relabeled.skills).shape == (4, 2)


def test_segment_end_relabel_outcome_is_the_episode_final_state():
    trainer = _trainer("segment_end")
    _, transitions, replay_state = _rollout(trainer)
    _, _, _, outcomes, _ = policy_hindsight_batch(
        trainer.agent.config, trainer.replay, replay_state, jnp.arange(4, 8, dtype=jnp.int32),
        jax.random.PRNGKey(21), hindsight=True)
    np.testing.assert_allclose(np.asarray(outcomes), np.asarray(transitions.next_observations[12:16]))


def test_relabel_label_is_the_unit_achieved_displacement():
    """The relabeled skill is unit(phi(achieved) - phi(anchor)); skill-off rows stay off."""
    learner = SkillLearner(SkillConfig(method="potential", observation_dim=6, skill_dim=4,
                                       discrete_skills=False, hidden_dims=(16,),
                                       achieved_skill_relabeling=True))
    state = learner.initialize(jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (16, 6))
    anchors = jax.random.normal(jax.random.PRNGKey(2), (16, 6))
    skills = learner.sample_skills(jax.random.PRNGKey(3), 16).at[0].set(0.0)
    rows = Transition(obs, anchors, jnp.zeros((16, 1)), jnp.zeros((16,)), jnp.ones((16,)), obs,
                      anchors, jnp.zeros((16,)), skills, skills, jnp.ones((16,)),
                      jnp.zeros((16,)), jnp.zeros((16,)))
    relabeled, _ = learner.relabel_achieved_skills(state, rows, jnp.ones((16,), bool))
    phi = lambda x: learner._apply(state.params, learner.encoder_observations(x))
    delta = np.asarray(phi(obs) - phi(anchors))
    expected = delta / np.linalg.norm(delta, axis=1, keepdims=True)
    np.testing.assert_allclose(np.asarray(relabeled.skills)[1:], expected[1:], atol=1e-5)
    assert np.all(np.asarray(relabeled.skills)[0] == 0.0)
