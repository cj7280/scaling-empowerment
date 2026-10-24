"""The skill classifier trains on a toy batch and its reward is finite."""
import jax
import jax.numpy as jnp
import numpy as np

from empowerment_warp.agents.skill_classifier import SkillBatch, SkillConfig, SkillLearner


def test_classifier_update_decreases_loss_on_a_toy_batch():
    config = SkillConfig(method='potential', observation_dim=5, skill_dim=2, discrete_skills=False,
                         hidden_dims=(32,), layer_norm=False, phi_noise_std=0.0)
    learner = SkillLearner(config)
    state = learner.initialize(jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (64, 5))
    skills = learner.sample_skills(jax.random.PRNGKey(2), 64)
    # The future state moves along the commanded skill, so the skill is identifiable.
    future = obs + jnp.pad(skills, ((0, 0), (0, 3)))
    batch = SkillBatch(obs, future, future, obs, skills, jnp.ones((64,), dtype=jnp.bool_))
    loss = lambda s: float(learner.evaluate_preprocessed(
        s, learner.preprocess_batch(batch), jax.random.PRNGKey(3), apply_phi_noise=False).loss)
    before = loss(state)
    for step in range(50):
        state, metrics = learner.update(state, batch, jax.random.PRNGKey(10 + step))
    assert int(state.step) == 50
    assert all(np.isfinite(float(value)) for value in metrics)
    assert loss(state) < before - 0.1
    reward = learner.intrinsic_reward(state, batch, jax.random.PRNGKey(4))
    assert reward.shape == (64,) and np.isfinite(np.asarray(reward)).all()
