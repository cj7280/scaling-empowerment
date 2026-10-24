"""Agents: one self-contained file per agent (OGBench ``impls/agents``).

``AGENTS`` maps an agent name to a ``"module:attribute"`` string. Nothing is
imported until ``load`` is called, so importing this package stays cheap and
never pulls in JAX, MuJoCo or Warp ahead of the CLI's environment setup.
"""
import importlib

AGENTS = {
    'see_sac': 'empowerment_warp.agents.see_sac:SEEAgent',
    'see_pqn': 'empowerment_warp.agents.see_pqn:PotentialPQNAgent',
    'skill_classifier': 'empowerment_warp.agents.skill_classifier:SkillLearner',
    'vim': 'empowerment_warp.agents.vim:create_vim_train_states',
    'latent_gce_vector': 'empowerment_warp.agents.latent_gce:create_vector_gce_train_state',
    'ppo': 'empowerment_warp.agents.ppo:make_ppo_update',
}


def load(name):
    """Import and return the object registered under ``name``."""
    module, attribute = AGENTS[name].split(':')
    return getattr(importlib.import_module(module), attribute)
