"""Craftax SEE agent: recurrent potential-PQN controller and its runner-driven stages.

Recurrent PQN controller for Craftax. The controller math follows PureJaxQL's
recurrent Craftax PQN. The networks (ScannedLSTM, SkillLogitAdapter, RecurrentQNetwork,
BatchRenorm) live in ``empowerment_warp.utils.networks``; ModuleDict, TrainState and
nonpytree_field in ``empowerment_warp.utils.flax_utils``.

``PotentialPQNAgent`` is LEAN: the representation encoder lives in a separate
`SkillLearner`/`SkillState`, NOT inside the agent's TrainState. So the class keeps only the pieces that are intrinsically the agent's:

  * the two-Q recurrent controller: `potential_q` (z = 0) and the skill Q, stored under
    the checkpointed module key `effective_q`; no encoder
    modules and no param-group optimizer;
  * `_potential_values` = max(0, E_z max_a Q_skill(s, s, z, a)) with a zero recurrent
    carry; the nonnegative floor follows the skill average.
  * `_prepare_rollout_batch` = add the skill-command boundary and consume the
    frozen empowerment reward the RUNNER precomputes (via SkillLearner intrinsic
    reward + these potential values), because the encoder is not in the agent.

The frozen-reward assembly and encoder update need the external SkillState, so they are
module-level factories here (`make_frozen_reward_fn`, `make_encoder_update_burst`), each
returning its own jitted stage, which the runner
(training/pqn/runner.py) calls once per chunk.
"""

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from empowerment_warp.agents.skill_classifier import SkillBatch, SkillLearner
from empowerment_warp.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from empowerment_warp.utils.networks import RecurrentQNetwork


def linear_epsilon(
    update_index,
    *,
    start,
    end,
    total_env_steps,
    rollout_env_steps,
    decay_fraction,
):
    """Reference epsilon schedule, clocked once per complete rollout update."""
    total_updates = max(1, int(total_env_steps) // max(1, int(rollout_env_steps)))
    decay_updates = max(1.0, float(decay_fraction) * total_updates)
    fraction = jnp.clip(
        jnp.asarray(update_index, dtype=jnp.float32) / float(decay_updates), 0.0, 1.0
    )
    return jnp.asarray(start, jnp.float32) + fraction * (
        jnp.asarray(end, jnp.float32) - jnp.asarray(start, jnp.float32)
    )


def q_lambda_targets(q_values, rewards, dones, *, discount, trace_lambda):
    """Stopped PureJaxQL Q(lambda) targets for the first ``T - 1`` rows.

    ``q_values`` is time-major ``[T, B, A]``.  The final row is a bootstrap
    observation and intentionally bears no loss, matching the pinned reference.
    """
    q_values = jnp.asarray(q_values)
    rewards = jnp.asarray(rewards, dtype=q_values.dtype)
    dones = jnp.asarray(dones, dtype=q_values.dtype)
    if q_values.shape[0] < 2:
        raise ValueError('PQN Q(lambda) requires at least two rollout rows.')

    train_q = q_values[:-1]
    train_rewards = rewards[:-1]
    train_dones = dones[:-1]
    final_bootstrap = jnp.max(q_values[-1], axis=-1)
    final_return = train_rewards[-1] + discount * (1.0 - train_dones[-1]) * final_bootstrap
    final_next_q = jnp.max(train_q[-1], axis=-1)

    def backward(carry, transition):
        next_return, next_q = carry
        reward, q_value, done = transition
        one_step = reward + discount * (1.0 - done) * next_q
        current_return = one_step + discount * trace_lambda * (next_return - next_q)
        current_return = (1.0 - done) * current_return + done * reward
        return (current_return, jnp.max(q_value, axis=-1)), current_return

    _, prefix = jax.lax.scan(
        backward,
        (final_return, final_next_q),
        (train_rewards[:-1], train_q[:-1], train_dones[:-1]),
        reverse=True,
    )
    return jax.lax.stop_gradient(jnp.concatenate([prefix, final_return[None]], axis=0))


def _action_indices(actions):
    actions = jnp.asarray(actions)
    if actions.ndim <= 2 and (actions.ndim < 2 or actions.shape[-1] <= 1):
        return actions.astype(jnp.int32).reshape(actions.shape[:1])
    return jnp.argmax(actions, axis=-1).astype(jnp.int32)


class PotentialPQNAgent(flax.struct.PyTreeNode):
    """Two-Q recurrent PQN whose reward is the frozen empowerment potential."""

    rng: Any
    network: Any
    batch_stats: Any
    config: Any = nonpytree_field()
    # Retired discrete-skill library: always None (continuous skills). Kept so the struct
    # field layout of saved agents is unchanged.
    skill_library: Any = None
    env_steps: Any = 0
    n_updates: Any = 0
    grad_steps: Any = 0
    previous_prefix_labels: Any = None
    previous_prefix_episode_ids: Any = None
    reward_softmax_w: Any = None
    encoder_acc_ema: Any = None

    def recurrent_state_size(self):
        return int(self.config['pqn_recurrent_state_size'])

    def initial_recurrent_state(self, batch_size, dtype=jnp.float32):
        return jnp.zeros((int(batch_size), self.recurrent_state_size()), dtype=dtype)

    def _unpack_state(self, state):
        hidden_size = int(self.config['pqn_hidden_size'])
        state = jnp.asarray(state)
        potential = [(state[..., :hidden_size], state[..., hidden_size : 2 * hidden_size])]
        offset = 2 * hidden_size
        skill_q = [
            (
                state[..., offset : offset + hidden_size],
                state[..., offset + hidden_size : offset + 2 * hidden_size],
            )
        ]
        offset += 2 * hidden_size
        return potential, skill_q, state[..., offset].astype(jnp.int32)

    def _pack_state(self, potential, skill_q, last_action):
        parts = [potential[0][0], potential[0][1], skill_q[0][0], skill_q[0][1]]
        parts.append(jnp.asarray(last_action, dtype=parts[0].dtype)[..., None])
        return jnp.concatenate(parts, axis=-1)

    def exploration_epsilon(self):
        return linear_epsilon(
            self.n_updates,
            start=self.config['pqn_epsilon_start'],
            end=self.config['pqn_epsilon_end'],
            total_env_steps=self.config['pqn_total_env_steps'],
            rollout_env_steps=self.config['pqn_rollout_env_steps'],
            decay_fraction=self.config['pqn_epsilon_fraction'],
        )

    def _module_apply(
        self,
        module_name,
        hidden,
        observations,
        resets,
        last_actions,
        *,
        params=None,
        batch_stats=None,
        train=False,
        mutable=False,
    ):
        variables = {
            'params': self.network.params if params is None else params,
            'batch_stats': self.batch_stats if batch_stats is None else batch_stats,
        }
        return self.network.apply_fn(
            variables,
            hidden,
            observations,
            resets,
            last_actions,
            name=module_name,
            train=train,
            mutable=['batch_stats'] if mutable else False,
        )

    def _skill_q_inputs(self, observations, anchors, skills):
        return jnp.concatenate([observations, anchors, skills], axis=-1)

    def sample_actions_recurrent_from_params(
        self,
        policy_variables,
        observations,
        skills,
        anchor_observations,
        recurrent_state,
        recurrent_reset,
        seed,
        epsilon,
        *,
        deterministic=False,
    ):
        """Advance the recurrences one step and return one-hot actions.

        `_select_action` decides how the network outputs become an action
        (epsilon-greedy on Q-values).
        """
        potential_state, skill_q_carry, previous_action = self._unpack_state(recurrent_state)
        resets = jnp.asarray(recurrent_reset, dtype=bool)[None]
        previous_action_t = previous_action[None]
        variables = {
            'params': policy_variables['params'],
            'batch_stats': policy_variables['batch_stats'],
        }
        potential_state, potential_q = self.network.apply_fn(
            variables,
            potential_state,
            jnp.asarray(observations)[None],
            resets,
            previous_action_t,
            name='potential_q',
            train=False,
        )
        q_values = potential_q[0]
        skill_q_carry, skill_q = self.network.apply_fn(
            variables,
            skill_q_carry,
            self._skill_q_inputs(
                jnp.asarray(observations),
                jnp.asarray(anchor_observations),
                jnp.asarray(skills),
            )[None],
            resets,
            previous_action_t,
            name='effective_q',
            train=False,
        )
        skill_on = jnp.linalg.norm(skills, axis=-1) > float(
            self.config.get('skill_zero_norm_eps', 1e-6)
        )
        q_values = jnp.where(skill_on[:, None], skill_q[0], q_values)

        action_idx = self._select_action(q_values, seed, epsilon, deterministic=deterministic)
        next_state = self._pack_state(potential_state, skill_q_carry, action_idx)
        actions = jax.nn.one_hot(
            action_idx, int(self.config['action_dim']), dtype=jnp.float32
        )
        return actions, action_idx, next_state

    def _select_action(self, outputs, seed, epsilon, *, deterministic):
        """Epsilon-greedy over Q-values (reference PQN)."""
        greedy = jnp.argmax(outputs, axis=-1).astype(jnp.int32)
        if deterministic:
            return greedy
        random_key, choose_key = jax.random.split(seed)
        random_action = jax.random.randint(
            random_key, greedy.shape, 0, int(self.config['action_dim'])
        )
        return jnp.where(
            jax.random.uniform(choose_key, greedy.shape) < epsilon,
            random_action,
            greedy,
        )

    @partial(jax.jit, static_argnames=('deterministic',))
    def sample_actions_recurrent(
        self,
        observations,
        skills,
        anchor_observations,
        recurrent_state,
        reset,
        seed,
        temperature=1.0,
        deterministic=False,
    ):
        """Stateful scalar/vector evaluation hook used by the shared evaluator."""
        del temperature
        epsilon = jnp.asarray(0.0 if deterministic else self.exploration_epsilon())
        actions, _, next_state = self.sample_actions_recurrent_from_params(
            {'params': self.network.params, 'batch_stats': self.batch_stats},
            observations,
            skills,
            anchor_observations,
            recurrent_state,
            reset,
            seed,
            epsilon,
            deterministic=deterministic,
        )
        return actions, next_state

    @partial(jax.jit, static_argnames=('deterministic',))
    def sample_actions(
        self,
        observations,
        skills=None,
        anchor_observations=None,
        seed=None,
        temperature=1.0,
        deterministic=False,
    ):
        """Stateless one-step evaluation; greedy evaluation uses epsilon zero."""
        del temperature
        observations = jnp.asarray(observations)
        if observations.ndim == 1:
            observations = observations[None]
        batch_size = observations.shape[0]
        if skills is None:
            skills = jnp.zeros((batch_size, int(self.config.get('skill_dim', 0))), jnp.float32)
        if anchor_observations is None:
            anchor_observations = observations
        if seed is None:
            seed = self.rng
        policy_variables = {'params': self.network.params, 'batch_stats': self.batch_stats}
        actions, _, _ = self.sample_actions_recurrent_from_params(
            policy_variables,
            observations,
            skills,
            anchor_observations,
            self.initial_recurrent_state(batch_size, observations.dtype),
            jnp.ones((batch_size,), dtype=bool),
            seed,
            jnp.asarray(0.0 if deterministic else self.exploration_epsilon()),
            deterministic=deterministic,
        )
        return actions

    def _probe_skills(self, dtype, *, rows, key=None):
        """The skills the z=0 value averages over, shaped ``[K, rows, skill_dim]``.

        K = ``potential_skill_samples`` FRESH Monte Carlo draws z ~ unit sphere per
        row and per call (the same rule as SAC's random skill value), so the z=0 value is
        an unbiased estimate of E_z over the prior.
        """
        num_samples = int(self.config.get('potential_skill_samples', 0) or 0)
        if num_samples <= 0:
            raise ValueError('continuous skills need potential_skill_samples > 0 for the z=0 value')
        if key is None:
            raise ValueError('continuous skill-0 value needs a PRNG key for its Monte Carlo draws')
        skill_dim = int(self.config['skill_dim'])
        draws = jax.random.normal(key, (num_samples, int(rows), skill_dim), dtype=dtype)
        return draws / jnp.sqrt(jnp.sum(jnp.square(draws), axis=-1, keepdims=True) + 1e-8)

    def potential_values(self, observations, *, batch=None, initial_state=None, key=None):
        """Average skill values with fresh memory or the observed pre-step history."""
        # Configs written before this flag existed reset memory at the switch.
        if not self.config.get('pqn_preserve_memory_at_switch', False):
            return self._potential_values(observations, key=key)
        if batch is None or initial_state is None:
            raise ValueError('memory-preserving potential values require the rollout and its initial recurrent state')
        worlds = initial_state.shape[0]
        def time_env(x):
            return x.reshape((-1, worlds) + x.shape[1:])
        _, carry, last_action = self._unpack_state(initial_state)
        module = 'effective_q'
        actions = jnp.asarray(batch['actions'])
        if actions.ndim > 1:
            actions = jnp.argmax(actions, axis=-1)
        previous = jnp.concatenate([last_action[None], time_env(actions)[:-1]], axis=0)
        resets = time_env(batch['rnn_resets']).astype(bool)
        # Replay the actual commands once. Probe skills replace only the current
        # command; the observed past and its pre-observation memory stay fixed.
        _, _, post_carries = self.network.apply_fn(
            {'params': self.network.params, 'batch_stats': self.batch_stats},
            carry, self._skill_q_inputs(time_env(batch['observations']),
                                          time_env(batch['anchor_observations']),
                                          time_env(batch['skills'])),
            resets, previous, name=module, train=False, return_carries=True,
        )
        pre_carries = jax.tree_util.tree_map(
            lambda initial, post: jnp.concatenate([initial[None], post[:-1]], axis=0).reshape(
                (-1, initial.shape[-1])), carry, post_carries)
        return self._potential_values(observations, key=key, carry=pre_carries,
                                      resets=resets.reshape(-1), last_actions=previous.reshape(-1))

    def _potential_values(self, observations, key=None, *, carry=None, resets=None, last_actions=None):
        """Potential value E_z max_a Q_skill(s, s, z, a), using the supplied pre-observation memory."""
        observations = jnp.asarray(observations)
        batch_size = observations.shape[0]
        skills_to_scan = self._probe_skills(observations.dtype, rows=batch_size, key=key)
        num_skills = int(skills_to_scan.shape[0])
        hidden_size = int(self.config['pqn_hidden_size'])
        network_def = RecurrentQNetwork(
            action_dim=int(self.config['action_dim']), hidden_size=hidden_size
        )
        if carry is None:
            carry = network_def.initialize_carry(batch_size)
            resets = jnp.ones((batch_size,), dtype=bool)
            last_actions = jnp.zeros((batch_size,), dtype=jnp.int32)
        variables = {'params': self.network.params, 'batch_stats': self.batch_stats}

        def accumulate(total, skills):
            skill_q_input = self._skill_q_inputs(
                observations, observations, skills
            )
            _, q_values = self.network.apply_fn(
                variables,
                carry,
                skill_q_input[None],
                resets[None],
                last_actions[None],
                name='effective_q',
                train=False,
            )
            return total + jnp.max(q_values[0], axis=-1), None

        values, _ = jax.lax.scan(
            accumulate, jnp.zeros((batch_size,), observations.dtype), skills_to_scan
        )
        return jnp.maximum(values / float(num_skills), 0.0)

    def _prepare_rollout_batch(self, batch, *, num_envs, chunk_steps):
        """Add the skill-command boundary; consume the runner's frozen reward.

        The runner precomputes ``batch['pqn_rewards']`` = frozen empowerment
        reward (SkillLearner intrinsic on skill-on rows, ``_potential_values`` on
        skill-off rows, x (1-gamma), stop-gradient) because that needs the
        external SkillState. Here we only add ``pqn_command_boundary`` so
        ``update_rollout`` folds skill switches into the done mask (parity with
        the reference two-Q done semantics).
        """
        del chunk_steps
        prepared = dict(batch)
        if "pqn_rewards" not in prepared:
            raise ValueError(
                "potential-PQN expects the runner to precompute frozen 'pqn_rewards' "
                "(SkillLearner intrinsic + _potential_values); none found in the batch."
            )
        if "pqn_command_boundary" not in prepared:
            prepared["pqn_command_boundary"] = (
                jnp.linalg.norm(
                    jnp.asarray(batch["next_skills"]) - jnp.asarray(batch["skills"]),
                    axis=-1,
                )
                > float(self.config.get("skill_zero_norm_eps", 1e-6))
            ).astype(jnp.float32)
        return prepared, {}, {}

    def _q_values_for_loss(
        self,
        params,
        batch_stats,
        initial_state,
        batch,
        *,
        train,
    ):
        potential_state, skill_q_carry, initial_last_action = self._unpack_state(initial_state)
        actions = _action_indices(batch['actions'])
        last_actions = jnp.concatenate([initial_last_action[None], actions[:-1]], axis=0)
        reset = jnp.asarray(batch['rnn_resets'], dtype=bool)

        (potential_state, potential_q), mutable = self._module_apply(
            'potential_q',
            potential_state,
            batch['observations'],
            reset,
            last_actions,
            params=params,
            batch_stats=batch_stats,
            train=train,
            mutable=train,
        )
        next_batch_stats = mutable['batch_stats'] if train else batch_stats
        skill_q_input = self._skill_q_inputs(
            batch['observations'], batch['anchor_observations'], batch['skills']
        )
        (skill_q_carry, skill_q), mutable = self._module_apply(
            'effective_q',
            skill_q_carry,
            skill_q_input,
            reset,
            last_actions,
            params=params,
            batch_stats=next_batch_stats,
            train=train,
            mutable=train,
        )
        next_batch_stats = mutable['batch_stats'] if train else next_batch_stats
        skill_on = jnp.linalg.norm(batch['skills'], axis=-1) > float(
            self.config.get('skill_zero_norm_eps', 1e-6)
        )
        return jnp.where(skill_on[..., None], skill_q, potential_q), next_batch_stats

    def _minibatch_loss(self, params, batch_stats, minibatch_state, minibatch):
        """Reference Q(lambda) regression on one env-axis minibatch (time-major sequences)."""
        q_values, updated_batch_stats = self._q_values_for_loss(
            params,
            batch_stats,
            minibatch_state,
            minibatch,
            train=True,
        )
        target = q_lambda_targets(
            jax.lax.stop_gradient(q_values),
            minibatch['pqn_rewards'],
            minibatch['pqn_dones'],
            discount=float(self.config['discount']),
            trace_lambda=float(self.config['pqn_lambda']),
        )
        chosen = jnp.take_along_axis(
            q_values,
            _action_indices(minibatch['actions'])[..., None],
            axis=-1,
        ).squeeze(-1)[:-1]
        loss = 0.5 * jnp.square(chosen - target).mean()
        info = {
            'pqn/td_loss': loss,
            'pqn/q_mean': chosen.mean(),
            'pqn/target_mean': target.mean(),
        }
        return loss, (info, updated_batch_stats)

    @partial(jax.jit, static_argnames=('num_envs', 'chunk_steps'))
    def update_rollout(
        self,
        batch,
        *,
        num_envs,
        chunk_steps,
        initial_recurrent_state=None,
    ):
        """Freeze one rollout, then perform reference env-axis PQN minibatches."""
        if int(chunk_steps) < 2:
            raise ValueError('PQN rollout chunks must contain at least two time steps.')
        if int(num_envs) % int(self.config['pqn_num_minibatches']) != 0:
            raise ValueError('pqn_num_minibatches must divide num_envs exactly.')
        if initial_recurrent_state is None:
            initial_recurrent_state = self.initial_recurrent_state(num_envs)

        prepared, prepare_info, state_updates = self._prepare_rollout_batch(
            batch, num_envs=num_envs, chunk_steps=chunk_steps
        )

        def time_env(value):
            value = jnp.asarray(value)
            return value.reshape((chunk_steps, num_envs) + value.shape[1:])

        learner_batch = {
            key: time_env(prepared[key])
            for key in (
                'observations',
                'actions',
                'terminals',
                'rnn_resets',
                'pqn_rewards',
            )
        }
        for key in ('skills', 'next_skills', 'anchor_observations'):
            if key in prepared:
                learner_batch[key] = time_env(prepared[key])
        if 'pqn_command_boundary' in prepared:
            learner_batch['pqn_command_boundary'] = time_env(
                prepared['pqn_command_boundary']
            )

        dones = jnp.asarray(learner_batch['terminals'], dtype=jnp.float32)
        skill_change = learner_batch.get('pqn_command_boundary')
        if skill_change is None:
            skill_change = jnp.linalg.norm(
                learner_batch['next_skills'] - learner_batch['skills'], axis=-1
            ) > float(self.config.get('skill_zero_norm_eps', 1e-6))
        dones = jnp.maximum(dones, skill_change.astype(jnp.float32))
        learner_batch['pqn_dones'] = dones

        new_rng, update_rng = jax.random.split(self.rng)
        minibatches = int(self.config['pqn_num_minibatches'])
        epochs = int(self.config['pqn_num_epochs'])
        batch_per_minibatch = num_envs // minibatches

        def one_epoch(carry, epoch_key):
            network, batch_stats = carry
            permutation = jax.random.permutation(epoch_key, num_envs)

            def split_envs(value):
                value = value[:, permutation]
                value = value.reshape(
                    (chunk_steps, minibatches, batch_per_minibatch) + value.shape[2:]
                )
                return jnp.swapaxes(value, 0, 1)

            shuffled = jax.tree_util.tree_map(split_envs, learner_batch)
            shuffled_state = initial_recurrent_state[permutation].reshape(
                (minibatches, batch_per_minibatch, initial_recurrent_state.shape[-1])
            )

            def one_minibatch(carry, inputs):
                network, batch_stats = carry
                minibatch, minibatch_state = inputs

                def loss_fn(params):
                    return self._minibatch_loss(params, batch_stats, minibatch_state, minibatch)

                (_, (info, updated_batch_stats)), grads = jax.value_and_grad(
                    loss_fn, has_aux=True
                )(network.params)
                network = network.apply_gradients(grads=grads)
                return (network, updated_batch_stats), info

            (network, batch_stats), info = jax.lax.scan(
                one_minibatch,
                (network, batch_stats),
                (shuffled, shuffled_state),
            )
            return (network, batch_stats), info

        epoch_keys = jax.random.split(update_rng, epochs)
        (network, batch_stats), info_steps = jax.lax.scan(
            one_epoch, (self.network, self.batch_stats), epoch_keys
        )
        info = {key: jnp.mean(value) for key, value in dict(info_steps).items()}
        info.update(prepare_info)
        rollout_rows = int(num_envs) * int(chunk_steps)
        info.update(
            {
                'pqn/update_epochs': float(epochs),
                'pqn/num_minibatches': float(minibatches),
                'pqn/minibatch_sequences': float(batch_per_minibatch),
                'pqn/sequence_length': float(chunk_steps),
                'pqn/rollout_size': float(rollout_rows),
                'pqn/usable_rollout_size': float(num_envs * (chunk_steps - 1)),
                'pqn/context_free_recurrence': float(
                    bool(self.config.get('pqn_context_free_recurrence', False))
                ),
            }
        )
        info.update({'pqn/epsilon': self.exploration_epsilon()})
        updates = dict(
            rng=new_rng,
            network=network,
            batch_stats=batch_stats,
            env_steps=self.env_steps + rollout_rows,
            n_updates=self.n_updates + 1,
            grad_steps=self.grad_steps + epochs * minibatches,
        )
        updates.update(state_updates)
        return self.replace(**updates), info

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        ex_encoder_observations=None,
        **kwargs,
    ):
        del ex_encoder_observations, kwargs
        config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
        action_dim = int(ex_actions.shape[-1])
        if action_dim <= 1:
            raise ValueError('PQN requires one-hot discrete actions with action_dim > 1.')
        if str(config.get('benchmark_backend', 'craftax')) != 'craftax':
            raise ValueError('PQN is currently implemented only for benchmark_backend=craftax.')

        hidden_size = int(config.get('pqn_hidden_size', 512))
        # Only the two-Q potential controller is implemented (no plain PQN, no shared
        # Q(s, anchor, z) head).
        if not bool(config.get('potential_pqn', False)):
            raise ValueError('potential_pqn must be True; plain PQN is not supported')
        if not bool(config.get('separate_skill0_q', True)):
            raise ValueError('separate_skill0_q must be True; a shared-Q controller is not supported')
        context_free_recurrence = bool(config.get('pqn_context_free_recurrence', False))
        # Separate potential and skill Q-heads carry two LSTMs.
        recurrent_state_size = 4 * hidden_size + 1
        config.update(
            action_dim=action_dim,
            discrete_actions=True,
            on_policy_updates=True,
            updates_use_skill_buffer=False,
            pqn_recurrent=True,
            pqn_recurrent_state_size=recurrent_state_size,
            ppo_recurrent=False,
            separate_skill0_q=True,
        )
        expected_envs = int(config.get('pqn_expected_num_envs', 1024))
        expected_steps = int(config.get('pqn_rollout_steps', 128))
        config['pqn_rollout_env_steps'] = expected_envs * expected_steps

        def q_def(*, recurrent_context_dim=0):
            return RecurrentQNetwork(
                action_dim=action_dim,
                hidden_size=hidden_size,
                num_layers=int(config.get('pqn_num_layers', 1)),
                num_rnn_layers=1,
                norm_input=bool(config.get('pqn_norm_input', True)),
                add_last_action=True,
                # Configs written before this flag existed reset memory at the switch.
                reset_last_action=bool(config.get('pqn_preserve_memory_at_switch', False)),
                recurrent_context_dim=recurrent_context_dim,
                skill_dim=int(config.get('skill_dim', 0) or 0),
                skill_logit_init_std=float(config.get('skill_logit_init_std', 0.0) or 0.0),
                value_head=False,
            )
        init_obs = jnp.asarray(ex_observations[:1])[None]
        init_reset = jnp.zeros((1, 1), dtype=bool)
        init_action = jnp.zeros((1, 1), dtype=jnp.int32)

        skill_dim = int(config['skill_dim'])
        skill_q_width = int(ex_observations.shape[-1]) * 2 + skill_dim
        skill_q_context_dim = (
            int(ex_observations.shape[-1]) + skill_dim if context_free_recurrence else 0
        )
        potential_q = q_def()
        skill_q = q_def(recurrent_context_dim=skill_q_context_dim)
        network_info = {
            'potential_q': (
                potential_q,
                dict(
                    hidden=potential_q.initialize_carry(1),
                    observations=init_obs,
                    resets=init_reset,
                    last_actions=init_action,
                    train=False,
                ),
            ),
            # 'effective_q' = the skill Q; the key names the checkpointed param subtree.
            'effective_q': (
                skill_q,
                dict(
                    hidden=skill_q.initialize_carry(1),
                    observations=jnp.zeros((1, 1, skill_q_width), jnp.float32),
                    resets=init_reset,
                    last_actions=init_action,
                    train=False,
                ),
            ),
        }

        network_def = ModuleDict({key: value[0] for key, value in network_info.items()})
        network_args = {key: value[1] for key, value in network_info.items()}
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        variables = network_def.init(init_rng, **network_args)
        params = variables['params']
        batch_stats = variables.get('batch_stats', flax.core.freeze({}))

        learning_rate = float(config.get('lr', 3e-4))
        if bool(config.get('pqn_lr_linear_decay', True)):
            total_updates = max(
                1,
                int(config['pqn_total_env_steps'])
                // int(config['pqn_rollout_env_steps']),
            )
            learning_rate = optax.linear_schedule(
                init_value=learning_rate,
                end_value=1e-20,
                transition_steps=(
                    total_updates
                    * int(config['pqn_num_epochs'])
                    * int(config['pqn_num_minibatches'])
                ),
            )
        # Reference PQN optimizer: global-norm clip + RAdam (schedule already applied).
        policy_tx = optax.chain(
            optax.clip_by_global_norm(float(config.get('pqn_max_grad_norm', 0.5))),
            optax.radam(learning_rate),
        )
        network = TrainState.create(network_def, params, tx=policy_tx)

        return cls(
            rng=rng,
            network=network,
            batch_stats=batch_stats,
            config=flax.core.FrozenDict(**config),
            skill_library=None,
            env_steps=jnp.asarray(0, jnp.int32),
            n_updates=jnp.asarray(0, jnp.int32),
            grad_steps=jnp.asarray(0, jnp.int32),
            previous_prefix_labels=jnp.full((expected_envs,), -1, jnp.int32),
            previous_prefix_episode_ids=jnp.full((expected_envs,), -1, jnp.int32),
            reward_softmax_w=jnp.asarray(0.0, jnp.float32),
            encoder_acc_ema=jnp.asarray(1.0, jnp.float32),
        )


def get_config():
    """Potential-PQN defaults: reference base PQN + empowerment two-Q knobs.

    Reward-reference / encoder semantics are NOT here — they live in the warp
    SkillConfig (`potential_reward_reference`, `reward_encoder`). The mapping
    use_phi_Q=fixed_anchor <-> potential_reward_reference='anchor' and
    use_phi_encoder=anchor_from_current <-> encoder delta is the correspondence with the
    reference PQN.
    """
    config = dict(
        agent_name='pqn-base',
        lr=3e-4,
        discount=0.99,
        reward_scale=1.0,
        pqn_hidden_size=512,
        pqn_recurrent=True,
        pqn_recurrent_state_size=1025,
        pqn_context_free_recurrence=False,
        pqn_preserve_memory_at_switch=True,  # new configs carry memory across the switch
        pqn_num_layers=1,
        pqn_norm_input=True,
        pqn_lr_linear_decay=True,
        pqn_rollout_steps=128,
        pqn_expected_num_envs=1024,
        pqn_num_epochs=4,
        pqn_num_minibatches=4,
        pqn_lambda=0.5,
        pqn_max_grad_norm=0.5,
        pqn_epsilon_start=1.0,
        pqn_epsilon_end=0.005,
        pqn_epsilon_fraction=0.1,
        pqn_total_env_steps=1_000_000_000,
        potential_pqn=False,
        benchmark_backend='craftax',
        skill_dim=0,
        discrete_skills=False,
        num_skills=1,
        discrete_actions=True,
        on_policy_updates=True,
        updates_use_skill_buffer=False,
    )
    config.update(
        agent_name="potential-pqn",
        potential_pqn=True,
        pqn_context_free_recurrence=True,
        discount=0.95,
        skill_dim=16,
        discrete_skills=True,
        num_skills=16,
        potential_skill_samples=0,  # Enumerate the discrete library; mean over skills.
        # True = two separate Q heads (potential_q + effective_q, the skill Q), reference /
        # Ant-consistent (Ant base uses separate potential actor/critic). False = one
        # shared Q(s,anchor,z) with a z=0 switch (halves Q compute + recurrent state) --
        # the A9-efficiency A/B axis.
        separate_skill0_q=True,
        skill_zero_norm_eps=1e-6,
    )
    return config


def make_encoder_update_burst(skill, archive, *, batch_size: int, update_steps: int,
                              preprocessed: bool = False):
    """Execute the post-Q encoder burst with fresh samples and optimizer state."""
    if update_steps <= 0:
        raise ValueError("encoder burst requires at least one update")

    @jax.jit
    def burst(skill_state, archive_state, key):
        def one_update(carry, _):
            state, rng = carry
            rng, sample_key, update_key = jax.random.split(rng, 3)
            batch, archive_info = archive.sample_encoder_batch(archive_state, sample_key, batch_size)
            state, metrics = (skill.update_preprocessed(state, batch, update_key)
                              if preprocessed else skill.update(state, batch, update_key))
            return (state, rng), (metrics, archive_info)

        (skill_state, key), infos = jax.lax.scan(
            one_update, (skill_state, key), None, length=update_steps
        )
        metrics, archive_info = jax.tree_util.tree_map(lambda values: values.mean(axis=0), infos)
        return skill_state, key, metrics, archive_info

    return burst


def make_frozen_reward_fn(skill: SkillLearner, discount: float, zero_eps: float):
    """Build a jitted frozen-reward assembler closed over the SkillLearner.

    frozen = (1 - gamma) * [ skill_on ? intrinsic(s,s',anchor,z) : E_z max_a Q_skill(s,s,z) (potential value) ]
    stop-gradient. Byte-faithful to potential_pqn.py::_prepare_rollout_batch's reward
    line, with the intrinsic term routed through warp's SkillLearner (documented
    deviation: encoder external) and the potential term through the agent's skill Q (`effective_q`).
    """
    norm_factor = 1.0 - float(discount)

    @jax.jit
    def frozen_reward(agent, skill_state, observations, next_observations, anchors, skills, timesteps, key,
                      batch=None, initial_state=None, partition_weight=None):
        intrinsic_key, value_key = jax.random.split(key)
        reward_batch = SkillBatch(
            observations=observations,
            next_observations=next_observations,
            future_observations=next_observations,  # unused by the potential reward path
            anchor_observations=anchors,
            skills=skills,
            future_valid=jnp.zeros((observations.shape[0],), dtype=jnp.float32),
        )
        intrinsic = (skill.intrinsic_reward(skill_state, reward_batch, intrinsic_key)
                     if partition_weight is None else
                     skill.intrinsic_reward(skill_state, reward_batch, intrinsic_key,
                                            partition_weight=partition_weight))
        # Potential: mean over Monte Carlo unit-sphere skill draws at anchor=current s.
        # The agent uses fresh memory by default, or replays the observed history
        # when pqn_preserve_memory_at_switch is enabled.
        potential = agent.potential_values(observations, batch=batch, initial_state=initial_state, key=value_key)
        skill_on = jnp.linalg.norm(skills, axis=-1) > zero_eps
        frozen = norm_factor * jnp.where(skill_on, intrinsic, potential)
        info = {
            "reward/partition_weight": jnp.asarray(1.0 if partition_weight is None else partition_weight),
            "reward/empowerment_mean": intrinsic.mean(),
            "reward/potential_value_mean": potential.mean(),
            "reward/potential_to_skill_ratio": potential.mean() / (
                jnp.sum(jnp.where(skill_on, intrinsic, 0.0)) / jnp.maximum(jnp.sum(skill_on), 1.0) + 1e-6
            ),
            "reward/pqn_reward_mean": frozen.mean(),
            "reward/skill_on_frac": skill_on.astype(jnp.float32).mean(),
        }
        return jax.lax.stop_gradient(frozen), info

    return frozen_reward
