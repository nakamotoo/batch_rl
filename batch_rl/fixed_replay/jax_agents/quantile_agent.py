# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An extension of Rainbow to perform quantile regression.
This loss is computed as in "Distributional Reinforcement Learning with Quantile
Regression" - Dabney et. al, 2017"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import logging
from dopamine.jax import networks
from dopamine.jax.agents.quantile import quantile_agent
from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow.compat.v1 as tf


def compute_pessimistic_q_values(logits: jnp.ndarray, truncate_c: float) -> jnp.ndarray:
  tau = int(truncate_c * logits.shape[-1])
  return jnp.mean(outputs.logits[..., :tau], axis=-1)


def conservative_q_loss(q_values: jnp.ndarray, chosen_action_q: jnp.ndarray) -> jnp.ndarray:
  """Implementation of the CQL loss."""
  logsumexp_q = jax.scipy.special.logsumexp(q_values)
  return logsumexp_q - chosen_action_q


@functools.partial(jax.jit, static_argnums=(2,))
def calibrated_ent_loss(optimal_action_logits: jnp.ndarray,
                        chosen_action_logits: jnp.ndarray, k: int) -> jnp.ndarray:
  """Implementation of entropy calibration loss."""
  optimal_diffs = optimal_action_logits[None] - optimal_action_logits[..., None]
  optimal_action_ent = jnp.sum(jnp.tril(jnp.triu(optimal_diffs), k=k), axis=0)
  optimal_action_ent = -jnp.mean(jnp.log(1 + optimal_action_ent / k))
                              
  chosen_diffs = chosen_action_logits[None] - chosen_action_logits[..., None]
  chosen_action_ent = jnp.sum(jnp.tril(jnp.triu(chosen_diffs), k=k), axis=0)
  chosen_action_ent = -jnp.mean(jnp.log(1 + chosen_action_ent / k))

  return chosen_action_ent - optimal_action_ent


@functools.partial(jax.jit, static_argnums=(0, 3, 11, 12, 13, 14, 15, 16, 17))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, loss_weights,
          kappa, num_atoms, cumulative_gamma,
          cql_penalty_weight=0.0, ent_penalty_weight=0.0, ent_k=20, truncate_c=1.0):
  """Run a training step."""
  def loss_fn(params, target, loss_multipliers):
    def q_online(state):
      return network_def.apply(params, state)

    outputs = jax.vmap(q_online)(states)
    logits = jnp.squeeze(outputs.logits)
    # Fetch the logits for its selected action. We use vmap to perform this
    # indexing across the batch.
    chosen_action_logits = jax.vmap(lambda x, y: x[y])(logits, actions)
    bellman_errors = (target[:, None, :] -
                      chosen_action_logits[:, :, None])  # Input `u' of Eq. 9.
    # Eq. 9 of paper.
    huber_loss = (
        (jnp.abs(bellman_errors) <= kappa).astype(jnp.float32) *
        0.5 * bellman_errors ** 2 +
        (jnp.abs(bellman_errors) > kappa).astype(jnp.float32) *
        kappa * (jnp.abs(bellman_errors) - 0.5 * kappa))

    tau_hat = ((jnp.arange(num_atoms, dtype=jnp.float32) + 0.5) /
               num_atoms)  # Quantile midpoints.  See Lemma 2 of paper.
    # Eq. 10 of paper.
    tau_bellman_diff = jnp.abs(
        tau_hat[None, :, None] - (bellman_errors < 0).astype(jnp.float32))
    quantile_huber_loss = tau_bellman_diff * huber_loss
    # Sum over tau dimension, average over target value dimension.
    loss = jnp.sum(jnp.mean(quantile_huber_loss, 2), 1)
    q_loss = jnp.mean(loss_multipliers * loss)

    # Add CQL loss.
    chosen_action_q = jax.vmap(lambda x, y: x[y])(outputs.q_values, actions)
    cql_loss = jnp.mean(
      loss_multipliers * jax.vmap(conservative_q_loss)(outputs.q_values, chosen_action_q))

    # Add entropy calibration loss.
    ent_loss_fn = functools.partial(calibrated_ent_loss, k=ent_k)
    pessimistic_q_values = compute_pessimistic_q_values(outputs.logits, truncate_c)
    optimal_actions = jnp.argmax(pessimistic_q_values, axis=-1)
    optimal_action_logits = jax.vmap(lambda x, y: x[y])(logits, optimal_actions)
    ent_loss = jnp.mean(
      loss_multipliers * jax.vmap(ent_loss_fn)(optimal_action_logits, chosen_action_logits))

    mean_loss =  q_loss + cql_penalty_weight * cql_loss + ent_penalty_weight * ent_loss
    return mean_loss, (loss, q_loss, cql_loss, ent_loss)

  def q_target(state):
    return network_def.apply(target_params, state)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  target = quantile_agent.target_distribution(q_target,
                                              next_states,
                                              rewards,
                                              terminals,
                                              cumulative_gamma)
  (_, (loss, q_loss, cql_loss, ent_loss)), grad = grad_fn(online_params, target, loss_weights)
  updates, optimizer_state = optimizer.update(grad, optimizer_state,
                                              params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, q_loss, cql_loss, ent_loss


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 10, 11, 12))
def select_action(network_def, params, state, rng, num_actions, eval_mode,
                  epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn, truncate_c=1.0):
  """Select an action from the set of available actions.
  Chooses an action randomly with probability self._calculate_epsilon(), and
  otherwise acts greedily according to the current Q-value estimates.
  Args:
    network_def: Linen Module to use for inference.
    params: Linen params (frozen dict) to use for inference.
    state: input state to use for inference.
    rng: Jax random number generator.
    num_actions: int, number of actions (static_argnum).
    eval_mode: bool, whether we are in eval mode (static_argnum).
    epsilon_eval: float, epsilon value to use in eval mode (static_argnum).
    epsilon_train: float, epsilon value to use in train mode (static_argnum).
    epsilon_decay_period: float, decay period for epsilon value for certain
      epsilon functions, such as linearly_decaying_epsilon, (static_argnum).
    training_steps: int, number of training steps so far.
    min_replay_history: int, minimum number of steps in replay buffer
      (static_argnum).
    epsilon_fn: function used to calculate epsilon value (static_argnum).
    truncate_c: float, proportion of lower quantiles used for pessimistic Q-values.
  Returns:
    rng: Jax random number generator.
    action: int, the selected action.
  """
  epsilon = jnp.where(eval_mode,
                      epsilon_eval,
                      epsilon_fn(epsilon_decay_period,
                                 training_steps,
                                 min_replay_history,
                                 epsilon_train))

  rng, rng1, rng2 = jax.random.split(rng, num=3)
  p = jax.random.uniform(rng1)
  outputs = network_def.apply(params, state)
  pessimistic_q_values = compute_pessimistic_q_values(outputs.logits, truncate_c)
  return rng, jnp.where(p <= epsilon,
                        jax.random.randint(rng2, (), 0, num_actions),
                        jnp.argmax(pessimistic_q_values))


@gin.configurable
class FixedReplayJaxQuantileAgent(quantile_agent.JaxQuantileAgent):
  """An implementation of Quantile regression DQN agent."""

  def __init__(self,
               num_actions,
               replay_data_dir,
               replay_suffix=None,
               init_checkpoint_dir=None,
               cql_penalty_weight=0.0,
               ent_penalty_weight=0.0,
               ent_k=20,
               truncate_c=1.0,               
               **kwargs):
    """Initializes the agent and constructs the Graph.
    Args:
      num_actions: int, number of actions the agent can take at any state.
      replay_data_dir: str, log Directory from which to load the replay buffer.
      replay_suffix: int, If not None, then only load the replay buffer
        corresponding to the specific suffix in data directory.
      init_checkpoint_dir: str, directory from which initial checkpoint before
        training is loaded if there doesn't exist any checkpoint in the current
        agent directory. If None, no initial checkpoint is loaded
      cql_penalty_weight: float, weight for cql loss.
      ent_penalty_weight: float, weight for entropy loss.
      ent_k: int, number of k-nearest neighbors in entropy computation.
      truncate_c: float, proportion of lower quantiles used for pessimistic Q-values.
      **kwargs: Arbitrary keyword arguments.
    """
    assert replay_data_dir is not None
    logging.info(
        'Creating FixedReplayJaxAgent with replay directory: %s', replay_data_dir)
    logging.info('\t init_checkpoint_dir %s', init_checkpoint_dir)
    logging.info('\t replay_suffix %s', replay_suffix)
    # Set replay_log_dir before calling parent's initializer
    self._replay_data_dir = replay_data_dir
    self._replay_suffix = replay_suffix
    if init_checkpoint_dir is not None:
      self._init_checkpoint_dir = os.path.join(
          init_checkpoint_dir, 'checkpoints')
    else:
      self._init_checkpoint_dir = None
    self.cql_penalty_weight = cql_penalty_weight
    self.ent_penalty_weight = ent_penalty_weight
    self.ent_k = ent_k
    self.truncate_c = truncate_c
    super(FixedReplayJaxQuantileAgent, self).__init__(num_actions, **kwargs)

  def end_episode(self, reward, terminal=True):
    assert self.eval_mode, 'Eval mode is not set to be True.'
    super(FixedReplayJaxQuantileAgent, self).end_episode(reward, terminal)

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""
    if self._replay_scheme not in ['uniform', 'prioritized']:
      raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
    # Both replay schemes use the same data structure, but the 'uniform' scheme
    # sets all priorities to the same value (which yields uniform sampling).
    return fixed_replay_buffer.FixedReplayBuffer(
        data_dir=self._replay_data_dir,
        replay_suffix=self._replay_suffix,
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype)

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.
    Args:
      observation: numpy array, the environment's initial observation.
    Returns:
      int, the selected action.
    """
    self._reset_state()
    self._record_observation(observation)

    if not self.eval_mode:
      self._train_step()

    self._rng, self.action = select_action(self.network_def,
                                           self.online_params,
                                           self.state,
                                           self._rng,
                                           self.num_actions,
                                           self.eval_mode,
                                           self.epsilon_eval,
                                           self.epsilon_train,
                                           self.epsilon_decay_period,
                                           self.training_steps,
                                           self.min_replay_history,
                                           self.epsilon_fn,
                                           self.truncate_c)
    self.action = onp.asarray(self.action)
    return self.action

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.
    We store the observation of the last time step since we want to store it
    with the reward.
    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.
    Returns:
      int, the selected action.
    """
    self._last_observation = self._observation
    self._record_observation(observation)

    if not self.eval_mode:
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    self._rng, self.action = select_action(self.network_def,
                                           self.online_params,
                                           self.preprocess_fn(self.state),
                                           self._rng,
                                           self.num_actions,
                                           self.eval_mode,
                                           self.epsilon_eval,
                                           self.epsilon_train,
                                           self.epsilon_decay_period,
                                           self.training_steps,
                                           self.min_replay_history,
                                           self.epsilon_fn,
                                           self.truncate_c)
    self.action = onp.asarray(self.action)
    return self.action

  def _train_step(self):
    """Runs a single training step.
    Runs training if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.
    Also, syncs weights from online_params to target_network_params if training
    steps is a multiple of target update period.
    """
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()

        if self._replay_scheme == 'prioritized':
          # The original prioritized experience replay uses a linear exponent
          # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
          # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
          # suggested a fixed exponent actually performs better, except on Pong.
          probs = self.replay_elements['sampling_probabilities']
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
        else:
          loss_weights = jnp.ones(self.replay_elements['state'].shape[0])

        self.optimizer_state, self.online_params, loss, q_loss, cql_loss, ent_loss = train(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            self.preprocess_fn(self.replay_elements['state']),
            self.replay_elements['action'],
            self.preprocess_fn(self.replay_elements['next_state']),
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            loss_weights,
            self._kappa,
            self._num_atoms,
            self.cumulative_gamma,
            self.cql_penalty_weight,
            self.ent_penalty_weight,
            self.ent_k,
            self.truncate_c)

        if self._replay_scheme == 'prioritized':
          # Rainbow and prioritized replay are parametrized by an exponent
          # alpha, but in both cases it is set to 0.5 - for simplicity's sake we
          # leave it as is here, using the more direct sqrt(). Taking the square
          # root "makes sense", as we are dealing with a squared loss.  Add a
          # small nonzero value to the loss to avoid 0 priority items. While
          # technically this may be okay, setting all items to 0 priority will
          # cause troubles, and also result in 1.0 / 0.0 = NaN correction terms.
          self._replay.set_priority(self.replay_elements['indices'],
                                    jnp.sqrt(loss + 1e-10))

        if self.summary_writer is not None:
          summary = tf.Summary(value=[
              tf.Summary.Value(tag='Losses/QuantileLoss', simple_value=q_loss),
              tf.Summary.Value(tag='Losses/CQLPenalty', simple_value=cql_loss),
              tf.Summary.Value(tag='Losses/EntropyPenalty', simple_value=ent_loss)])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
