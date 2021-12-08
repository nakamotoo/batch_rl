# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""C51 agent with fixed replay buffer(s)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl import logging
from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from dopamine.jax.agents.rainbow import rainbow_agent
from dopamine.jax import losses
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow.compat.v1 as tf


def conservative_q_loss(q_values: jnp.ndarray, chosen_action_q: jnp.ndarray) -> jnp.ndarray:
  """Implementation of the CQL loss."""
  logsumexp_q = jax.scipy.special.logsumexp(q_values)
  return logsumexp_q - chosen_action_q


def calibrated_ent_loss(optimal_action_probs: jnp.ndarray,
                        chosen_action_probs: jnp.ndarray) -> jnp.ndarray:
  """Implementation of entropy calibration loss."""
  return (
      jnp.sum(optimal_action_probs * jnp.log(optimal_action_probs)) -
      jnp.sum(chosen_action_probs * jnp.log(chosen_action_probs))
  )


@functools.partial(jax.jit, static_argnums=(0, 3, 12, 13, 14))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, loss_weights,
          support, cumulative_gamma, cql_penalty_weight=0.0, ent_penalty_weight=0.0):
  """Run a training step."""
  def loss_fn(params, target, loss_multipliers):
    def q_online(state):
      return network_def.apply(params, state, support)

    outputs = jax.vmap(q_online)(states)
    # Fetch the logits for its selected action. We use vmap to perform this
    # indexing across the batch.
    chosen_action_logits = jax.vmap(lambda x, y: x[y])(outputs.logits, actions)
    loss = jax.vmap(losses.softmax_cross_entropy_loss_with_logits)(
        target,
        chosen_action_logits)
    ce_loss = jnp.mean(loss_multipliers * loss) 

    chosen_action_q = jax.vmap(lambda x, y: x[y])(outputs.q_values, actions)
    cql_loss = jnp.mean(
      loss_multipliers * jax.vmap(conservative_q_loss)(outputs.q_values, chosen_action_q))
    
    optimal_actions = jnp.argmax(outputs.q_values, axis=-1)
    optimal_action_probs = jax.vmap(lambda x, y: x[y])(outputs.probabilities, optimal_actions)
    chosen_action_probs = jax.vmap(lambda x, y: x[y])(outputs.probabilities, actions)
    ent_loss = jnp.mean(
      loss_multipliers * jax.vmap(calibrated_ent_loss)(optimal_action_probs, chosen_action_probs))

    mean_loss =  ce_loss + cql_penalty_weight * cql_loss + ent_penalty_weight * ent_loss
    return mean_loss, (loss, ce_loss, cql_loss, ent_loss)

  def q_target(state):
    return network_def.apply(target_params, state, support)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  target = rainbow_agent.target_distribution(q_target,
                                             next_states,
                                             rewards,
                                             terminals,
                                             support,
                                             cumulative_gamma)

  # Get the unweighted loss without taking its mean for updating priorities.
  (_, (loss, ce_loss, cql_loss, ent_loss)), grad = grad_fn(online_params, target, loss_weights)
  updates, optimizer_state = optimizer.update(grad, optimizer_state,
                                              params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, loss, ce_loss, cql_loss, ent_loss


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 10, 11, 13))
def select_action(network_def, params, state, rng, num_actions, eval_mode,
                  epsilon_eval, epsilon_train, epsilon_decay_period,
                  training_steps, min_replay_history, epsilon_fn, support, std_c):
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
    support: support for the distribution.
    std_c: float, weight for standard deviation for pessimistic Q-values.
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

  outputs = network_def.apply(params, state, support)
  # Compute standard deviation as sqrt(E[Q^2] - E[Q]^2)
  q_values_std = jnp.sqrt(
      jnp.sum(jnp.square(support) * outputs.probabilities, axis=-1) -
      jnp.square(outputs.q_values)
  )
  pessimistic_q_values = outputs.q_values - std_c * q_values_std
  return rng, jnp.where(
      p <= epsilon,
      jax.random.randint(rng2, (), 0, num_actions),
      jnp.argmax(pessimistic_q_values))


@gin.configurable
class FixedReplayJaxRainbowAgent(rainbow_agent.JaxRainbowAgent):
  """An implementation of the DQN agent with fixed replay buffer(s)."""

  def __init__(self,
               num_actions,
               replay_data_dir,
               replay_suffix=None,
               init_checkpoint_dir=None,
               cql_penalty_weight=0.0,
               ent_penalty_weight=0.0,
               std_c=0.0,
               **kwargs):
    """Initializes the agent and constructs the components of its graph.

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
      std_c: float, weight for standard deviation for pessimistic Q-values.
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
    self.std_c = std_c
    super(FixedReplayJaxRainbowAgent, self).__init__(num_actions, **kwargs)

  def end_episode(self, reward, terminal=True):
    assert self.eval_mode, 'Eval mode is not set to be True.'
    super(FixedReplayJaxRainbowAgent, self).end_episode(reward, terminal)

  def _build_replay_buffer(self):
    """Creates the replay buffer used by the agent."""

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
                                           self._support,
                                           self.std_c)
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
                                           self._support,
                                           self.std_c)
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
          # Weight the loss by the inverse priorities.
          loss_weights = 1.0 / jnp.sqrt(probs + 1e-10)
          loss_weights /= jnp.max(loss_weights)
        else:
          loss_weights = jnp.ones(self.replay_elements['state'].shape[0])

        self.optimizer_state, self.online_params, loss, ce_loss, cql_loss, ent_loss = train(
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
            self._support,
            self.cumulative_gamma,
            self.cql_penalty_weight,
            self.ent_penalty_weight)

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
              tf.Summary.Value(tag='Losses/CrossEntropyLoss', simple_value=ce_loss),
              tf.Summary.Value(tag='Losses/CQLPenalty', simple_value=cql_loss),
              tf.Summary.Value(tag='Losses/EntropyPenalty', simple_value=ent_loss)])
          self.summary_writer.add_summary(summary, self.training_steps)
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
