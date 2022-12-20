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

"""DQN agent with fixed replay buffer(s)."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import functools
import os

from absl import logging
from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.jax import losses
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow.compat.v1 as tf


def conservative_q_loss(q_values: jnp.ndarray, replay_chosen_q: jnp.ndarray) -> jnp.ndarray:
  """Implementation of the CQL loss."""
  logsumexp_q = jax.scipy.special.logsumexp(q_values)
  return logsumexp_q - replay_chosen_q
  

@functools.partial(jax.jit, static_argnums=(0, 3, 10, 11, 12))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          states, actions, next_states, rewards, terminals, cumulative_gamma,
          cql_weight_penalty=0.1, loss_type='huber'):
  """Run the training step."""
  def loss_fn(params, target):

    def q_online(state):
      return network_def.apply(params, state)

    q_values = jax.vmap(q_online)(states).q_values
    q_values = jnp.squeeze(q_values)
    replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, actions)
    if loss_type == 'huber':
      loss = jnp.mean(jax.vmap(losses.huber_loss)(target, replay_chosen_q))
    else:
      loss = jnp.mean(jax.vmap(losses.mse_loss)(target, replay_chosen_q))

    # Add CQL penalty
    cql_loss = jnp.mean(jax.vmap(conservative_q_loss)(q_values, replay_chosen_q))

    log_metrics = {
      "bellman_loss": loss,
      "cql_loss": cql_loss,
      "replay_chosen_q": replay_chosen_q.mean(),
      "logsumexp_q": jax.scipy.special.logsumexp(q_values)
    }

    return loss + cql_weight_penalty * cql_loss, log_metrics

  def q_target(state):
    return network_def.apply(target_params, state)

  target = dqn_agent.target_q(q_target,
                              next_states,
                              rewards,
                              terminals,
                              cumulative_gamma)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, log_metrics), grad = grad_fn(online_params, target)
  updates, optimizer_state = optimizer.update(grad, optimizer_state,
                                              params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  return optimizer_state, online_params, log_metrics

@gin.configurable
class FixedReplayJaxDQNAgent(dqn_agent.JaxDQNAgent):
  """An implementation of the DQN agent with fixed replay buffer(s)."""

  def __init__(self,
               num_actions,
               replay_data_dir,
               replay_suffix=None,
               init_checkpoint_dir=None,
               cql_penalty_weight=0.0,
               **kwargs):
    """Initializes the agent and constructs the components of its graph.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      replay_data_dir: str, log Directory from which to load the replay buffer.
      replay_suffix: int, If not None, then only load the replay buffer
        corresponding to the specific suffix in data directory.
      init_checkpoint_dir: str, directory from which initial checkpoint before
        training is loaded if there doesn't exist any checkpoint in the current
        agent directory. If None, no initial checkpoint is loaded.
      cql_penalty_weight: float, weight for cql loss.
      **kwargs: Arbitrary keyword arguments.
    """
    assert replay_data_dir is not None
    logging.info(
      'Creating FixedReplayJaxAgent with replay directory: %s', replay_data_dir)
    logging.info('\t init_checkpoint_dir %s', init_checkpoint_dir)
    logging.info('\t replay_suffix %s', replay_suffix)
    print("====================")
    print("cql_penalty_weight", cql_penalty_weight)
    print(kwargs)
    print("====================")
    # Set replay_log_dir before calling parent's initializer
    self._replay_data_dir = replay_data_dir
    self._replay_suffix = replay_suffix
    if init_checkpoint_dir is not None:
      self._init_checkpoint_dir = os.path.join(
          init_checkpoint_dir, 'checkpoints')
    else:
      self._init_checkpoint_dir = None
    self.cql_penalty_weight = cql_penalty_weight
    super(FixedReplayJaxDQNAgent, self).__init__(num_actions, **kwargs)

  def end_episode(self, reward, terminal=True):
    assert self.eval_mode, 'Eval mode is not set to be True.'
    super(FixedReplayJaxDQNAgent, self).end_episode(reward, terminal)

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

  def _train_step(self):
    """Runs a single training step."""

    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sample_from_replay_buffer()
        states = self.preprocess_fn(self.replay_elements['state'])
        next_states = self.preprocess_fn(self.replay_elements['next_state'])
        self.optimizer_state, self.online_params, log_metrics = train(
            self.network_def,
            self.online_params,
            self.target_network_params,
            self.optimizer,
            self.optimizer_state,
            states,
            self.replay_elements['action'],
            next_states,
            self.replay_elements['reward'],
            self.replay_elements['terminal'],
            self.cumulative_gamma,
            self.cql_penalty_weight,
            self._loss_type)

        if (self.summary_writer is not None and
            self.training_steps > 0 and
            self.training_steps % self.summary_writing_frequency == 0):
          summary = tf.Summary(value=[tf.Summary.Value(tag=f'Losses/{key}', simple_value=value) for key, value in log_metrics.items() ])
          self.summary_writer.add_summary(summary, self.training_steps)
          self.summary_writer.flush()
      if self.training_steps % self.target_update_period == 0:
        self._sync_weights()

    self.training_steps += 1
