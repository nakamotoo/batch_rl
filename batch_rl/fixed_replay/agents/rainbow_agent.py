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

import os

from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from dopamine.agents.rainbow import rainbow_agent
import gin
import tensorflow.compat.v1 as tf


@gin.configurable
class FixedReplayRainbowAgent(rainbow_agent.RainbowAgent):
  """An implementation of the DQN agent with fixed replay buffer(s)."""

  def __init__(self,
               sess,
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
      sess: tf.Session, for executing ops.
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
    tf.logging.info(
        'Creating FixedReplayAgent with replay directory: %s', replay_data_dir)
    tf.logging.info('\t init_checkpoint_dir %s', init_checkpoint_dir)
    tf.logging.info('\t replay_suffix %s', replay_suffix)
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

    super(FixedReplayRainbowAgent, self).__init__(sess, num_actions, **kwargs)

  def step(self, reward, observation):
    """Records the most recent transition and returns the agent's next action.

    Args:
      reward: float, the reward received from the agent's most recent action.
      observation: numpy array, the most recent observation.

    Returns:
      int, the selected action.
    """
    self._record_observation(observation)
    self.action = self._select_action()
    return self.action

  def end_episode(self, reward):
    assert self.eval_mode, 'Eval mode is not set to be True.'
    super(FixedReplayRainbowAgent, self).end_episode(reward)

  def _build_replay_buffer(self, use_staging):
    """Creates the replay buffer used by the agent."""

    return fixed_replay_buffer.WrappedFixedReplayBuffer(
        data_dir=self._replay_data_dir,
        replay_suffix=self._replay_suffix,
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

  def _build_train_op(self):
    """Builds a training op.
    Returns:
      train_op: An op performing one step of training from replay data.
    """
    target_distribution = tf.stop_gradient(self._build_target_distribution())

    # size of indices: batch_size x 1.
    indices = tf.range(tf.shape(self._replay_net_outputs.logits)[0])[:, None]
    # size of reshaped_actions: batch_size x 2.
    reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1)
    # For each element of the batch, fetch the logits for its selected action.
    chosen_action_logits = tf.gather_nd(self._replay_net_outputs.logits,
                                        reshaped_actions)

    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=target_distribution,
        logits=chosen_action_logits)

    # Compute CQL penalty.
    logsumexp_q = tf.reduce_logsumexp(self._replay_net_outputs.q_values, axis=1,
                                      name='logsumexp_q')
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot, axis=1,
        name='replay_chosen_q')
    cql_loss = logsumexp_q - replay_chosen_q

    # Compute entropy penalty.
    optimal_actions = tf.cast(
      tf.argmax(self._replay_net_outputs.q_values, axis=1), dtype=tf.int32)
    reshaped_optimal_actions = tf.concat([indices, optimal_actions[:, None]], 1)    
    optimal_action_probs = tf.gather_nd(self._replay_net_outputs.probabilities,
                                        reshaped_optimal_actions)
    chosen_action_probs = tf.gather_nd(self._replay_net_outputs.probabilities,
                                       reshaped_actions)
    ent_loss = (
      tf.reduce_sum(optimal_action_probs * tf.math.log(optimal_action_probs), axis=1) -
      tf.reduce_sum(chosen_action_probs * tf.math.log(chosen_action_probs), axis=1)
    )
  
    if self._replay_scheme == 'prioritized':
      # The original prioritized experience replay uses a linear exponent
      # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
      # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
      # a fixed exponent actually performs better, except on Pong.
      probs = self._replay.transition['sampling_probabilities']
      loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
      loss_weights /= tf.reduce_max(loss_weights)

      # Rainbow and prioritized replay are parametrized by an exponent alpha,
      # but in both cases it is set to 0.5 - for simplicity's sake we leave it
      # as is here, using the more direct tf.sqrt(). Taking the square root
      # "makes sense", as we are dealing with a squared loss.
      # Add a small nonzero value to the loss to avoid 0 priority items. While
      # technically this may be okay, setting all items to 0 priority will cause
      # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
      update_priorities_op = self._replay.tf_set_priority(
          self._replay.indices, tf.sqrt(loss + 1e-10))

      # Weight the loss by the inverse priorities.
      loss = loss_weights * loss
      cql_loss = loss_weights * cql_loss
      ent_loss = loss_weights * ent_loss
    else:
      update_priorities_op = tf.no_op()

    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.compat.v1.variable_scope('Losses'):
          tf.compat.v1.summary.scalar('CrossEntropyLoss', tf.reduce_mean(loss))
          tf.compat.v1.summary.scalar('CQLPenalty', tf.reduce_mean(cql_loss))
          tf.compat.v1.summary.scalar('EntropyPenalty', tf.reduce_mean(ent_loss))
      # Schaul et al. reports a slightly different rule, where 1/N is also
      # exponentiated by beta. Not doing so seems more reasonable, and did not
      # impact performance in our experiments.
      loss = loss + self.cql_penalty_weight * cql_loss + self.ent_penalty_weight * ent_loss
      return self.optimizer.minimize(tf.reduce_mean(loss)), loss

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.
    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """

    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    self._net_outputs = self.online_convnet(self.state_ph)
    # TODO(bellemare): Ties should be broken. They are unlikely to happen when
    # using a deep network, but may affect performance with a linear
    # approximation scheme.
    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)

    # Compute standard deviation as sqrt(E[Q^2] - E[Q]^2)
    q_values_std = tf.math.sqrt(
      tf.reduce_sum(
        tf.math.square(self._support) * self._net_outputs.probabilities, axis=2) -
      tf.math.square(self._net_outputs.q_values)
    )
    q_values = self._net_outputs.q_values - self.std_c * q_values_std
    self._q_argmax = tf.argmax(q_values, axis=1)[0]
