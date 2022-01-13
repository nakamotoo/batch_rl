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
from dopamine.jax.agents.rainbow import rainbow_agent
from dopamine.jax import losses
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow.compat.v1 as tf


def compute_optimistic_q_values(q_values: jnp.ndarray,
                                probabilities: jnp.ndarray,
                                support: jnp.ndarray,
                                std_c: float) -> jnp.ndarray:
  # Compute standard deviation as sqrt(E[Q^2] - E[Q]^2)
  q_values_std = jnp.sqrt(
      jnp.sum(jnp.square(support) * probabilities, axis=-1) -
      jnp.square(q_values)
  )
  return q_values + std_c * q_values_std


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
  optimistic_q_values = compute_optimistic_q_values(
      outputs.q_values, outputs.probabilities, support, std_c)
  return rng, jnp.where(
      p <= epsilon,
      jax.random.randint(rng2, (), 0, num_actions),
      jnp.argmax(optimistic_q_values))


@gin.configurable
class ExplorationJaxRainbowAgent(rainbow_agent.JaxRainbowAgent):
  """An implementation of the rainbow agent with UCB exploration."""

  def __init__(self,
               num_actions,
               std_c=0.0,
               **kwargs):
    """Initializes the agent and constructs the components of its graph.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      std_c: float, weight for standard deviation for pessimistic Q-values.
      **kwargs: Arbitrary keyword arguments.
    """
    logging.info('Creating ExplorationJaxRainbowAgent')
    self.std_c = std_c
    super(ExplorationJaxRainbowAgent, self).__init__(num_actions, **kwargs)

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
