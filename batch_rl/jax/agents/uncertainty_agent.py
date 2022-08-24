"""Compact implementation of an offline uncertainty agent in JAX."""

import collections
import functools

from absl import logging
from dopamine.jax.agents.dqn import dqn_agent as base_dqn_agent
import gin
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tensorflow as tf
from batch_rl.jax import networks


@gin.configurable
@functools.partial(jax.jit, static_argnums=(0, 2, 3))
def linearly_decaying_value(decay_period, step, initial_value, final_value):
  """Returns the value based on current step."""
  shift = jnp.clip(step * 1.0 / decay_period, 0, 1.0)
  return (final_value - initial_value) * shift + initial_value


def compute_dr3_loss(state_representations, next_state_representations):
  """Minimizes dot product between state and next state representations."""
  dot_products = jnp.einsum('ij,ij->i', state_representations,
                            next_state_representations)
  # Minimize |\phi(s) \phi(s')|
  return jnp.mean(jnp.abs(dot_products))


@functools.partial(
    jax.vmap,
    in_axes=(None, None, None, 0, 0, 0, 0, None, None, None, 0),
    out_axes=(0, 0, 0, 0))
def target_delta_values(network_def, online_params, target_params, next_states,
                        rewards, terminals, deltas, num_delta_prime_samples,
                        cumulative_gamma, double_dqn, rng):
  """Build the targets for return values at given deltas.

  Args:
    network_def: Linen Module used for inference.
    online_params: Parameters used for the online network.
    target_params: Parameters used for the target network.
    next_states: numpy array of batched next states.
    rewards: numpy array of batched rewards.
    terminals: numpy array of batched terminals.
    deltas: numpy array of batched deltas.
    num_delta_prime_samples: int, number of delta' samples (static_argnum).
    cumulative_gamma: float, cumulative gamma to use (static_argnum).
    double_dqn: bool, whether to use double DQN (static_argnum).
    rng: Jax random number generator.

  Returns:
    Jax random number generator.
    The target delta values.
    Randomly sampled deltas to be used in backup.
  """
  num_deltas = deltas.shape[0]
  rng, rng1 = jax.random.split(rng, num=2)
  delta_samples = (deltas[:, None] - 1e-3) * jax.random.uniform(
      rng1, shape=[num_deltas, num_delta_prime_samples])
  delta_samples = delta_samples.reshape((-1))

  rewards = jnp.tile(rewards, [num_deltas * num_delta_prime_samples])
  is_terminal_multiplier = 1. - terminals.astype(jnp.float32)
  # Incorporate terminal state to discount factor.
  gamma_with_terminal = cumulative_gamma * is_terminal_multiplier
  gamma_with_terminal = jnp.tile(gamma_with_terminal,
                                 [num_deltas * num_delta_prime_samples])

  # Compute Q-values which are used for action selection for the next states
  # in the replay buffer. Compute the argmax over the Q-values.
  if double_dqn:
    next_state_target_action = network_def.apply(
        online_params, next_states, delta_samples / network_def.num_actions)
  else:
    next_state_target_action = network_def.apply(
        target_params, next_states, delta_samples / network_def.num_actions)
  # Get the indices of the maximium Q-value across the action dimension.
  next_state_target = network_def.apply(target_params, next_states,
                                        delta_samples / network_def.num_actions)

  # Compute bellman loss for upper-bound delta values.
  # --------------------------------------------------------------------------
  # Shape: num_deltas * num_delta_prime_samples.
  ub_next_qt_argmax = jnp.argmax(
      next_state_target_action.ub_delta_values, axis=-1)
  target_ub_delta_vals = (
      jax.vmap(lambda x, y: x[y])(next_state_target.ub_delta_values,
                                  ub_next_qt_argmax))
  target_ub_delta_vals = jnp.clip(
      rewards + gamma_with_terminal * target_ub_delta_vals, a_min=0, a_max=1e2)
  target_ub_delta_vals = target_ub_delta_vals.reshape(
      (num_deltas, num_delta_prime_samples))

  # Compute target values for lower-bound delta values.
  # --------------------------------------------------------------------------
  # Shape: num_deltas * num_delta_prime_samples.
  lb_next_qt_argmax = jnp.argmax(
      next_state_target_action.lb_delta_values, axis=-1)
  target_lb_delta_vals = (
      jax.vmap(lambda x, y: x[y])(next_state_target.lb_delta_values,
                                  lb_next_qt_argmax))
  target_lb_delta_vals = jnp.clip(
      rewards + gamma_with_terminal * target_lb_delta_vals, a_min=0, a_max=1e2)
  target_lb_delta_vals = target_lb_delta_vals.reshape(
      (num_deltas, num_delta_prime_samples))

  # We return with an extra dimension, which is expected by train.
  return (rng, jax.lax.stop_gradient(target_ub_delta_vals),
          jax.lax.stop_gradient(target_lb_delta_vals),
          delta_samples.reshape(num_deltas, num_delta_prime_samples))


@functools.partial(
    jax.jit, static_argnums=(0, 3, 5, 8, 15, 16, 17, 18, 19, 20, 21))
def train(network_def, online_params, target_params, optimizer, optimizer_state,
          rnd_network_def, rnd_online_params, rnd_target_params, rnd_optimizer,
          rnd_optimizer_state, states, actions, next_states, rewards, terminals,
          num_delta_samples, num_delta_prime_samples, dr3_alpha, lcb_alpha, 
          lcb_delta, cumulative_gamma, double_dqn, rng):
  """Run a training step."""
  batch_size = states.shape[0]
  rng, delta_rng = jax.random.split(rng, num=2)
  deltas = jax.random.uniform(delta_rng, shape=[batch_size, num_delta_samples])

  def rnd_loss_fn(rnd_params):

    def rnd(state):
      return rnd_network_def.apply(rnd_params, state)

    def target_rnd(state):
      return rnd_network_def.apply(rnd_target_params, state)

    rnd_values = jax.vmap(rnd)(states)
    target_rnd_values = jax.lax.stop_gradient(jax.vmap(target_rnd)(states))
    bonuses = jnp.sum((rnd_values - target_rnd_values)**2, axis=-1)
    rnd_loss = jnp.mean(bonuses)
    return rnd_loss, jax.lax.stop_gradient(bonuses)

  def loss_fn(params, target_ub_delta_vals, target_lb_delta_vals, bonuses,
              delta_samples):

    def online(state, deltas):
      return network_def.apply(params, state, deltas)

    predicted = jax.vmap(online)(states, deltas)
    # Compute DR3 loss.
    representations = jnp.squeeze(predicted.representation)
    next_states_predicted = jax.vmap(online)(next_states, deltas)
    next_state_representations = jnp.squeeze(
        next_states_predicted.representation)
    dr3_loss = compute_dr3_loss(representations, next_state_representations)

    # Compute CQL coefficients.
    delta_samples_bonuses = deltas[..., None] - delta_samples
    b = lcb_alpha * jnp.clip(
        bonuses[:, None, None] * jnp.log(1 / delta_samples_bonuses),
        a_min=1e-6,
        a_max=1e2)

    replay_action_ub_delta_vals = jax.vmap(
        lambda x, y: x[:, y][..., None])(predicted.ub_delta_values, actions)
    ub_bellman_errors = (target_ub_delta_vals - replay_action_ub_delta_vals)**2
    ub_penalties = b * (
        jax.scipy.special.logsumexp(
            predicted.ub_delta_values, axis=-1)[..., None] -
        replay_action_ub_delta_vals)
    ub_bellman_losses = ub_bellman_errors - ub_penalties
    ub_bellman_argmin = jnp.argmin(ub_bellman_losses, axis=-1)
    ub_bellman_loss = jnp.mean(
        jax.vmap(lambda x, y: x[jnp.arange(num_delta_samples), y])(
            ub_bellman_losses, ub_bellman_argmin))
    target_ub_delta_primes = jnp.mean(
        jax.vmap(lambda x, y: x[jnp.arange(num_delta_samples), y])(
            delta_samples, ub_bellman_argmin) / deltas)
    ub_cql_alpha = jnp.mean(
        jax.vmap(lambda x, y: x[jnp.arange(num_delta_samples), y])(
            b, ub_bellman_argmin))

    replay_action_lb_delta_vals = jax.vmap(
        lambda x, y: x[:, y][..., None])(predicted.lb_delta_values, actions)
    lb_bellman_errors = (target_lb_delta_vals - replay_action_lb_delta_vals)**2
    lb_penalties = b * (
        jax.scipy.special.logsumexp(
            predicted.lb_delta_values, axis=-1)[..., None] -
        replay_action_lb_delta_vals)
    lb_bellman_losses = lb_bellman_errors + lb_penalties
    lb_bellman_argmax = jnp.argmax(lb_bellman_losses, axis=-1)
    lb_bellman_loss = jnp.mean(
        jax.vmap(lambda x, y: x[jnp.arange(num_delta_samples), y])(
            lb_bellman_losses, lb_bellman_argmax))
    target_lb_delta_primes = jnp.mean(
        jax.vmap(lambda x, y: x[jnp.arange(num_delta_samples), y])(
            delta_samples, lb_bellman_argmax) / deltas)
    lb_cql_alpha = jnp.mean(
        jax.vmap(lambda x, y: x[jnp.arange(num_delta_samples), y])(
            b, lb_bellman_argmax))

    bellman_loss = ub_bellman_loss + lb_bellman_loss + dr3_alpha * dr3_loss
    return bellman_loss, (target_ub_delta_primes, target_lb_delta_primes,
                          ub_cql_alpha, lb_cql_alpha)

  def compute_metrics_fn(params):
    # Only for logging.
    delta = jnp.array([lcb_delta])

    def online(state):
      return network_def.apply(params, state, delta)

    predicted = jax.vmap(online)(states)

    ub_delta_values = predicted.ub_delta_values[:, 0]
    replay_action_ub_delta_values = jax.vmap(lambda x, y: x[y])(ub_delta_values,
                                                                actions)
    ub_delta_vals_data = jnp.mean(replay_action_ub_delta_values)
    ub_delta_vals_pi = jnp.mean(jnp.max(ub_delta_values, axis=-1))

    lb_delta_values = predicted.lb_delta_values[:, 0]
    replay_action_lb_delta_values = jax.vmap(lambda x, y: x[y])(lb_delta_values,
                                                                actions)
    lb_delta_vals_data = jnp.mean(replay_action_lb_delta_values)
    lb_delta_vals_pi = jnp.mean(jnp.max(lb_delta_values, axis=-1))

    return (ub_delta_vals_data, ub_delta_vals_pi, lb_delta_vals_data,
            lb_delta_vals_pi)

  # Train the RND network. Use errors as exploration bonuses.
  rnd_grad_fn = jax.value_and_grad(rnd_loss_fn, has_aux=True)
  (rnd_loss, bonuses), rnd_grad = rnd_grad_fn(rnd_online_params)
  rnd_updates, rnd_optimizer_state = rnd_optimizer.update(
      rnd_grad, rnd_optimizer_state, params=rnd_online_params)
  rnd_online_params = optax.apply_updates(rnd_online_params, rnd_updates)

  # Train the delta-values network
  rng, target_rng = jax.random.split(rng, num=2)
  batched_target_rng = jnp.stack(jax.random.split(target_rng, num=batch_size))
  (_, target_ub_delta_vals, target_lb_delta_vals,
   delta_samples) = target_delta_values(
       network_def, online_params, target_params, next_states, rewards,
       terminals, deltas, num_delta_prime_samples,
       cumulative_gamma, double_dqn, batched_target_rng)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, aux), grad = grad_fn(
      online_params, target_ub_delta_vals, target_lb_delta_vals,
      bonuses, delta_samples)
  ub_delta_prime, lb_delta_prime, ub_cql_alpha, lb_cql_alpha = aux
  updates, optimizer_state = optimizer.update(
      grad, optimizer_state, params=online_params)
  online_params = optax.apply_updates(online_params, updates)
  (ub_delta_vals_data, ub_delta_vals_pi, lb_delta_vals_data,
   lb_delta_vals_pi) = compute_metrics_fn(online_params)
  return (rng, optimizer_state, online_params,
          rnd_optimizer_state, rnd_online_params, loss, rnd_loss,
          ub_delta_vals_data, ub_delta_vals_pi,
          lb_delta_vals_data, lb_delta_vals_pi,
          ub_delta_prime, lb_delta_prime,
          ub_cql_alpha, lb_cql_alpha)



@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 9, 10, 12, 13))
def select_action(network_def, params, state, rng, num_actions, lcb_delta,
                  lcb_threshold, eval_mode, epsilon_eval, epsilon_train,
                  epsilon_decay_period, training_steps, min_replay_history,
                  epsilon_fn):
  """Select an action from the set of available actions.

  Chooses an action randomly with probability self._calculate_epsilon(), and
  otherwise acts greedily according to the current Q-value estimates.

  Args:
    network_def: Linen Module to use for inference.
    params: Linen params (frozen dict) to use for inference.
    state: input state to use for inference.
    rng: Jax random number generator.
    num_actions: int, number of actions (static_argnum).
    lcb_delta: float, delta value used for action selection (static_argnum).
    lcb_threshold: float, threshold used for action selection (static_argnum).
    eval_mode: bool, whether we are in eval mode (static_argnum).
    epsilon_eval: float, epsilon value to use in eval mode (static_argnum).
    epsilon_train: float, epsilon value to use in train mode (static_argnum).
    epsilon_decay_period: float, decay period for epsilon value for certain
      epsilon functions, such as linearly_decaying_epsilon, (static_argnum).
    training_steps: int, number of training steps so far.
    min_replay_history: int, minimum number of steps in replay buffer
      (static_argnum).
    epsilon_fn: function used to calculate epsilon value (static_argnum).

  Returns:
    rng: Jax random number generator.
    action: int, the selected action.
  """
  epsilon = jnp.where(
      eval_mode, epsilon_eval,
      epsilon_fn(epsilon_decay_period, training_steps, min_replay_history,
                 epsilon_train))

  rng, rng1, rng2 = jax.random.split(rng, num=3)
  p = jax.random.uniform(rng1)
  delta = jnp.array([lcb_delta])
  predicted = network_def.apply(params, state, delta)
  ub_delta_vals, lb_delta_vals = (predicted.ub_delta_values[0],
                                  predicted.lb_delta_values[0])

  # Choose actions with highest upper-bound that satisfy not having too low
  # of a lower-bound.
  lb_delta_vals_max = jnp.max(lb_delta_vals)
  action_values = (lb_delta_vals >
                   lcb_threshold * lb_delta_vals_max) * ub_delta_vals

  return rng, jnp.where(p <= epsilon,
                        jax.random.randint(rng2, (), 0, num_actions),
                        jnp.argmax(action_values))


@gin.configurable
class UncertaintyAgent(base_dqn_agent.JaxDQNAgent):
  """A JAX implementation of the uncertainty-aware agent."""

  def __init__(self,
               num_actions,
               replay_data_dir,
               delta_embedding_dim=64,
               double_dqn=False,
               num_delta_samples=32,
               num_delta_prime_samples=32,
               dr3_alpha=0.0,
               lcb_alpha=1.0,
               offline_lcb_delta=0.1,
               fine_tuning_lcb_delta=0.9,
               offline_lcb_threshold=0.9,
               offline_data_ratio=0.0,
               fine_tuning_lr=None,
               phase_decay_steps=100000,
               summary_writer=None,
               replay_buffer_builder=None):
    """Initializes the agent and constructs the necessary components.

    Args:
      num_actions: int, number of actions the agent can take at any state.
      replay_data_dir: str, log Directory from which to load the replay buffer.
      delta_embedding_dim: int, delta embedding dimension.
      double_dqn: bool, whether to use double Q-networks.
      num_delta_samples: int, number of delta samples
      num_delta_prime_samples: int, number of delta' samples in backups
      dr3_alpha: float, coefficient for DR3 loss.
      lcb_alpha: float, coefficient on confidence widths
      offline_lcb_delta: float, delta value used for offline action selection.
      fine_tuning_lcb_delta: float, delta value used for initial online action
        selection.
      offline_lcb_threshold: float, threshold on the lower-bounds for offline
        action selection.
      fine_tuning_lr: Learning rate for the fine-tuning phase.
      phase_decay_steps: int, steps for gradually decaying learning rate and
        cql coefficient when switching from pretraining to online phase.
      summary_writer: SummaryWriter object for outputting training statistics.
      replay_buffer_builder: Callable object that takes "self" as an argument
        and returns a replay buffer to use for training offline. If None, it
        will use the default FixedReplayBuffer.
    """
    logging.info('Creating %s agent with the following parameters:',
                 self.__class__.__name__)
    self.replay_data_dir = replay_data_dir
    if replay_buffer_builder is not None:
      self._build_replay_buffer = replay_buffer_builder
    self.pretraining = True
      
    self.rnd_network_def = networks.RNDNetwork(num_actions=num_actions)
    super().__init__(
        num_actions,
        network=functools.partial(
            networks.ImplicitDeltaNetwork,
            delta_embedding_dim=delta_embedding_dim),
        summary_writer=summary_writer)
    self.delta_embedding_dim = delta_embedding_dim
    self.double_dqn = double_dqn
    self.num_delta_samples = num_delta_samples
    self.num_delta_prime_samples = num_delta_prime_samples

    self._dr3_alpha = dr3_alpha
    self._lcb_alpha = lcb_alpha
    self._offline_lcb_delta = offline_lcb_delta
    self._fine_tuning_lcb_delta = fine_tuning_lcb_delta
    self._online_lcb_delta = fine_tuning_lcb_delta
    self._offline_lcb_threshold = offline_lcb_threshold
    self._online_training_steps = 0

    self._offline_replay = self._build_offline_replay_buffer()
    self._phase_decay_steps = phase_decay_steps
    self._fine_tuning_lr = fine_tuning_lr

  def _build_offline_replay_buffer(self):
    """Creates the fixed replay buffer used by the agent."""
    return fixed_replay.JaxFixedReplayBuffer(
        data_dir=self.replay_data_dir,
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype)

  def reload_data(self):
    self._offline_replay.reload_data()

  def set_phase(self, pretraining=False):
    self.pretraining = pretraining
    if not pretraining:
      if self._fine_tuning_lr:
        self.optimizer_state.hyperparams['learning_rate'] = self._fine_tuning_lr
        logging.info('Learning rate set to %.6f', self._fine_tuning_lr)
    else:
      self.optimizer_state.hyperparams['learning_rate'] = self._pretraining_lr
      logging.info('Learning rate set to %.6f', self._pretraining_lr)

  def _sample_from_offline_replay_buffer(self):
    samples = self._offline_replay.sample_transition_batch()
    types = self._offline_replay.get_transition_elements()
    self.offline_replay_elements = collections.OrderedDict()
    for element, element_type in zip(samples, types):
      self.offline_replay_elements[element_type.name] = element

  def _sample_from_replay_buffer(self):
    super()._sample_from_replay_buffer()
    self.online_replay_elements = self.replay_elements.copy()

  def _build_networks_and_optimizer(self):
    self._rng, rng1, rng2, rng3 = jax.random.split(self._rng, num=4)
    self.online_params = self.network_def.init(
        rng1, x=self.state, deltas=jnp.zeros(shape=[1]))
    self.optimizer = dqn_agent.create_fine_tuning_optimizer(
        self._optimizer_name, inject_hparams=True)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self._pretraining_lr = float(
        self.optimizer_state.hyperparams['learning_rate'])
    self.target_network_params = self.online_params
    # RND network and optimizer
    self.rnd_online_params = self.rnd_network_def.init(
        rng2, x=self.state)
    self.rnd_optimizer = dqn_agent.create_fine_tuning_optimizer(
        self._optimizer_name, learning_rate=1e-5, inject_hparams=True)
    self.rnd_optimizer_state = self.rnd_optimizer.init(self.rnd_online_params)
    self.rnd_target_network_params = self.rnd_network_def.init(
        rng3, x=self.state)

  def begin_episode(self, observation):
    """Returns the agent's first action for this episode.

    Args:
      observation: numpy array, the environment's initial observation.

    Returns:
      int, the selected action.
    """
    self._reset_state()
    self._record_observation(observation)

    if not (self.pretraining or self.eval_mode):
      self._train_step()

    lcb_delta = (
        self._offline_lcb_delta if self.pretraining else self._online_lcb_delta)
    lcb_threshold = self._offline_lcb_threshold if self.pretraining else 0.
    self._rng, self.action = select_action(
        self.network_def, self.online_params, self.preprocess_fn(self.state),
        self._rng, self.num_actions, lcb_delta, lcb_threshold,
        self.eval_mode, self.epsilon_eval, self.epsilon_train,
        self.epsilon_decay_period, self.training_steps, self.min_replay_history,
        self.epsilon_fn)
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

    if not (self.pretraining or self.eval_mode):
      self._store_transition(self._last_observation, self.action, reward, False)
      self._train_step()

    lcb_delta = (
        self._offline_lcb_delta if self.pretraining else self._online_lcb_delta)
    lcb_threshold = self._offline_lcb_threshold if self.pretraining else 0.
    self._rng, self.action = select_action(
        self.network_def, self.online_params, self.preprocess_fn(self.state),
        self._rng, self.num_actions, lcb_delta, lcb_threshold,
        self.eval_mode, self.epsilon_eval, self.epsilon_train,
        self.epsilon_decay_period, self.training_steps, self.min_replay_history,
        self.epsilon_fn)
    self.action = onp.asarray(self.action)
    return self.action

  def _update_hparams(self):
    """Update training hyper-parameters."""
    self._online_lcb_delta = onp.array(
        linearly_decaying_value(
            self._phase_decay_steps,
            self._online_training_steps,
            final_value=0.5,
            initial_value=self._fine_tuning_lcb_delta)).item()

    self.optimizer_state.hyperparams['learning_rate'] = linearly_decaying_value(
        self._phase_decay_steps,
        self._online_training_steps,
        final_value=self._fine_tuning_lr,
        initial_value=self._pretraining_lr)

  def _train_step(self):
    """Runs a single training step."""
    if self.pretraining:
      self._sample_from_offline_replay_buffer()
      self._opt_step(
          replay_elements=self.offline_replay_elements,
          loss_prefix='Offline',
          target_update_period=self.target_update_period)
    else:
      if self._replay.add_count > self.min_replay_history:
        if self.training_steps % self.update_period == 0:
          self._sample_from_replay_buffer()
          self._opt_step(
              replay_elements=self.online_replay_elements,
              loss_prefix='Online',
              target_update_period=self.target_update_period)

          self._online_training_steps += 1
          self._update_hparams()
    self.training_steps += 1

  def _opt_step(self, replay_elements, loss_prefix, target_update_period):
    lcb_delta = (
        self._offline_lcb_delta
        if self.pretraining else self._online_lcb_delta)
    (self._rng, self.optimizer_state, self.online_params,
     self.rnd_optimizer_state, self.rnd_online_params, loss, rnd_loss,
     ub_delta_data, ub_delta_pi, lb_delta_data, lb_delta_pi,
     mean_ub_delta_prime, mean_lb_delta_prime,
     ub_cql_alpha, lb_cql_alpha) = train(
         self.network_def, self.online_params, self.target_network_params,
         self.optimizer, self.optimizer_state, self.rnd_network_def,
         self.rnd_online_params, self.rnd_target_network_params,
         self.rnd_optimizer, self.rnd_optimizer_state,
         self.preprocess_fn(replay_elements['state']),
         replay_elements['action'],
         self.preprocess_fn(replay_elements['next_state']),
         replay_elements['reward'], replay_elements['terminal'],
         self.num_delta_samples, self.num_delta_prime_samples, self._dr3_alpha,
         self._lcb_alpha, lcb_delta, self.cumulative_gamma, self.double_dqn,
         self._rng)

    if (self.summary_writer is not None and self.training_steps > 0 and
        self.training_steps > 0 and
        self.training_steps % self.summary_writing_frequency == 0):
      with self.summary_writer.as_default():
        tf.summary.scalar(
            f'{loss_prefix}/Bellman', loss, step=self.training_steps)
        tf.summary.scalar(
            f'{loss_prefix}/RND', rnd_loss, step=self.training_steps)
        tf.summary.scalar(
            f'{loss_prefix}/lcb_delta', lcb_delta, step=self.training_steps)
        tf.summary.scalar(
            f'{loss_prefix}/ub_delta_vals_data',
            ub_delta_data,
            step=self.training_steps)
        tf.summary.scalar(
            f'{loss_prefix}/ub_delta_vals_pi',
            ub_delta_pi,
            step=self.training_steps)
        tf.summary.scalar(
            f'{loss_prefix}/lb_delta_vals_data',
            lb_delta_data,
            step=self.training_steps)
        tf.summary.scalar(
            f'{loss_prefix}/lb_delta_vals_pi',
            lb_delta_pi,
            step=self.training_steps)
        tf.summary.scalar(
            f'{loss_prefix}/backup_ub_delta_prime',
            mean_ub_delta_prime,
            step=self.training_steps)
        tf.summary.scalar(
            f'{loss_prefix}/backup_lb_delta_prime',
            mean_lb_delta_prime,
            step=self.training_steps)
        tf.summary.scalar(
            f'{loss_prefix}/ub_cql_alpha', ub_cql_alpha,
            step=self.training_steps)
        tf.summary.scalar(
            f'{loss_prefix}/lb_cql_alpha', lb_cql_alpha,
            step=self.training_steps)
      self.summary_writer.flush()

    if self.training_steps % target_update_period == 0:
      self._sync_weights()
