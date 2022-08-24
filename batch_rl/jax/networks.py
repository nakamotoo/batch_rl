import time
from typing import Tuple, Union

from dopamine.discrete_domains import atari_lib
from flax import linen as nn
import gin
import jax
import jax.numpy as jnp
import numpy as onp


### Implicit Delta Network ###
@gin.configurable
class ImplicitDeltaNetwork(nn.Module):
  """The Implicit Delta Network (similar to IQN networks)."""
  num_actions: int
  delta_embedding_dim: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x, deltas):
    initializer = nn.initializers.variance_scaling(
        scale=1.0 / jnp.sqrt(3.0),
        mode='fan_in',
        distribution='uniform')
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1),
                kernel_init=initializer)(x)
    x = nn.relu(x)
    x = x.reshape((-1))  # flatten
    representation = nn.Dense(features=512, kernel_init=initializer)(x)

    num_deltas = deltas.shape[0]
    state_net_tiled = jnp.tile(representation, [num_deltas, 1])
    delta_net = jnp.tile(deltas[..., None], [1, self.delta_embedding_dim])
    delta_net = (
        jnp.arange(1, self.delta_embedding_dim + 1, 1).astype(jnp.float32)
        * onp.pi
        * delta_net)
    delta_net = jnp.cos(delta_net)
    delta_net = nn.Dense(features=512, kernel_init=initializer)(delta_net)
    delta_net = nn.relu(delta_net)
    delta_representation = state_net_tiled * delta_net

    # Get upper-bound delta values.
    x = nn.Dense(features=512, kernel_init=initializer)(delta_representation)
    x = nn.relu(x)
    ub_delta_values = nn.Dense(features=self.num_actions,
                               kernel_init=initializer)(x)
    # Get lower-bound delta values.
    lb_delta_values = nn.Dense(features=self.num_actions,
                               kernel_init=initializer)(x)
    return ImplicitDeltaNetworkType(
        ub_delta_values, lb_delta_values, representation)


@gin.configurable
class RNDNetwork(nn.Module):
  """Random distillation network."""
  num_actions: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x):
    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    x = nn.Conv(
        features=32,
        kernel_size=(8, 8),
        strides=(4, 4),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(4, 4),
        strides=(2, 2),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = x.reshape(-1)  # flatten

    ortho_initializer = nn.initializers.orthogonal(scale=jnp.sqrt(2))
    x = nn.Dense(features=256, kernel_init=ortho_initializer)(x)
    return x


@gin.configurable
class RNDNetworkWithAction(nn.Module):
  """Random distillation network."""
  num_actions: int
  inputs_preprocessed: bool = False

  @nn.compact
  def __call__(self, x, a):
    initializer = nn.initializers.xavier_uniform()
    if not self.inputs_preprocessed:
      x = preprocess_atari_inputs(x)
    x = nn.Conv(
        features=32,
        kernel_size=(8, 8),
        strides=(4, 4),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = nn.Conv(
        features=64,
        kernel_size=(4, 4),
        strides=(2, 2),
        kernel_init=initializer)(
            x)
    x = nn.relu(x)
    x = x.reshape(-1)  # flatten

    ortho_initializer = nn.initializers.orthogonal(scale=onp.sqrt(2))
    a = nn.Embed(
        num_embeddings=self.num_actions,
        features=64)(a)
    x = nn.Dense(features=64, kernel_init=ortho_initializer)(x)
    x = nn.relu(x)
    x_a = x * a
    x_a = nn.Dense(features=256, kernel_init=ortho_initializer)(x_a)
    return x_a
