# Copyright 2021 Google LLC
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

"""Modules for NeRF models."""
import functools
from typing import Any, Optional, Tuple

from flax import linen as nn
import gin
import jax
import jax.numpy as jnp

from hypernerf import model_utils
from hypernerf import types


def get_norm_layer(norm_type):
  """Translates a norm type to a norm constructor."""
  if norm_type is None or norm_type == 'none':
    return None
  elif norm_type == 'layer':
    return functools.partial(nn.LayerNorm, use_scale=False, use_bias=False)
  elif norm_type == 'group':
    return functools.partial(nn.GroupNorm, use_scale=False, use_bias=False)
  elif norm_type == 'batch':
    return functools.partial(nn.BatchNorm, use_scale=False, use_bias=False)
  else:
    raise ValueError(f'Unknown norm type {norm_type}')


class MLP(nn.Module):
  """Basic MLP class with hidden layers and an output layers."""
  depth: int
  width: int
  hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()
  hidden_activation: types.Activation = nn.relu
  hidden_norm: Optional[types.Normalizer] = None
  output_init: Optional[types.Initializer] = None
  output_channels: int = 0
  output_activation: Optional[types.Activation] = lambda x: x
  use_bias: bool = True
  skips: Tuple[int] = tuple()

  @nn.compact
  def __call__(self, x):
    inputs = x
    for i in range(self.depth):
      layer = nn.Dense(
          self.width,
          use_bias=self.use_bias,
          kernel_init=self.hidden_init,
          name=f'hidden_{i}')
      if i in self.skips:
        x = jnp.concatenate([x, inputs], axis=-1)
      x = layer(x)
      if self.hidden_norm is not None:
        x = self.hidden_norm()(x)  # pylint: disable=not-callable
      x = self.hidden_activation(x)

    if self.output_channels > 0:
      logit_layer = nn.Dense(
          self.output_channels,
          use_bias=self.use_bias,
          kernel_init=self.output_init,
          name='logit')
      x = logit_layer(x)
      if self.output_activation is not None:
        x = self.output_activation(x)

    return x


class NerfMLP(nn.Module):
  """A simple MLP.

  Attributes:
    nerf_trunk_depth: int, the depth of the first part of MLP.
    nerf_trunk_width: int, the width of the first part of MLP.
    nerf_rgb_branch_depth: int, the depth of the second part of MLP.
    nerf_rgb_branch_width: int, the width of the second part of MLP.
    activation: function, the activation function used in the MLP.
    skips: which layers to add skip layers to.
    alpha_channels: int, the number of alpha_channelss.
    rgb_channels: int, the number of rgb_channelss.
    condition_density: if True put the condition at the begining which
      conditions the density of the field.
  """
  trunk_depth: int = 8
  trunk_width: int = 256

  rgb_branch_depth: int = 1
  rgb_branch_width: int = 128
  rgb_channels: int = 3

  alpha_branch_depth: int = 0
  alpha_branch_width: int = 128
  alpha_channels: int = 1

  activation: types.Activation = nn.relu
  norm: Optional[Any] = None
  skips: Tuple[int] = (4,)

  @nn.compact
  def __call__(self, x, alpha_condition, rgb_condition):
    """Multi-layer perception for nerf.

    Args:
      x: sample points with shape [batch, num_coarse_samples, feature].
      alpha_condition: a condition array provided to the alpha branch.
      rgb_condition: a condition array provided in the RGB branch.

    Returns:
      raw: [batch, num_coarse_samples, rgb_channels+alpha_channels].
    """
    dense = functools.partial(
        nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())

    feature_dim = x.shape[-1]
    num_samples = x.shape[1]
    x = x.reshape([-1, feature_dim])

    def broadcast_condition(c):
      # Broadcast condition from [batch, feature] to
      # [batch, num_coarse_samples, feature] since all the samples along the
      # same ray has the same viewdir.
      c = jnp.tile(c[:, None, :], (1, num_samples, 1))
      # Collapse the [batch, num_coarse_samples, feature] tensor to
      # [batch * num_coarse_samples, feature] to be fed into nn.Dense.
      c = c.reshape([-1, c.shape[-1]])
      return c

    trunk_mlp = MLP(depth=self.trunk_depth,
                    width=self.trunk_width,
                    hidden_activation=self.activation,
                    hidden_norm=self.norm,
                    hidden_init=jax.nn.initializers.glorot_uniform(),
                    skips=self.skips)
    rgb_mlp = MLP(depth=self.rgb_branch_depth,
                  width=self.rgb_branch_width,
                  hidden_activation=self.activation,
                  hidden_norm=self.norm,
                  hidden_init=jax.nn.initializers.glorot_uniform(),
                  output_init=jax.nn.initializers.glorot_uniform(),
                  output_channels=self.rgb_channels)
    alpha_mlp = MLP(depth=self.alpha_branch_depth,
                    width=self.alpha_branch_width,
                    hidden_activation=self.activation,
                    hidden_norm=self.norm,
                    hidden_init=jax.nn.initializers.glorot_uniform(),
                    output_init=jax.nn.initializers.glorot_uniform(),
                    output_channels=self.alpha_channels)

    if self.trunk_depth > 0:
      x = trunk_mlp(x)

    if (alpha_condition is not None) or (rgb_condition is not None):
      bottleneck = dense(self.trunk_width, name='bottleneck')(x)

    if alpha_condition is not None:
      alpha_condition = broadcast_condition(alpha_condition)
      alpha_input = jnp.concatenate([bottleneck, alpha_condition], axis=-1)
    else:
      alpha_input = x
    alpha = alpha_mlp(alpha_input)

    if rgb_condition is not None:
      rgb_condition = broadcast_condition(rgb_condition)
      rgb_input = jnp.concatenate([bottleneck, rgb_condition], axis=-1)
    else:
      rgb_input = x
    rgb = rgb_mlp(rgb_input)

    return {
        'rgb': rgb.reshape((-1, num_samples, self.rgb_channels)),
        'alpha': alpha.reshape((-1, num_samples, self.alpha_channels)),
    }


@gin.configurable(denylist=['name'])
class GLOEmbed(nn.Module):
  """A GLO encoder module, which is just a thin wrapper around nn.Embed.

  Attributes:
    num_embeddings: The number of embeddings.
    features: The dimensions of each embedding.
    embedding_init: The initializer to use for each.
  """

  num_embeddings: int = gin.REQUIRED
  num_dims: int = gin.REQUIRED
  embedding_init: types.Activation = nn.initializers.uniform(scale=0.05)

  def setup(self):
    self.embed = nn.Embed(
        num_embeddings=self.num_embeddings,
        features=self.num_dims,
        embedding_init=self.embedding_init)

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Method to get embeddings for specified indices.

    Args:
      inputs: The indices to fetch embeddings for.

    Returns:
      The embeddings corresponding to the indices provided.
    """
    if inputs.shape[-1] == 1:
      inputs = jnp.squeeze(inputs, axis=-1)

    return self.embed(inputs)


@gin.configurable(denylist=['name'])
class HyperSheetMLP(nn.Module):
  """An MLP that defines a bendy slicing surface through hyper space."""
  output_channels: int = gin.REQUIRED
  min_deg: int = 0
  max_deg: int = 1

  depth: int = 6
  width: int = 64
  skips: Tuple[int] = (4,)
  hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()
  output_init: types.Initializer = jax.nn.initializers.normal(1e-5)
  # output_init: types.Initializer = jax.nn.initializers.glorot_uniform()

  use_residual: bool = False

  @nn.compact
  def __call__(self, points, embed, alpha=None):
    points_feat = model_utils.posenc(
        points, self.min_deg, self.max_deg, alpha=alpha)
    inputs = jnp.concatenate([points_feat, embed], axis=-1)
    mlp = MLP(depth=self.depth,
              width=self.width,
              skips=self.skips,
              hidden_init=self.hidden_init,
              output_channels=self.output_channels,
              output_init=self.output_init)
    if self.use_residual:
      return mlp(inputs) + embed
    else:
      return mlp(inputs)
