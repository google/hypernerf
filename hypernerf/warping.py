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

"""Warp fields."""
from typing import Any, Iterable, Optional, Dict

from flax import linen as nn
import gin
import jax
import jax.numpy as jnp

from hypernerf import model_utils
from hypernerf import utils
from hypernerf import modules
from hypernerf import rigid_body as rigid
from hypernerf import types


@gin.configurable(denylist=['name'])
class TranslationField(nn.Module):
  """Network that predicts warps as a translation field.

  References:
    https://en.wikipedia.org/wiki/Vector_potential
    https://en.wikipedia.org/wiki/Helmholtz_decomposition

  Attributes:
    metadata_encoder: an encoder for metadata.
    alpha: the alpha for the positional encoding.
    skips: the index of the layers with skip connections.
    depth: the depth of the network excluding the output layer.
    hidden_channels: the width of the network hidden layers.
    activation: the activation for each layer.
    metadata_encoded: whether the metadata parameter is pre-encoded or not.
    hidden_initializer: the initializer for the hidden layers.
    output_initializer: the initializer for the last output layer.
  """
  min_deg: int = 0
  max_deg: int = 8
  use_posenc_identity: bool = True

  skips: Iterable[int] = (4,)
  depth: int = 6
  hidden_channels: int = 128
  activation: types.Activation = nn.relu
  norm: Optional[Any] = None
  hidden_init: types.Initializer = jax.nn.initializers.glorot_uniform()
  output_init: types.Initializer = jax.nn.initializers.uniform(scale=1e-4)

  def setup(self):
    # Note that this must be done this way instead of using mutable list
    # operations.
    # See https://github.com/google/flax/issues/524.
    # pylint: disable=g-complex-comprehension
    output_dims = 3
    self.mlp = modules.MLP(
        width=self.hidden_channels,
        depth=self.depth,
        skips=self.skips,
        hidden_activation=self.activation,
        hidden_norm=self.norm,
        hidden_init=self.hidden_init,
        output_init=self.output_init,
        output_channels=output_dims)

  def warp(self,
           points: jnp.ndarray,
           metadata: jnp.ndarray,
           extra_params: Dict[str, Any]):
    points_embed = model_utils.posenc(points,
                                      min_deg=self.min_deg,
                                      max_deg=self.max_deg,
                                      use_identity=self.use_posenc_identity,
                                      alpha=extra_params['warp_alpha'])
    inputs = jnp.concatenate([points_embed, metadata], axis=-1)
    translation = self.mlp(inputs)
    warped_points = points + translation

    return warped_points

  def __call__(self,
               points: jnp.ndarray,
               metadata: jnp.ndarray,
               extra_params: Dict[str, Any],
               return_jacobian: bool = False):
    """Warp the given points using a warp field.

    Args:
      points: the points to warp.
      metadata: encoded metadata features.
      extra_params: extra parameters used in the warp field e.g., the warp
        alpha.
      return_jacobian: if True compute and return the Jacobian of the warp.

    Returns:
      The warped points and the Jacobian of the warp if `return_jacobian` is
        True.
    """
    out = {
        'warped_points': self.warp(points, metadata, extra_params)
    }

    if return_jacobian:
      jac_fn = jax.jacfwd(lambda *x: self.warp(*x)[..., :3], argnums=0)
      out['jacobian'] = jac_fn(points, metadata, extra_params)

    return out


@gin.configurable(denylist=['name'])
class SE3Field(nn.Module):
  """Network that predicts warps as an SE(3) field.

  Attributes:
    points_encoder: the positional encoder for the points.
    metadata_encoder: an encoder for metadata.
    alpha: the alpha for the positional encoding.
    skips: the index of the layers with skip connections.
    depth: the depth of the network excluding the logit layer.
    hidden_channels: the width of the network hidden layers.
    activation: the activation for each layer.
    metadata_encoded: whether the metadata parameter is pre-encoded or not.
    hidden_initializer: the initializer for the hidden layers.
    output_initializer: the initializer for the last logit layer.
  """
  min_deg: int = 0
  max_deg: int = 8
  use_posenc_identity: bool = False

  activation: types.Activation = nn.relu
  norm: Optional[Any] = None
  skips: Iterable[int] = (4,)
  trunk_depth: int = 6
  trunk_width: int = 128
  rotation_depth: int = 0
  rotation_width: int = 128
  pivot_depth: int = 0
  pivot_width: int = 128
  translation_depth: int = 0
  translation_width: int = 128

  default_init: types.Initializer = jax.nn.initializers.xavier_uniform()
  rotation_init: types.Initializer = jax.nn.initializers.uniform(scale=1e-4)
  translation_init: types.Initializer = jax.nn.initializers.uniform(scale=1e-4)

  # Unused, here for backwards compatibility.
  num_hyper_dims: int = 0
  hyper_depth: int = 0
  hyper_width: int = 0
  hyper_init: Optional[types.Initializer] = None

  def setup(self):
    self.trunk = modules.MLP(
        depth=self.trunk_depth,
        width=self.trunk_width,
        hidden_activation=self.activation,
        hidden_norm=self.norm,
        hidden_init=self.default_init,
        skips=self.skips)

    branches = {
        'w':
            modules.MLP(
                depth=self.rotation_depth,
                width=self.rotation_width,
                hidden_activation=self.activation,
                hidden_norm=self.norm,
                hidden_init=self.default_init,
                output_init=self.rotation_init,
                output_channels=3),
        'v':
            modules.MLP(
                depth=self.translation_depth,
                width=self.translation_width,
                hidden_activation=self.activation,
                hidden_norm=self.norm,
                hidden_init=self.default_init,
                output_init=self.translation_init,
                output_channels=3),
    }

    # Note that this must be done this way instead of using mutable operations.
    # See https://github.com/google/flax/issues/524.
    self.branches = branches

  def warp(self,
           points: jnp.ndarray,
           metadata_embed: jnp.ndarray,
           extra_params: Dict[str, Any]):
    points_embed = model_utils.posenc(points,
                                      min_deg=self.min_deg,
                                      max_deg=self.max_deg,
                                      use_identity=self.use_posenc_identity,
                                      alpha=extra_params['warp_alpha'])
    inputs = jnp.concatenate([points_embed, metadata_embed], axis=-1)
    trunk_output = self.trunk(inputs)

    w = self.branches['w'](trunk_output)
    v = self.branches['v'](trunk_output)
    theta = jnp.linalg.norm(w, axis=-1)
    w = w / theta[..., jnp.newaxis]
    v = v / theta[..., jnp.newaxis]
    screw_axis = jnp.concatenate([w, v], axis=-1)
    transform = rigid.exp_se3(screw_axis, theta)

    warped_points = points
    warped_points = rigid.from_homogenous(
        utils.matmul(transform, rigid.to_homogenous(warped_points)))

    return warped_points

  def __call__(self,
               points: jnp.ndarray,
               metadata: jnp.ndarray,
               extra_params: Dict[str, Any],
               return_jacobian: bool = False):
    """Warp the given points using a warp field.

    Args:
      points: the points to warp.
      metadata: metadata indices if metadata_encoded is False else pre-encoded
        metadata.
      extra_params: A dictionary containing
        'alpha': the alpha value for the positional encoding.
      return_jacobian: if True compute and return the Jacobian of the warp.

    Returns:
      The warped points and the Jacobian of the warp if `return_jacobian` is
        True.
    """

    out = {
        'warped_points': self.warp(points, metadata, extra_params)
    }

    if return_jacobian:
      jac_fn = jax.jacfwd(self.warp, argnums=0)
      out['jacobian'] = jac_fn(points, metadata, extra_params)

    return out
