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

"""Different model implementation plus a general port for all the models."""
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Sequence, Mapping

from flax import linen as nn
import gin
import immutabledict
import jax
from jax import random
import jax.numpy as jnp

from hypernerf import model_utils
from hypernerf import modules
from hypernerf import types
# pylint: disable=unused-import
from hypernerf import warping


def filter_sigma(points, sigma, render_opts):
  """Filters the density based on various rendering arguments.

   - `dust_threshold` suppresses any sigma values below a threshold.
   - `bounding_box` suppresses any sigma values outside of a 3D bounding box.

  Args:
    points: the input points for each sample.
    sigma: the array of sigma values.
    render_opts: a dictionary containing any of the options listed above.

  Returns:
    A filtered sigma density field.
  """
  if render_opts is None:
    return sigma

  # Clamp densities below the set threshold.
  if 'dust_threshold' in render_opts:
    dust_thres = render_opts.get('dust_threshold', 0.0)
    sigma = (sigma >= dust_thres).astype(jnp.float32) * sigma
  if 'bounding_box' in render_opts:
    xmin, xmax, ymin, ymax, zmin, zmax = render_opts['bounding_box']
    render_mask = ((points[..., 0] >= xmin) & (points[..., 0] <= xmax)
                   & (points[..., 1] >= ymin) & (points[..., 1] <= ymax)
                   & (points[..., 2] >= zmin) & (points[..., 2] <= zmax))
    sigma = render_mask.astype(jnp.float32) * sigma

  return sigma


@gin.configurable(denylist=['name'])
class NerfModel(nn.Module):
  """Nerf NN Model with both coarse and fine MLPs.

  Attributes:
    embeddings_dict: a dictionary containing the embeddings of each metadata
      key.
    use_viewdirs: bool, use viewdirs as a condition.
    noise_std: float, std dev of noise added to regularize sigma output.
    nerf_trunk_depth: int, the depth of the first part of MLP.
    nerf_trunk_width: int, the width of the first part of MLP.
    nerf_rgb_branch_depth: int, the depth of the second part of MLP.
    nerf_rgb_branch_width: int, the width of the second part of MLP.
    nerf_skips: which layers to add skip layers in the NeRF model.
    spatial_point_min_deg: min degree of positional encoding for positions.
    spatial_point_max_deg: max degree of positional encoding for positions.
    hyper_point_min_deg: min degree of positional encoding for hyper points.
    hyper_point_max_deg: max degree of positional encoding for hyper points.
    viewdir_min_deg: min degree of positional encoding for viewdirs.
    viewdir_max_deg: max degree of positional encoding for viewdirs.

    alpha_channels: int, the number of alpha_channelss.
    rgb_channels: int, the number of rgb_channelss.
    activation: the activation function used in the MLP.
    sigma_activation: the activation function applied to the sigma density.

    near: float, near clip.
    far: float, far clip.
    num_coarse_samples: int, the number of samples for coarse nerf.
    num_fine_samples: int, the number of samples for fine nerf.
    use_stratified_sampling: use stratified sampling.
    use_white_background: composite rendering on to a white background.
    use_linear_disparity: sample linearly in disparity rather than depth.

    use_nerf_embed: whether to use the template metadata.
    use_alpha_condition: whether to feed the appearance metadata to the alpha
      branch.
    use_rgb_condition: whether to feed the appearance metadata to the rgb
      branch.

    use_warp: whether to use the warp field or not.
    warp_metadata_config: the config for the warp metadata encoder.
    warp_min_deg: min degree of positional encoding for warps.
    warp_max_deg: max degree of positional encoding for warps.
  """
  embeddings_dict: Mapping[str, Sequence[int]] = gin.REQUIRED
  near: float = gin.REQUIRED
  far: float = gin.REQUIRED

  # NeRF architecture.
  use_viewdirs: bool = True
  noise_std: Optional[float] = None
  nerf_trunk_depth: int = 8
  nerf_trunk_width: int = 256
  nerf_rgb_branch_depth: int = 1
  nerf_rgb_branch_width: int = 128
  nerf_skips: Tuple[int] = (4,)

  # NeRF rendering.
  num_coarse_samples: int = 196
  num_fine_samples: int = 196
  use_stratified_sampling: bool = True
  use_white_background: bool = False
  use_linear_disparity: bool = False
  use_sample_at_infinity: bool = True

  spatial_point_min_deg: int = 0
  spatial_point_max_deg: int = 10
  hyper_point_min_deg: int = 0
  hyper_point_max_deg: int = 4
  viewdir_min_deg: int = 0
  viewdir_max_deg: int = 4
  use_posenc_identity: bool = True

  alpha_channels: int = 1
  rgb_channels: int = 3
  activation: types.Activation = nn.relu
  norm_type: Optional[str] = None
  sigma_activation: types.Activation = nn.softplus

  # NeRF metadata configs.
  use_nerf_embed: bool = False
  nerf_embed_cls: Callable[..., nn.Module] = (
      functools.partial(modules.GLOEmbed, num_dims=8))
  nerf_embed_key: str = 'appearance'
  use_alpha_condition: bool = False
  use_rgb_condition: bool = False
  hyper_slice_method: str = 'none'
  hyper_embed_cls: Callable[..., nn.Module] = (
      functools.partial(modules.GLOEmbed, num_dims=8))
  hyper_embed_key: str = 'appearance'
  hyper_use_warp_embed: bool = True
  hyper_sheet_mlp_cls: Callable[..., nn.Module] = modules.HyperSheetMLP
  hyper_sheet_use_input_points: bool = True

  # Warp configs.
  use_warp: bool = False
  warp_field_cls: Callable[..., nn.Module] = warping.SE3Field
  warp_embed_cls: Callable[..., nn.Module] = (
      functools.partial(modules.GLOEmbed, num_dims=8))
  warp_embed_key: str = 'warp'

  @property
  def num_nerf_embeds(self):
    return max(self.embeddings_dict[self.nerf_embed_key]) + 1

  @property
  def num_warp_embeds(self):
    return max(self.embeddings_dict[self.warp_embed_key]) + 1

  @property
  def num_hyper_embeds(self):
    return max(self.embeddings_dict[self.hyper_embed_key]) + 1

  @property
  def nerf_embeds(self):
    return jnp.array(self.embeddings_dict[self.nerf_embed_key], jnp.uint32)

  @property
  def warp_embeds(self):
    return jnp.array(self.embeddings_dict[self.warp_embed_key], jnp.uint32)

  @property
  def hyper_embeds(self):
    return jnp.array(self.embeddings_dict[self.hyper_embed_key], jnp.uint32)

  @property
  def has_hyper(self):
    """Whether the model uses a separate hyper embedding."""
    return self.hyper_slice_method != 'none'

  @property
  def has_hyper_embed(self):
    """Whether the model uses a separate hyper embedding."""
    # If the warp field outputs the hyper coordinates then there is no separate
    # hyper embedding.
    return self.has_hyper

  @property
  def has_embeds(self):
    return self.has_hyper_embed or self.use_warp or self.use_nerf_embed

  @staticmethod
  def _encode_embed(embed, embed_fn):
    """Encodes embeddings.

    If the channel size 1, it is just a single metadata ID.
    If the channel size is 3:
      the first channel is the left metadata ID,
      the second channel is the right metadata ID,
      the last channel is the progression from left to right (between 0 and 1).

    Args:
      embed: a (*, 1) or (*, 3) array containing metadata.
      embed_fn: the embedding function.

    Returns:
      A (*, C) array containing encoded embeddings.
    """
    if embed.shape[-1] == 3:
      left, right, progression = jnp.split(embed, 3, axis=-1)
      left = embed_fn(left.astype(jnp.uint32))
      right = embed_fn(right.astype(jnp.uint32))
      return (1.0 - progression) * left + progression * right
    else:
      return embed_fn(embed)

  def encode_hyper_embed(self, metadata):
    if self.hyper_slice_method == 'axis_aligned_plane':
      # return self._encode_embed(metadata[self.hyper_embed_key],
      #                           self.hyper_embed)
      if self.hyper_use_warp_embed:
        return self._encode_embed(metadata[self.warp_embed_key],
                                  self.warp_embed)
      else:
        return self._encode_embed(metadata[self.hyper_embed_key],
                                  self.hyper_embed)
    elif self.hyper_slice_method == 'bendy_sheet':
      # The bendy sheet shares the metadata of the warp.
      if self.hyper_use_warp_embed:
        return self._encode_embed(metadata[self.warp_embed_key],
                                  self.warp_embed)
      else:
        return self._encode_embed(metadata[self.hyper_embed_key],
                                  self.hyper_embed)
    else:
      raise RuntimeError(
          f'Unknown hyper slice method {self.hyper_slice_method}.')

  def encode_nerf_embed(self, metadata):
    return self._encode_embed(metadata[self.nerf_embed_key], self.nerf_embed)

  def encode_warp_embed(self, metadata):
    return self._encode_embed(metadata[self.warp_embed_key], self.warp_embed)

  def setup(self):
    if (self.use_nerf_embed
        and not (self.use_rgb_condition
                 or self.use_alpha_condition)):
      raise ValueError('Template metadata is enabled but none of the condition'
                       'branches are.')

    if self.use_nerf_embed:
      self.nerf_embed = self.nerf_embed_cls(num_embeddings=self.num_nerf_embeds)
    if self.use_warp:
      self.warp_embed = self.warp_embed_cls(num_embeddings=self.num_warp_embeds)

    if self.hyper_slice_method == 'axis_aligned_plane':
      self.hyper_embed = self.hyper_embed_cls(
          num_embeddings=self.num_hyper_embeds)
    elif self.hyper_slice_method == 'bendy_sheet':
      if not self.hyper_use_warp_embed:
        self.hyper_embed = self.hyper_embed_cls(
            num_embeddings=self.num_hyper_embeds)
      self.hyper_sheet_mlp = self.hyper_sheet_mlp_cls()

    if self.use_warp:
      self.warp_field = self.warp_field_cls()

    norm_layer = modules.get_norm_layer(self.norm_type)
    nerf_mlps = {
        'coarse': modules.NerfMLP(
            trunk_depth=self.nerf_trunk_depth,
            trunk_width=self.nerf_trunk_width,
            rgb_branch_depth=self.nerf_rgb_branch_depth,
            rgb_branch_width=self.nerf_rgb_branch_width,
            activation=self.activation,
            norm=norm_layer,
            skips=self.nerf_skips,
            alpha_channels=self.alpha_channels,
            rgb_channels=self.rgb_channels)
    }
    if self.num_fine_samples > 0:
      nerf_mlps['fine'] = modules.NerfMLP(
          trunk_depth=self.nerf_trunk_depth,
          trunk_width=self.nerf_trunk_width,
          rgb_branch_depth=self.nerf_rgb_branch_depth,
          rgb_branch_width=self.nerf_rgb_branch_width,
          activation=self.activation,
          norm=norm_layer,
          skips=self.nerf_skips,
          alpha_channels=self.alpha_channels,
          rgb_channels=self.rgb_channels)
    self.nerf_mlps = nerf_mlps

  def get_condition_inputs(self, viewdirs, metadata, metadata_encoded=False):
    """Create the condition inputs for the NeRF template."""
    alpha_conditions = []
    rgb_conditions = []

    # Point attribute predictions
    if self.use_viewdirs:
      viewdirs_feat = model_utils.posenc(
          viewdirs,
          min_deg=self.viewdir_min_deg,
          max_deg=self.viewdir_max_deg,
          use_identity=self.use_posenc_identity)
      rgb_conditions.append(viewdirs_feat)

    if self.use_nerf_embed:
      if metadata_encoded:
        nerf_embed = metadata['encoded_nerf']
      else:
        nerf_embed = metadata[self.nerf_embed_key]
        nerf_embed = self.nerf_embed(nerf_embed)
      if self.use_alpha_condition:
        alpha_conditions.append(nerf_embed)
      if self.use_rgb_condition:
        rgb_conditions.append(nerf_embed)

    # The condition inputs have a shape of (B, C) now rather than (B, S, C)
    # since we assume all samples have the same condition input. We might want
    # to change this later.
    alpha_conditions = (
        jnp.concatenate(alpha_conditions, axis=-1)
        if alpha_conditions else None)
    rgb_conditions = (
        jnp.concatenate(rgb_conditions, axis=-1)
        if rgb_conditions else None)
    return alpha_conditions, rgb_conditions

  def query_template(self,
                     level,
                     points,
                     viewdirs,
                     metadata,
                     extra_params,
                     metadata_encoded=False):
    """Queries the NeRF template."""
    alpha_condition, rgb_condition = (
        self.get_condition_inputs(viewdirs, metadata, metadata_encoded))

    points_feat = model_utils.posenc(
        points[..., :3],
        min_deg=self.spatial_point_min_deg,
        max_deg=self.spatial_point_max_deg,
        use_identity=self.use_posenc_identity,
        alpha=extra_params['nerf_alpha'])
    # Encode hyper-points if present.
    if points.shape[-1] > 3:
      hyper_feats = model_utils.posenc(
          points[..., 3:],
          min_deg=self.hyper_point_min_deg,
          max_deg=self.hyper_point_max_deg,
          use_identity=False,
          alpha=extra_params['hyper_alpha'])
      points_feat = jnp.concatenate([points_feat, hyper_feats], axis=-1)

    raw = self.nerf_mlps[level](points_feat, alpha_condition, rgb_condition)
    raw = model_utils.noise_regularize(
        self.make_rng(level), raw, self.noise_std, self.use_stratified_sampling)

    rgb = nn.sigmoid(raw['rgb'])
    sigma = self.sigma_activation(jnp.squeeze(raw['alpha'], axis=-1))

    return rgb, sigma

  def map_spatial_points(self, points, warp_embed, extra_params, use_warp=True,
                         return_warp_jacobian=False):
    warp_jacobian = None
    if self.use_warp and use_warp:
      warp_fn = jax.vmap(jax.vmap(self.warp_field, in_axes=(0, 0, None, None)),
                         in_axes=(0, 0, None, None))
      warp_out = warp_fn(points,
                         warp_embed,
                         extra_params,
                         return_warp_jacobian)
      if return_warp_jacobian:
        warp_jacobian = warp_out['jacobian']
      warped_points = warp_out['warped_points']
    else:
      warped_points = points

    return warped_points, warp_jacobian

  def map_hyper_points(self, points, hyper_embed, extra_params,
                       hyper_point_override=None):
    """Maps input points to hyper points.

    Args:
      points: the input points.
      hyper_embed: the hyper embeddings.
      extra_params: extra params to pass to the slicing MLP if applicable.
      hyper_point_override: this may contain an override for the hyper points.
        Useful for rendering at specific hyper dimensions.

    Returns:
      An array of hyper points.
    """
    if hyper_point_override is not None:
      hyper_points = jnp.broadcast_to(
          hyper_point_override[:, None, :],
          (*points.shape[:-1], hyper_point_override.shape[-1]))
    elif self.hyper_slice_method == 'axis_aligned_plane':
      hyper_points = hyper_embed
    elif self.hyper_slice_method == 'bendy_sheet':
      hyper_points = self.hyper_sheet_mlp(
          points,
          hyper_embed,
          alpha=extra_params['hyper_sheet_alpha'])
    else:
      return None

    return hyper_points

  def map_points(self, points, warp_embed, hyper_embed, extra_params,
                 use_warp=True, return_warp_jacobian=False,
                 hyper_point_override=None):
    """Map input points to warped spatial and hyper points.

    Args:
      points: the input points to warp.
      warp_embed: the warp embeddings.
      hyper_embed: the hyper embeddings.
      extra_params: extra parameters to pass to the warp field/hyper field.
      use_warp: whether to use the warp or not.
      return_warp_jacobian: whether to return the warp jacobian or not.
      hyper_point_override: this may contain an override for the hyper points.
        Useful for rendering at specific hyper dimensions.

    Returns:
      A tuple containing `(warped_points, warp_jacobian)`.
    """
    # Map input points to warped spatial and hyper points.
    spatial_points, warp_jacobian = self.map_spatial_points(
        points, warp_embed, extra_params, use_warp=use_warp,
        return_warp_jacobian=return_warp_jacobian)
    hyper_points = self.map_hyper_points(
        points, hyper_embed, extra_params,
        # Override hyper points if present in metadata dict.
        hyper_point_override=hyper_point_override)

    if hyper_points is not None:
      warped_points = jnp.concatenate([spatial_points, hyper_points], axis=-1)
    else:
      warped_points = spatial_points

    return warped_points, warp_jacobian

  def apply_warp(self, points, warp_embed, extra_params):
    warp_embed = self.warp_embed(warp_embed)
    return self.warp_field(points, warp_embed, extra_params)

  def render_samples(self,
                     level,
                     points,
                     z_vals,
                     directions,
                     viewdirs,
                     metadata,
                     extra_params,
                     use_warp=True,
                     metadata_encoded=False,
                     return_warp_jacobian=False,
                     use_sample_at_infinity=False,
                     render_opts=None):
    out = {'points': points}

    batch_shape = points.shape[:-1]
    # Create the warp embedding.
    if use_warp:
      if metadata_encoded:
        warp_embed = metadata['encoded_warp']
      else:
        warp_embed = metadata[self.warp_embed_key]
        warp_embed = self.warp_embed(warp_embed)
    else:
      warp_embed = None

    # Create the hyper embedding.
    if self.has_hyper_embed:
      if metadata_encoded:
        hyper_embed = metadata['encoded_hyper']
      elif self.hyper_use_warp_embed:
        hyper_embed = warp_embed
      else:
        hyper_embed = metadata[self.hyper_embed_key]
        hyper_embed = self.hyper_embed(hyper_embed)
    else:
      hyper_embed = None

    # Broadcast embeddings.
    if warp_embed is not None:
      warp_embed = jnp.broadcast_to(
          warp_embed[:, jnp.newaxis, :],
          shape=(*batch_shape, warp_embed.shape[-1]))
    if hyper_embed is not None:
      hyper_embed = jnp.broadcast_to(
          hyper_embed[:, jnp.newaxis, :],
          shape=(*batch_shape, hyper_embed.shape[-1]))

    # Map input points to warped spatial and hyper points.
    warped_points, warp_jacobian = self.map_points(
        points, warp_embed, hyper_embed, extra_params, use_warp=use_warp,
        return_warp_jacobian=return_warp_jacobian,
        # Override hyper points if present in metadata dict.
        hyper_point_override=metadata.get('hyper_point'))

    rgb, sigma = self.query_template(
        level,
        warped_points,
        viewdirs,
        metadata,
        extra_params=extra_params,
        metadata_encoded=metadata_encoded)

    # Filter densities based on rendering options.
    sigma = filter_sigma(points, sigma, render_opts)

    if warp_jacobian is not None:
      out['warp_jacobian'] = warp_jacobian
    out['warped_points'] = warped_points
    out.update(model_utils.volumetric_rendering(
        rgb,
        sigma,
        z_vals,
        directions,
        use_white_background=self.use_white_background,
        sample_at_infinity=use_sample_at_infinity))

    # Add a map containing the returned points at the median depth.
    depth_indices = model_utils.compute_depth_index(out['weights'])
    med_points = jnp.take_along_axis(
        # Unsqueeze axes: sample axis, coords.
        warped_points, depth_indices[..., None, None], axis=-2)
    out['med_points'] = med_points

    return out

  def __call__(
      self,
      rays_dict: Dict[str, Any],
      extra_params: Dict[str, Any],
      metadata_encoded=False,
      use_warp=True,
      return_points=False,
      return_weights=False,
      return_warp_jacobian=False,
      near=None,
      far=None,
      use_sample_at_infinity=None,
      render_opts=None,
      deterministic=False,
  ):
    """Nerf Model.

    Args:
      rays_dict: a dictionary containing the ray information. Contains:
        'origins': the ray origins.
        'directions': unit vectors which are the ray directions.
        'viewdirs': (optional) unit vectors which are viewing directions.
        'metadata': a dictionary of metadata indices e.g., for warping.
      extra_params: parameters for the warp e.g., alpha.
      metadata_encoded: if True, assume the metadata is already encoded.
      use_warp: if True use the warp field (if also enabled in the model).
      return_points: if True return the points (and warped points if
        applicable).
      return_weights: if True return the density weights.
      return_warp_jacobian: if True computes and returns the warp Jacobians.
      near: if not None override the default near value.
      far: if not None override the default far value.
      use_sample_at_infinity: override for `self.use_sample_at_infinity`.
      render_opts: an optional dictionary of render options.
      deterministic: whether evaluation should be deterministic.

    Returns:
      ret: list, [(rgb, disp, acc), (rgb_coarse, disp_coarse, acc_coarse)]
    """
    use_warp = self.use_warp and use_warp
    # Extract viewdirs from the ray array
    origins = rays_dict['origins']
    directions = rays_dict['directions']
    metadata = rays_dict['metadata']
    if 'viewdirs' in rays_dict:
      viewdirs = rays_dict['viewdirs']
    else:  # viewdirs are normalized rays_d
      viewdirs = directions

    if near is None:
      near = self.near
    if far is None:
      far = self.far
    if use_sample_at_infinity is None:
      use_sample_at_infinity = self.use_sample_at_infinity

    # Evaluate coarse samples.
    z_vals, points = model_utils.sample_along_rays(
        self.make_rng('coarse'), origins, directions, self.num_coarse_samples,
        near, far, self.use_stratified_sampling,
        self.use_linear_disparity)
    coarse_ret = self.render_samples(
        'coarse',
        points,
        z_vals,
        directions,
        viewdirs,
        metadata,
        extra_params,
        use_warp=use_warp,
        metadata_encoded=metadata_encoded,
        return_warp_jacobian=return_warp_jacobian,
        use_sample_at_infinity=self.use_sample_at_infinity)
    out = {'coarse': coarse_ret}

    # Evaluate fine samples.
    if self.num_fine_samples > 0:
      z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
      z_vals, points = model_utils.sample_pdf(
          self.make_rng('fine'), z_vals_mid, coarse_ret['weights'][..., 1:-1],
          origins, directions, z_vals, self.num_fine_samples,
          self.use_stratified_sampling)
      out['fine'] = self.render_samples(
          'fine',
          points,
          z_vals,
          directions,
          viewdirs,
          metadata,
          extra_params,
          use_warp=use_warp,
          metadata_encoded=metadata_encoded,
          return_warp_jacobian=return_warp_jacobian,
          use_sample_at_infinity=use_sample_at_infinity,
          render_opts=render_opts)

    if not return_weights:
      del out['coarse']['weights']
      del out['fine']['weights']

    if not return_points:
      del out['coarse']['points']
      del out['coarse']['warped_points']
      del out['fine']['points']
      del out['fine']['warped_points']

    return out


def construct_nerf(key, batch_size: int, embeddings_dict: Dict[str, int],
                   near: float, far: float):
  """Neural Randiance Field.

  Args:
    key: jnp.ndarray. Random number generator.
    batch_size: the evaluation batch size used for shape inference.
    embeddings_dict: a dictionary containing the embeddings for each metadata
      type.
    near: the near plane of the scene.
    far: the far plane of the scene.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  model = NerfModel(
      embeddings_dict=immutabledict.immutabledict(embeddings_dict),
      near=near,
      far=far)

  init_rays_dict = {
      'origins': jnp.ones((batch_size, 3), jnp.float32),
      'directions': jnp.ones((batch_size, 3), jnp.float32),
      'metadata': {
          'warp': jnp.ones((batch_size, 1), jnp.uint32),
          'camera': jnp.ones((batch_size, 1), jnp.uint32),
          'appearance': jnp.ones((batch_size, 1), jnp.uint32),
          'time': jnp.ones((batch_size, 1), jnp.float32),
      }
  }
  extra_params = {
      'nerf_alpha': 0.0,
      'warp_alpha': 0.0,
      'hyper_alpha': 0.0,
      'hyper_sheet_alpha': 0.0,
  }

  key, key1, key2 = random.split(key, 3)
  params = model.init({
      'params': key,
      'coarse': key1,
      'fine': key2
  }, init_rays_dict, extra_params=extra_params)['params']

  return model, params
