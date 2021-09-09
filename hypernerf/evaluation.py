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

"""Module for evaluating a trained NeRF."""
import math
import time

from absl import logging
from flax import jax_utils
import jax
from jax import tree_util
import jax.numpy as jnp
import numpy as np

from hypernerf import utils


def encode_metadata(model, params, metadata):
  """Encodes metadata embeddings.

  Args:
    model: a NerfModel.
    params: the parameters of the model.
    metadata: the metadata dict.

  Returns:
    A new metadata dict with the encoded embeddings.
  """
  encoded_metadata = {}
  if model.use_nerf_embed:
    encoded_metadata['encoded_nerf'] = model.apply(
        {'params': params}, metadata, method=model.encode_nerf_embed)
  if model.use_warp:
    encoded_metadata['encoded_warp'] = model.apply(
        {'params': params}, metadata, method=model.encode_warp_embed)
  if model.has_hyper_embed:
    encoded_metadata['encoded_hyper'] = model.apply(
        {'params': params}, metadata, method=model.encode_hyper_embed)
  return encoded_metadata


def render_image(
    state,
    rays_dict,
    model_fn,
    device_count,
    rng,
    chunk=8192,
    default_ret_key=None):
  """Render all the pixels of an image (in test mode).

  Args:
    state: model_utils.TrainState.
    rays_dict: dict, test example.
    model_fn: function, jit-ed render function.
    device_count: The number of devices to shard batches over.
    rng: The random number generator.
    chunk: int, the size of chunks to render sequentially.
    default_ret_key: either 'fine' or 'coarse'. If None will default to highest.

  Returns:
    rgb: jnp.ndarray, rendered color image.
    depth: jnp.ndarray, rendered depth.
    acc: jnp.ndarray, rendered accumulated weights per pixel.
  """
  batch_shape = rays_dict['origins'].shape[:-1]
  num_rays = np.prod(batch_shape)
  rays_dict = tree_util.tree_map(lambda x: x.reshape((num_rays, -1)), rays_dict)
  _, key_0, key_1 = jax.random.split(rng, 3)
  key_0 = jax.random.split(key_0, device_count)
  key_1 = jax.random.split(key_1, device_count)
  proc_id = jax.process_index()
  ret_maps = []
  start_time = time.time()
  num_batches = int(math.ceil(num_rays / chunk))
  logging.info('Rendering: num_batches = %d, num_rays = %d, chunk = %d',
               num_batches, num_rays, chunk)
  for batch_idx in range(num_batches):
    ray_idx = batch_idx * chunk
    logging.log_every_n_seconds(
        logging.INFO, 'Rendering batch %d/%d (%d/%d)', 2.0,
        batch_idx, num_batches, ray_idx, num_rays)
    # pylint: disable=cell-var-from-loop
    chunk_slice_fn = lambda x: x[ray_idx:ray_idx + chunk]
    chunk_rays_dict = tree_util.tree_map(chunk_slice_fn, rays_dict)
    num_chunk_rays = chunk_rays_dict['origins'].shape[0]
    remainder = num_chunk_rays % device_count
    if remainder != 0:
      padding = device_count - remainder
      # pylint: disable=cell-var-from-loop
      chunk_pad_fn = lambda x: jnp.pad(x, ((0, padding), (0, 0)), mode='edge')
      chunk_rays_dict = tree_util.tree_map(chunk_pad_fn, chunk_rays_dict)
    else:
      padding = 0
    # After padding the number of chunk_rays is always divisible by
    # proc_count.
    per_proc_rays = num_chunk_rays // jax.process_count()
    logging.debug(
        'Rendering batch: num_chunk_rays = %d, padding = %d, remainder = %d, '
        'per_proc_rays = %d',
        num_chunk_rays, padding, remainder, per_proc_rays)
    chunk_rays_dict = tree_util.tree_map(
        lambda x: x[(proc_id * per_proc_rays):((proc_id + 1) * per_proc_rays)],
        chunk_rays_dict)
    chunk_rays_dict = utils.shard(chunk_rays_dict, device_count)
    model_out = model_fn(key_0, key_1, state.optimizer.target['model'],
                         chunk_rays_dict, state.extra_params)
    if not default_ret_key:
      ret_key = 'fine' if 'fine' in model_out else 'coarse'
    else:
      ret_key = default_ret_key
    ret_map = jax_utils.unreplicate(model_out[ret_key])
    ret_map = jax.tree_map(lambda x: utils.unshard(x, padding), ret_map)
    ret_maps.append(ret_map)
  ret_map = jax.tree_multimap(lambda *x: jnp.concatenate(x, axis=0), *ret_maps)
  logging.info('Rendering took %.04s', time.time() - start_time)
  out = {}
  for key, value in ret_map.items():
    out_shape = (*batch_shape, *value.shape[1:])
    logging.debug('Reshaping %s of shape %s to %s',
                  key, str(value.shape), str(out_shape))
    out[key] = value.reshape(out_shape)

  return out
