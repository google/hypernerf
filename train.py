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

# Lint as: python3
"""Training script for Nerf."""

import functools
from typing import Dict, Union

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
import jax
from jax import numpy as jnp
from jax import random
import numpy as np
import tensorflow as tf

from hypernerf import configs
from hypernerf import datasets
from hypernerf import gpath
from hypernerf import model_utils
from hypernerf import models
from hypernerf import schedules
from hypernerf import training
from hypernerf import utils

flags.DEFINE_enum('mode', None, ['jax_cpu', 'jax_gpu', 'jax_tpu'],
                  'Distributed strategy approach.')

flags.DEFINE_string('base_folder', None, 'where to store ckpts and logs')
flags.mark_flag_as_required('base_folder')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
flags.DEFINE_multi_string('gin_configs', (), 'Gin config files.')
FLAGS = flags.FLAGS


def _log_to_tensorboard(writer: tensorboard.SummaryWriter,
                        state: model_utils.TrainState,
                        scalar_params: training.ScalarParams,
                        stats: Dict[str, Union[Dict[str, jnp.ndarray],
                                               jnp.ndarray]],
                        time_dict: Dict[str, jnp.ndarray]):
  """Log statistics to Tensorboard."""
  step = int(state.optimizer.state.step)

  def _log_scalar(tag, value):
    if value is not None:
      writer.scalar(tag, value, step)

  _log_scalar('params/learning_rate', scalar_params.learning_rate)
  _log_scalar('params/nerf_alpha', state.nerf_alpha)
  _log_scalar('params/warp_alpha', state.warp_alpha)
  _log_scalar('params/hyper_sheet_alpha', state.hyper_sheet_alpha)
  _log_scalar('params/elastic_loss/weight', scalar_params.elastic_loss_weight)

  # pmean is applied in train_step so just take the item.
  for branch in {'coarse', 'fine'}:
    if branch not in stats:
      continue
    for stat_key, stat_value in stats[branch].items():
      writer.scalar(f'{stat_key}/{branch}', stat_value, step)

  _log_scalar('loss/background', stats.get('background_loss'))

  for k, v in time_dict.items():
    writer.scalar(f'time/{k}', v, step)


def _log_histograms(writer: tensorboard.SummaryWriter,
                    state: model_utils.TrainState,
                    model_out):
  """Log histograms to Tensorboard."""
  step = int(state.optimizer.state.step)
  params = state.optimizer.target['model']
  if 'nerf_embed' in params:
    embeddings = params['nerf_embed']['embed']['embedding']
    writer.histogram('nerf_embedding', embeddings, step)
  if 'hyper_embed' in params:
    embeddings = params['hyper_embed']['embed']['embedding']
    writer.histogram('hyper_embedding', embeddings, step)
  if 'warp_embed' in params:
    embeddings = params['warp_embed']['embed']['embedding']
    writer.histogram('warp_embedding', embeddings, step)

  for branch in {'coarse', 'fine'}:
    if 'warped_points' in model_out[branch]:
      points = model_out[branch]['points']
      warped_points = model_out[branch]['warped_points']
      writer.histogram(f'{branch}/spatial_points',
                       warped_points[..., :3], step)
      writer.histogram(f'{branch}/spatial_points_delta',
                       warped_points[..., :3] - points, step)
      if warped_points.shape[-1] > 3:
        writer.histogram(f'{branch}/hyper_points',
                         warped_points[..., 3:], step)


def _log_grads(writer: tensorboard.SummaryWriter, model: models.NerfModel,
               state: model_utils.TrainState):
  """Log histograms to Tensorboard."""
  step = int(state.optimizer.state.step)
  params = state.optimizer.target['model']
  if 'nerf_metadata_encoder' in params:
    embeddings = params['nerf_metadata_encoder']['embed']['embedding']
    writer.histogram('nerf_embedding', embeddings, step)
  if 'hyper_metadata_encoder' in params:
    embeddings = params['hyper_metadata_encoder']['embed']['embedding']
    writer.histogram('hyper_embedding', embeddings, step)
  if 'warp_field' in params and model.warp_metadata_config['type'] == 'glo':
    embeddings = params['warp_metadata_encoder']['embed']['embedding']
    writer.histogram('warp_embedding', embeddings, step)


def main(argv):
  jax.config.parse_flags_with_absl()
  tf.config.experimental.set_visible_devices([], 'GPU')
  del argv
  logging.info('*** Starting experiment')
  # Assume G3 path for config files when running locally.
  gin_configs = FLAGS.gin_configs

  logging.info('*** Loading Gin configs from: %s', str(gin_configs))
  gin.parse_config_files_and_bindings(
      config_files=gin_configs,
      bindings=FLAGS.gin_bindings,
      skip_unknown=True)

  # Load configurations.
  exp_config = configs.ExperimentConfig()
  train_config = configs.TrainConfig()
  dummy_model = models.NerfModel({}, 0, 0)

  # Get directory information.
  exp_dir = gpath.GPath(FLAGS.base_folder)
  if exp_config.subname:
    exp_dir = exp_dir / exp_config.subname
  summary_dir = exp_dir / 'summaries' / 'train'
  checkpoint_dir = exp_dir / 'checkpoints'

  # Log and create directories if this is the main process.
  if jax.process_index() == 0:
    logging.info('exp_dir = %s', exp_dir)
    if not exp_dir.exists():
      exp_dir.mkdir(parents=True, exist_ok=True)

    logging.info('summary_dir = %s', summary_dir)
    if not summary_dir.exists():
      summary_dir.mkdir(parents=True, exist_ok=True)

    logging.info('checkpoint_dir = %s', checkpoint_dir)
    if not checkpoint_dir.exists():
      checkpoint_dir.mkdir(parents=True, exist_ok=True)

  logging.info('Starting process %d. There are %d processes.',
               jax.process_index(), jax.process_count())
  logging.info('Found %d accelerator devices: %s.', jax.local_device_count(),
               str(jax.local_devices()))
  logging.info('Found %d total devices: %s.', jax.device_count(),
               str(jax.devices()))

  rng = random.PRNGKey(exp_config.random_seed)
  # Shift the numpy random seed by process_index() to shuffle data loaded by
  # different processes.
  np.random.seed(exp_config.random_seed + jax.process_index())

  if train_config.batch_size % jax.device_count() != 0:
    raise ValueError('Batch size must be divisible by the number of devices.')

  devices = jax.local_devices()
  logging.info('Creating datasource')
  datasource = exp_config.datasource_cls(
      image_scale=exp_config.image_scale,
      random_seed=exp_config.random_seed,
      # Enable metadata based on model needs.
      use_warp_id=dummy_model.use_warp,
      use_appearance_id=(
          dummy_model.nerf_embed_key == 'appearance'
          or dummy_model.hyper_embed_key == 'appearance'),
      use_camera_id=dummy_model.nerf_embed_key == 'camera',
      use_time=dummy_model.warp_embed_key == 'time')

  # Create Model.
  logging.info('Initializing models.')
  rng, key = random.split(rng)
  params = {}
  model, params['model'] = models.construct_nerf(
      key,
      batch_size=train_config.batch_size,
      embeddings_dict=datasource.embeddings_dict,
      near=datasource.near,
      far=datasource.far)

  # Create Jax iterator.
  logging.info('Creating dataset iterator.')
  train_iter = datasource.create_iterator(
      datasource.train_ids,
      flatten=True,
      shuffle=True,
      batch_size=train_config.batch_size,
      prefetch_size=3,
      shuffle_buffer_size=train_config.shuffle_buffer_size,
      devices=devices,
  )

  points_iter = None
  if train_config.use_background_loss:
    points = datasource.load_points(shuffle=True)
    points_batch_size = min(
        len(points),
        len(devices) * train_config.background_points_batch_size)
    points_batch_size -= points_batch_size % len(devices)
    points_dataset = tf.data.Dataset.from_tensor_slices(points)
    points_iter = datasets.iterator_from_dataset(
        points_dataset,
        batch_size=points_batch_size,
        prefetch_size=3,
        devices=devices)

  learning_rate_sched = schedules.from_config(train_config.lr_schedule)
  nerf_alpha_sched = schedules.from_config(train_config.nerf_alpha_schedule)
  warp_alpha_sched = schedules.from_config(train_config.warp_alpha_schedule)
  hyper_alpha_sched = schedules.from_config(train_config.hyper_alpha_schedule)
  hyper_sheet_alpha_sched = schedules.from_config(
      train_config.hyper_sheet_alpha_schedule)
  elastic_loss_weight_sched = schedules.from_config(
      train_config.elastic_loss_weight_schedule)

  optimizer_def = optim.Adam(learning_rate_sched(0))
  if train_config.use_weight_norm:
    optimizer_def = optim.WeightNorm(optimizer_def)
  optimizer = optimizer_def.create(params)
  state = model_utils.TrainState(
      optimizer=optimizer,
      nerf_alpha=nerf_alpha_sched(0),
      warp_alpha=warp_alpha_sched(0),
      hyper_alpha=hyper_alpha_sched(0),
      hyper_sheet_alpha=hyper_sheet_alpha_sched(0))
  scalar_params = training.ScalarParams(
      learning_rate=learning_rate_sched(0),
      elastic_loss_weight=elastic_loss_weight_sched(0),
      warp_reg_loss_weight=train_config.warp_reg_loss_weight,
      warp_reg_loss_alpha=train_config.warp_reg_loss_alpha,
      warp_reg_loss_scale=train_config.warp_reg_loss_scale,
      background_loss_weight=train_config.background_loss_weight,
      hyper_reg_loss_weight=train_config.hyper_reg_loss_weight)
  state = checkpoints.restore_checkpoint(checkpoint_dir, state)
  init_step = state.optimizer.state.step + 1
  state = jax_utils.replicate(state, devices=devices)
  del params

  summary_writer = None
  if jax.process_index() == 0:
    config_str = gin.operative_config_str()
    logging.info('Configuration: \n%s', config_str)
    with (exp_dir / 'config.gin').open('w') as f:
      f.write(config_str)
    summary_writer = tensorboard.SummaryWriter(str(summary_dir))
    summary_writer.text('gin/train', textdata=gin.markdown(config_str), step=0)

  train_step = functools.partial(
      training.train_step,
      model,
      elastic_reduce_method=train_config.elastic_reduce_method,
      elastic_loss_type=train_config.elastic_loss_type,
      use_elastic_loss=train_config.use_elastic_loss,
      use_background_loss=train_config.use_background_loss,
      use_warp_reg_loss=train_config.use_warp_reg_loss,
      use_hyper_reg_loss=train_config.use_hyper_reg_loss,
  )
  ptrain_step = jax.pmap(
      train_step,
      axis_name='batch',
      devices=devices,
      # rng_key, state, batch, scalar_params.
      in_axes=(0, 0, 0, None),
      # Treat use_elastic_loss as compile-time static.
      donate_argnums=(2,),  # Donate the 'batch' argument.
  )

  if devices:
    n_local_devices = len(devices)
  else:
    n_local_devices = jax.local_device_count()

  logging.info('Starting training')
  # Make random seed separate across processes.
  rng = rng + jax.process_index()
  keys = random.split(rng, n_local_devices)
  time_tracker = utils.TimeTracker()
  time_tracker.tic('data', 'total')
  for step, batch in zip(range(init_step, train_config.max_steps + 1),
                         train_iter):
    if points_iter is not None:
      batch['background_points'] = next(points_iter)
    time_tracker.toc('data')
    # See: b/162398046.
    # pytype: disable=attribute-error
    scalar_params = scalar_params.replace(
        learning_rate=learning_rate_sched(step),
        elastic_loss_weight=elastic_loss_weight_sched(step))
    # pytype: enable=attribute-error
    nerf_alpha = jax_utils.replicate(nerf_alpha_sched(step), devices)
    warp_alpha = jax_utils.replicate(warp_alpha_sched(step), devices)
    hyper_alpha = jax_utils.replicate(hyper_alpha_sched(step), devices)
    hyper_sheet_alpha = jax_utils.replicate(
        hyper_sheet_alpha_sched(step), devices)
    state = state.replace(nerf_alpha=nerf_alpha,
                          warp_alpha=warp_alpha,
                          hyper_alpha=hyper_alpha,
                          hyper_sheet_alpha=hyper_sheet_alpha)

    with time_tracker.record_time('train_step'):
      state, stats, keys, model_out = ptrain_step(
          keys, state, batch, scalar_params)
      time_tracker.toc('total')

    if step % train_config.print_every == 0 and jax.process_index() == 0:
      logging.info('step=%d, nerf_alpha=%.04f, warp_alpha=%.04f, %s', step,
                   nerf_alpha_sched(step),
                   warp_alpha_sched(step),
                   time_tracker.summary_str('last'))
      coarse_metrics_str = ', '.join(
          [f'{k}={v.mean():.04f}' for k, v in stats['coarse'].items()])
      fine_metrics_str = ', '.join(
          [f'{k}={v.mean():.04f}' for k, v in stats['fine'].items()])
      logging.info('\tcoarse metrics: %s', coarse_metrics_str)
      if 'fine' in stats:
        logging.info('\tfine metrics: %s', fine_metrics_str)

    if step % train_config.save_every == 0 and jax.process_index() == 0:
      training.save_checkpoint(checkpoint_dir, state, keep=2)

    if step % train_config.log_every == 0 and jax.process_index() == 0:
      # Only log via process 0.
      _log_to_tensorboard(
          summary_writer,
          jax_utils.unreplicate(state),
          scalar_params,
          jax_utils.unreplicate(stats),
          time_dict=time_tracker.summary('mean'))
      time_tracker.reset()

    if step % train_config.histogram_every == 0 and jax.process_index() == 0:
      _log_histograms(summary_writer, jax_utils.unreplicate(state), model_out)

    time_tracker.tic('data', 'total')

  if train_config.max_steps % train_config.save_every != 0:
    training.save_checkpoint(checkpoint_dir, state, keep=2)


if __name__ == '__main__':
  app.run(main)
