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
"""Non-differentiable utility functions."""
import collections
from concurrent import futures
import contextlib
import functools
import time
from typing import List, Union

import jax
from jax import tree_util
import jax.numpy as jnp
import numpy as np
from scipy import interpolate
from scipy.spatial import transform as scipy_transform


def clip_gradients(grad, grad_max_val=0.0, grad_max_norm=0.0, eps=1e-7):
  """Gradient clipping."""
  # Clip the gradient by value.
  if grad_max_val > 0:
    clip_fn = lambda z: jnp.clip(z, -grad_max_val, grad_max_val)
    grad = jax.tree_util.tree_map(clip_fn, grad)

  # Clip the (possibly value-clipped) gradient by norm.
  if grad_max_norm > 0:
    grad_norm = safe_sqrt(
        jax.tree_util.tree_reduce(
            lambda x, y: x + jnp.sum(y**2), grad, initializer=0))
    mult = jnp.minimum(1, grad_max_norm / (eps + grad_norm))
    grad = jax.tree_util.tree_map(lambda z: mult * z, grad)

  return grad


def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


# pylint: disable=unused-argument
@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def safe_norm(x, axis=-1, keepdims=False, tol=1e-9):
  """Calculates a np.linalg.norm(d) that's safe for gradients at d=0.

  These gymnastics are to avoid a poorly defined gradient for np.linal.norm(0)
  see https://github.com/google/jax/issues/3058 for details


  Args:
    x: A np.array
    axis: The axis along which to compute the norm
    keepdims: if True don't squeeze the axis.
    tol: the absolute threshold within which to zero out the gradient.

  Returns:
    Equivalent to np.linalg.norm(d)
  """
  return jnp.linalg.norm(x, axis=axis, keepdims=keepdims)


@safe_norm.defjvp
def _safe_norm_jvp(axis, keepdims, tol, primals, tangents):
  """Custom JVP rule for safe_norm."""
  x, = primals
  x_dot, = tangents
  safe_tol = max(tol, 1e-30)
  y = safe_norm(x, tol=safe_tol, axis=axis, keepdims=True)
  y_safe = jnp.maximum(y, tol)  # Prevent divide by zero.
  y_dot = jnp.where(y > safe_tol, x_dot * x / y_safe, jnp.zeros_like(x))
  y_dot = jnp.sum(y_dot, axis=axis, keepdims=True)
  # Squeeze the axis if `keepdims` is True.
  if not keepdims:
    y = jnp.squeeze(y, axis=axis)
    y_dot = jnp.squeeze(y_dot, axis=axis)
  return y, y_dot


def jacobian_to_curl(jacobian):
  """Computes the curl from the Jacobian."""
  dfx_dy = jacobian[..., 0, 1]
  dfx_dz = jacobian[..., 0, 2]
  dfy_dx = jacobian[..., 1, 0]
  dfy_dz = jacobian[..., 1, 2]
  dfz_dx = jacobian[..., 2, 0]
  dfz_dy = jacobian[..., 2, 1]

  return jnp.stack([
      dfz_dy - dfy_dz,
      dfx_dz - dfz_dx,
      dfy_dx - dfx_dy,
      ], axis=-1)


def jacobian_to_div(jacobian):
  """Computes the divergence from the Jacobian."""
  # If F : x -> x + f(x) then dF/dx = 1 + df/dx, so subtract 1 for each
  # diagonal of the Jacobian.
  return jnp.trace(jacobian, axis1=-2, axis2=-1) - 3.0


def compute_psnr(mse):
  """Compute psnr value given mse (we assume the maximum pixel value is 1).

  Args:
    mse: float, mean square error of pixels.

  Returns:
    psnr: float, the psnr value.
  """
  return -10. * jnp.log(mse) / jnp.log(10.)


@jax.jit
def robust_whiten(x):
  median = jnp.nanmedian(x)
  mad = jnp.nanmean(jnp.abs(x - median))
  return (x - median) / mad


def interpolate_codes(codes: Union[np.ndarray, List[np.ndarray]],
                      num_samples: int,
                      method='spline',
                      bc_type='natural'):
  """Interpolates latent codes.

  Args:
    codes: the codes to interpolate.
    num_samples: the number of samples to interpolate to.
    method: which method to use for interpolation.
    bc_type: interpolation type for spline interpolation.

  Returns:
    (np.ndarray): the interpolated codes.
  """
  if isinstance(codes, list):
    codes = np.array(codes)
  t = np.arange(len(codes))
  xs = np.linspace(0, len(codes) - 1, num_samples)
  if method == 'spline':
    cs = interpolate.CubicSpline(t, codes, bc_type=bc_type)
    return cs(xs).astype(np.float32)
  elif method in {'linear', 'cubic', 'quadratic', 'slinear'}:
    interp = interpolate.interp1d(t, codes, axis=0)
    return interp(xs).astype(np.float32)

  raise ValueError(f'Unknown method {method!r}')


def interpolate_cameras(cameras, num_samples: int):
  """Interpolates the cameras to the number of output samples.

  Uses a spherical linear interpolation (Slerp) to interpolate the camera
  orientations and a cubic spline to interpolate the camera positions.

  Args:
    cameras: the input cameras to interpolate.
    num_samples: the number of output cameras.

  Returns:
    (List[vision_sfm.Camera]): a list of interpolated cameras.
  """
  rotations = []
  positions = []
  for camera in cameras:
    rotations.append(camera.orientation)
    positions.append(camera.position)

  in_times = np.linspace(0, 1, len(rotations))
  slerp = scipy_transform.Slerp(
      in_times, scipy_transform.Rotation.from_dcm(rotations))
  spline = interpolate.CubicSpline(in_times, positions)

  out_times = np.linspace(0, 1, num_samples)
  out_rots = slerp(out_times).as_dcm()
  out_positions = spline(out_times)

  ref_camera = cameras[0]
  out_cameras = []
  for out_rot, out_pos in zip(out_rots, out_positions):
    out_camera = ref_camera.copy()
    out_camera.orientation = out_rot
    out_camera.position = out_pos
    out_cameras.append(out_camera)
  return out_cameras


def safe_sqrt(x, eps=1e-7):
  safe_x = jnp.where(x == 0, jnp.ones_like(x) * eps, x)
  return jnp.sqrt(safe_x)


@jax.jit
def general_loss_with_squared_residual(x_sq, alpha, scale):
  r"""Implements the general form of the loss.

  This implements the rho(x, \alpha, c) function described in "A General and
  Adaptive Robust Loss Function", Jonathan T. Barron,
  https://arxiv.org/abs/1701.03077.
  Args:
    x_sq: The residual for which the loss is being computed. x can have any
      shape, and alpha and scale will be broadcasted to match x's shape if
      necessary.
    alpha: The shape parameter of the loss (\alpha in the paper), where more
      negative values produce a loss with more robust behavior (outliers "cost"
      less), and more positive values produce a loss with less robust behavior
      (outliers are penalized more heavily). Alpha can be any value in
      [-infinity, infinity], but the gradient of the loss with respect to alpha
      is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
      interpolation between several discrete robust losses:
        alpha=-Infinity: Welsch/Leclerc Loss.
        alpha=-2: Geman-McClure loss.
        alpha=0: Cauchy/Lortentzian loss.
        alpha=1: Charbonnier/pseudo-Huber loss.
        alpha=2: L2 loss.
    scale: The scale parameter of the loss. When |x| < scale, the loss is an
      L2-like quadratic bowl, and when |x| > scale the loss function takes on a
      different shape according to alpha.

  Returns:
    The losses for each element of x, in the same shape as x.
  """
  eps = jnp.finfo(jnp.float32).eps

  # `scale` must be > 0.
  scale = jnp.maximum(eps, scale)

  # The loss when alpha == 2. This will get reused repeatedly.
  loss_two = 0.5 * x_sq / (scale**2)

  # "Safe" versions of log1p and expm1 that will not NaN-out.
  log1p_safe = lambda x: jnp.log1p(jnp.minimum(x, 3e37))
  expm1_safe = lambda x: jnp.expm1(jnp.minimum(x, 87.5))

  # The loss when not in one of the special casess.
  # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
  a = jnp.where(alpha >= 0, jnp.ones_like(alpha),
                -jnp.ones_like(alpha)) * jnp.maximum(eps, jnp.abs(alpha))
  # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
  b = jnp.maximum(eps, jnp.abs(alpha - 2))
  loss_ow = (b / a) * ((loss_two / (0.5 * b) + 1)**(0.5 * alpha) - 1)

  # Select which of the cases of the loss to return as a function of alpha.
  return scale * jnp.where(
      alpha == -jnp.inf, -expm1_safe(-loss_two),
      jnp.where(
          alpha == 0, log1p_safe(loss_two),
          jnp.where(alpha == 2, loss_two,
                    jnp.where(alpha == jnp.inf, expm1_safe(loss_two),
                              loss_ow))))


def points_bound(points):
  """Computes the min and max dims of the points."""
  min_dim = np.min(points, axis=0)
  max_dim = np.max(points, axis=0)
  return np.stack((min_dim, max_dim), axis=1)


def points_centroid(points):
  """Computes the centroid of the points from the bounding box."""
  return points_bound(points).mean(axis=1)


def points_bounding_size(points):
  """Computes the bounding size of the points from the bounding box."""
  bounds = points_bound(points)
  return np.linalg.norm(bounds[:, 1] - bounds[:, 0])


def shard(xs, device_count=None):
  """Split data into shards for multiple devices along the first dimension."""
  if device_count is None:
    jax.local_device_count()
  return jax.tree_map(lambda x: x.reshape((device_count, -1) + x.shape[1:]), xs)


def to_device(xs):
  """Transfer data to devices (GPU/TPU)."""
  return jax.tree_map(jnp.array, xs)


def unshard(x, padding=0):
  """Collect the sharded tensor to the shape before sharding."""
  if padding > 0:
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))[:-padding]
  else:
    return x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))


def normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x)


def parallel_map(f, iterable, max_threads=None, show_pbar=False, **kwargs):
  """Parallel version of map()."""
  with futures.ThreadPoolExecutor(max_threads) as executor:
    if show_pbar:
      # pylint: disable=g-import-not-at-top
      import tqdm
      results = tqdm.tqdm(
          executor.map(f, iterable, **kwargs), total=len(iterable))
    else:
      results = executor.map(f, iterable, **kwargs)
    return list(results)


def parallel_tree_map(f, tree, **kwargs):
  """Parallel version of jax.tree_map."""
  leaves, treedef = jax.tree_flatten(tree)
  results = parallel_map(f, leaves, **kwargs)
  return jax.tree_unflatten(treedef, results)


def strided_subset(sequence, count):
  """Returns a strided subset of a list."""
  if count:
    stride = max(1, len(sequence) // count)
    return sequence[::stride]
  return sequence


def tree_collate(list_of_pytrees):
  """Collates a list of pytrees with the same structure."""
  return tree_util.tree_multimap(lambda *x: np.stack(x), *list_of_pytrees)


@contextlib.contextmanager
def print_time(name):
  """Records the time elapsed."""
  start = time.time()
  yield
  elapsed = time.time() - start
  print(f'[{name}] time elapsed: {elapsed:.04f}')


class ValueMeter:
  """Tracks the average of a value."""

  def __init__(self):
    self._values = []

  def reset(self):
    """Resets the meter."""
    self._values.clear()

  def update(self, value):
    """Adds a value to the meter."""
    self._values.append(value)

  def reduce(self, reduction='mean'):
    """Reduces the tracked values."""
    if reduction == 'mean':
      return np.mean(self._values)
    elif reduction == 'std':
      return np.std(self._values)
    elif reduction == 'last':
      return self._values[-1]
    else:
      raise ValueError(f'Unknown reduction {reduction}')


class TimeTracker:
  """Tracks the average time elapsed over multiple steps."""

  def __init__(self):
    self._meters = collections.defaultdict(ValueMeter)
    self._marked_time = collections.defaultdict(float)

  @contextlib.contextmanager
  def record_time(self, key: str):
    """Records the time elapsed."""
    start = time.time()
    yield
    elapsed = time.time() - start
    self.update(key, elapsed)

  def update(self, key, value):
    """Updates the time value for a given key."""
    self._meters[key].update(value)

  def tic(self, *args):
    """Marks the starting time of an event."""
    for key in args:
      self._marked_time[key] = time.time()

  def toc(self, *args):
    """Records the time elapsed based on the previous call to `tic`."""
    for key in args:
      self.update(key, time.time() - self._marked_time[key])
      del self._marked_time[key]

  def reset(self):
    """Resets all time meters."""
    for meter in self._meters.values():
      meter.reset()

  def summary(self, reduction='mean'):
    """Returns a dictionary of reduced times."""
    time_dict = {k: v.reduce(reduction) for k, v in self._meters.items()}
    if 'total' not in time_dict:
      time_dict['total'] = sum(time_dict.values())

    time_dict['steps_per_sec'] = 1.0 / time_dict['total']
    return time_dict

  def summary_str(self, reduction='mean'):
    """Returns a string of reduced times."""
    strings = [f'{k}={v:.04f}' for k, v in self.summary(reduction).items()]
    return ', '.join(strings)
