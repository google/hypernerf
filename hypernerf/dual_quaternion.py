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

"""Dual quaternion math.

We encode a dual quaternion as an 8-dimensional array:
  [rx, ry, rz, rw, dx, dy, dz, dw]
which represent the dual quaternion:
  r + εd = (rx*i, ry*j, rz*k, rw) + ε(dx*i, dy*j, dz*k, dw)

References:
  https://en.wikipedia.org/wiki/Dual_quaternion
"""
from jax import numpy as jnp
from hypernerf import quaternion


def real_part(dq):
  """Returns the real part of the dual quaternion."""
  return dq[..., :4]


def dual_part(dq):
  """Returns the dual part of the dual quaternion."""
  return dq[..., 4:]


def split_parts(dq):
  """Splits the dual quaternion into its real and dual parts."""
  return real_part(dq), dual_part(dq)


def join_parts(real, dual):
  """Creates a dual quaternion from its real and dual parts."""
  return jnp.concatenate((real, dual), axis=-1)


def identity(dtype=jnp.float32):
  """Returns the dual quaternion encoding an identity transform."""
  return jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype)


def add(dq1, dq2):
  """Adds two dual quaternions."""
  return dq1 + dq2


def multiply(dq1, dq2):
  """Dual quaternion multiplication.

  Args:
    dq1: a (*,8) dimensional array representing a dual quaternion.
    dq2: a (*,8) dimensional array representing a dual quaternion.

  Returns:
    A (*,8) dimensional array representing the output dual quaternion.
  """
  a, b = split_parts(dq1)
  c, d = split_parts(dq2)

  real = quaternion.multiply(a, c)
  dual = quaternion.multiply(a, d) + quaternion.multiply(b, c)

  return join_parts(real, dual)


def quaternion_conjugate(dq):
  """Returns the quaternion conjugate."""
  real, dual = split_parts(dq)
  return join_parts(quaternion.conjugate(real), quaternion.conjugate(dual))


def dual_conjugate(dq):
  """Returns the dual number conjugate."""
  real, dual = split_parts(dq)
  return join_parts(real, -dual)


def quaternion_dual_conjugate(dq):
  """Returns the dual number and quaternion conjugate."""
  real, dual = split_parts(dq)
  return join_parts(-quaternion.conjugate(real), -quaternion.conjugate(dual))


def normalize(dq):
  """Normalize a dual quaternion."""
  real, dual = split_parts(dq)
  real_norm = quaternion.norm(real)
  return join_parts(real / real_norm, dual / real_norm)


def get_rotation(dq):
  """Returns a rotation quaternion this dual quaternion encodes."""
  return real_part(dq)


def get_translation(dq):
  """Returns a translation vector this dual quaternion encodes."""
  real, dual = split_parts(dq)
  return 2 * quaternion.im(
      quaternion.multiply(dual, quaternion.conjugate(real)))


def from_rotation_translation(q, t):
  """Creates a dual quaternion from a rotation and translation.

  Args:
    q: a (*,4) array containing a rotation quaternion.
    t: a (*,3) array containing a translation vector.

  Returns:
    A (*,8) array containing a dual quaternion.
  """
  # Pad t = [t; 0]
  t = jnp.concatenate((t, jnp.zeros_like(t[..., -1:])), axis=-1)
  dq_t = join_parts(quaternion.identity(), 0.5 * t)
  dq_r = join_parts(q, jnp.zeros_like(q))
  return multiply(dq_t, dq_r)
