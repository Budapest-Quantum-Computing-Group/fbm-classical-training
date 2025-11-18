#
# Copyright 2021-2024 Budapest Quantum Computing Group
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

from functools import partial

import jax
import jax.numpy as jnp

import numpy as np


@jax.jit
def get_so2_rot(theta):
    cos = jnp.cos(theta)
    sin = jnp.sin(theta)

    return jnp.array([[cos, -sin], [sin, cos]])


@partial(jax.jit, static_argnames="d")
def calculate_SO(weights, d):
    """
    Calculates the SO matrix on the covariance matrix space.
    """
    d_over_2 = d // 2
    len_modes_2_col = 2 * d_over_2 - 1

    SO = np.eye(d)

    def even_col(SO, start_idx):
        no_of_rots = SO.shape[0] // 2

        angles = jax.lax.dynamic_slice(weights, [start_idx], [no_of_rots])

        grouped = SO.reshape(no_of_rots, 2, d)

        R = jax.vmap(get_so2_rot)(angles)

        rotated = jnp.einsum("bij,bjk->bik", R, grouped)

        return rotated.reshape(SO.shape)

    def odd_col(SO, start_idx):
        smaller_SO = SO[1 : (d - 1)]

        rotated = even_col(smaller_SO, start_idx)

        return SO.at[1 : (d - 1)].set(rotated)

    def outer_loop(SO, col):
        start_idx = (col // 2) * len_modes_2_col + d_over_2 * (col % 2)

        SO = jax.lax.cond(
            col % 2 == 0,
            lambda SO: even_col(SO, start_idx),
            lambda SO: odd_col(SO, start_idx),
            SO,
        )

        return SO, None

    SO = jax.lax.scan(outer_loop, SO, jnp.arange(0, d))[0]

    return SO


@partial(jax.jit, static_argnames="d")
def calculate_modes(d):
    """
    The implementation here should follow the order in `calculate_SO`.

    Args:
        d (int): number of modes.
    """
    len_modes = d * (d - 1) // 2
    d_over_2 = d // 2
    len_modes_2_col = 2 * d_over_2 - 1

    modes = np.empty(shape=(len_modes, 2), dtype=int)

    def inner_loop(modes, idx):
        idx_mod = idx % len_modes_2_col
        mode = jax.lax.cond(
            idx_mod < d_over_2,
            lambda x: 2 * x,
            lambda x: 2 * (x - d_over_2) + 1,
            idx_mod,
        )

        index = jnp.array([mode, mode + 1], dtype=int)

        modes = modes.at[idx].set(index)

        return modes, None

    @jax.checkpoint
    def outer_loop(modes, col):
        xs = jnp.arange(0, d - 1) + col * len_modes_2_col

        modes = jax.lax.scan(inner_loop, modes, xs)[0]

        return modes, None

    modes = jax.lax.scan(outer_loop, modes, jnp.arange(0, d_over_2))[0]

    return modes
