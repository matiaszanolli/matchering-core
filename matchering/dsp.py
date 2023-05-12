# -*- coding: utf-8 -*-

"""
Matchering - Audio Matching and Mastering Python Library
Copyright (C) 2016-2022 Sergree

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import jax
import jax.numpy as jnp
import numpy as np
import statsmodels.api as sm

@jax.jit
def size(array):
    return array.shape[0]

@jax.jit
def channel_count(array):
    return array.shape[1]

@jax.jit
def is_mono(array):
    return array.shape[1] == 1

@jax.jit
def is_stereo(array):
    return array.shape[1] == 2

@jax.jit
def is_1d(array):
    return len(array.shape) == 1

@jax.jit
def mono_to_stereo(array):
    return jnp.repeat(array, repeats=2, axis=1)

@jax.jit
def count_max_peaks(array):
    max_value = jnp.abs(array).max()
    max_count = jnp.count_nonzero(
        jnp.logical_or(jnp.isclose(array, max_value), jnp.isclose(array, -max_value))
    )
    return max_value, max_count

@jax.jit
def lr_to_ms(array):
    array = jnp.copy(array)
    array = jnp.add(array[:, 0], array[:, 1])
    array = jnp.multiply(array, 0.5)
    mid = jnp.copy(array)
    array = jnp.subtract(array[:, jnp.newaxis], array)
    side = jnp.copy(array[:, 0])
    return mid, side

@jax.jit
def ms_to_lr(mid_array, side_array):
    return jnp.vstack((mid_array + side_array, mid_array - side_array)).T

def unfold(array, piece_size, divisions):
    # (len(array),) -> (divisions, piece_size)
    return array[: piece_size * divisions].reshape(-1, piece_size)

def rms(array):
    return jnp.sqrt(array @ array / array.shape[0])

@jax.jit
def batch_rms(array):
    piece_size = array.shape[1]
    # (divisions, piece_size) -> (divisions, 1, piece_size)
    multiplicand = array[:, None, :]
    # (divisions, piece_size) -> (divisions, piece_size, 1)
    multiplier = array[..., None]
    return jnp.sqrt(jnp.squeeze(multiplicand @ multiplier, axis=(1, 2)) / piece_size)

@jax.jit
def amplify(array, gain):
    return array * gain

def normalize(array, threshold, epsilon, normalize_clipped):
    coefficient = 1.0
    max_value = jnp.abs(array).max()
    if max_value < threshold or normalize_clipped:
        coefficient = np.amax(([epsilon, max_value / threshold]))
    return array / coefficient, coefficient

def smooth_lowess(array, frac, it, delta):
    return sm.nonparametric.lowess(
        array, jnp.linspace(0, 1, len(array)), frac=frac, it=it, delta=delta
    )[:, 1]

@jax.jit
def clip(array, to = 1):
    return jnp.clip(array, -to, to)

@jax.jit
def flip(array):
    return 1.0 - array

@jax.jit
def rectify(array, threshold):
    rectified = jnp.abs(array).max(1)
    indices = jnp.argwhere(rectified <= threshold)
    rectified.at(indices).set(threshold)
    rectified /= threshold
    return rectified

@jax.jit
def max_mix(*args):
    return jnp.maximum.reduce(args)

@jax.jit
def strided_app_2d(matrix, batch_size, step):
    matrix_length = matrix.shape[0]
    matrix_width = matrix.shape[1]
    if batch_size > matrix_length:
        return jnp.expand_dims(matrix, axis=0)
    batch_count = ((matrix_length - batch_size) // step) + 1
    stride_length, stride_width = matrix.strides
    return np.lib.stride_tricks.as_strided(
        matrix,
        shape=(batch_count, batch_size, matrix_width),
        strides=(step * stride_length, stride_length, stride_width),
    )

@jax.jit
def batch_rms_2d(array):
    piece_size = array.shape[1] * array.shape[2]
    multiplicand = array.reshape(array.shape[0], 1, piece_size)
    multiplier = array.reshape(array.shape[0], piece_size, 1)
    return jnp.sqrt(jnp.squeeze(multiplicand @ multiplier, axis=(1, 2)) / piece_size)

@jax.jit
def fade(array, fade_size):
    array = jnp.copy(array)
    fade_in = jnp.linspace(0, 1, fade_size)
    fade_out = fade_in[::-1]
    array = jnp.transpose(array)
    array = array.at[:fade_size].mul(fade_in)
    array = array.at[len(array) - fade_size:].mul(fade_out)
    return jnp.transpose(array)
