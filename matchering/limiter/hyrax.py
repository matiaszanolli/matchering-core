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

import numpy as np
import jax.numpy as jnp
import math
from jax import jit, vmap, lax, Array
from jax.lax import scan

from .. import Config
from ..log import debug
from ..dsp import rectify, flip, max_mix
from ..utils import make_odd, ms_to_samples

def __sliding_window_fast(
    array: Array, window_size: int, mode: str = "attack"
) -> Array:
    window_size = window_size if window_size % 2 != 0 else window_size+1
    radius = window_size//2
    pool = lax.max
    if mode == "hold":
        pool = lax.min
    elif mode == "attack":
        pool = lax.max
    sum_backward = jnp.cumsum(array[::-1], 0)[::-1]
    sum_forward = jnp.cumsum(array, 0)
    sum_ab = sum_backward + sum_forward
    slices = sum_ab[radius*2:]
    slices = slices - slices[:-radius*2]
    shape = (radius*2+1,)+array.shape
    matrix = jnp.ones_like(array)
    matrix = jnp.cumsum(matrix, 0)
    matrix_back = matrix[::-1] - 1
    matrix_back = matrix_back[radius*2:]
    matrix_back = matrix_back - matrix_back[:-radius*2]
    matrix = jnp.concatenate([jnp.zeros_like(matrix[:radius*2]), matrix])
    matrix = matrix[:-radius*2]
    res = pool(jnp.nan_to_num(slices/((matrix+1)**2-matrix_back**2)),0)
    return res


def __lfilter(b, a, x):
    """
    Filter data along one dimension using a digital filter.
    Args:
        b (array_like): The numerator coefficient vector in a 1-D sequence.
        a (array_like): The denominator coefficient vector in a 1-D sequence.
        x (array_like): An N-dimensional input array.
    Returns:
        jnp.ndarray: The output of the digital filter.
    """
    b = jnp.asarray(b)
    a = jnp.asarray(a)
    x = jnp.asarray(x)
    
    assert len(a) > 0, "The denominator coefficients are empty."
    assert len(b) > 0, "The numerator coefficients are empty."
    assert a[0] != 0, "The first denominator coefficient must not be zero."
    
    # Normalize coefficients
    a = a / a[0]
    b = b / a[0]
    a = a[1:]

    def loop_carry(carry, x_t):
        y_t, zf = carry
        y_t = jnp.dot(b, jnp.concatenate(([x_t], zf[:-1])))
        zf = jnp.roll(zf, shift=-1)
        zf = jnp.concatenate((zf[:-1], [0]))
        zf = zf - a * y_t
        return (y_t, zf), y_t

    _, y = scan(loop_carry, (0, jnp.zeros(len(a))), x)
    return y

@jit
def __filtfilt(b, a, x):
    """
    Apply a linear digital filter twice, once forward and once backward.
    Args:
        b (array_like): The numerator coefficient vector in a 1-D sequence.
        a (array_like): The denominator coefficient vector in a 1-D sequence.
        x (array_like): An N-dimensional input array.
    Returns:
        jnp.ndarray: The output of the digital filter.
    """
    # Forward filtering
    y_forward = __lfilter(b, a, x)
    
    # Reverse the filtered signal
    y_forward_reversed = jnp.flip(y_forward, 0)
    
    # Backward filtering
    y_backward = __lfilter(b, a, y_forward_reversed)
    
    # Reverse the backward-filtered signal
    y_filtfilt = jnp.flip(y_backward, 0)
    
    return y_filtfilt


def __butter(N, Wn, btype='low', analog=False, output='ba'):
    """
    Butterworth digital and analog filter design.
    Args:
        N (int): The order of the filter.
        Wn (array_like): A scalar or length-2 sequence giving the critical frequencies.
        btype (str, optional): The type of filter ('low', 'high', 'band', or 'stop'). Default is 'low'.
        analog (bool, optional): When True, return an analog filter. Default is False.
        output (str, optional): The type of output: 'ba' for numerator/denominator coefficients, 'zpk' for zeros, poles, and gain.
    Returns:
        b, a: ndarray, ndarray; Numerator (b) and denominator (a) polynomials of the IIR filter.
        or
        z, p, k: ndarray, ndarray, float; Zeros, poles, and system gain of the IIR filter.
    """
    if btype not in ['low', 'high', 'band', 'stop']:
        raise ValueError("Invalid filter type. Must be 'low', 'high', 'band', or 'stop'.")
    
    if output not in ['ba', 'zpk']:
        raise ValueError("Invalid output type. Must be 'ba' or 'zpk'.")

    Wn = jnp.atleast_1d(Wn)

    # Compute the filter coefficients
    if not analog:
        fs = 2.0
        Wn = 2 * fs * jnp.tan(jnp.pi * Wn / fs)

    z = jnp.array([])
    k = 1

    if btype in ['low', 'high']:
        if N % 2 == 0:
            z = jnp.r_[-1j, 1j] * z
        p = jnp.exp(1j * (2 * jnp.arange(N) + N - 1) * jnp.pi / (2 * N))
        if btype == 'high':
            z = -z
            p = -p
            k = -k
    else:
        Wn = jnp.asarray(Wn)
        if len(Wn) != 2:
            raise ValueError("For band and stop filters, Wn must be length-2 sequence.")
        
        Wn = jnp.sort(Wn)
        dw = Wn[1] - Wn[0]
        wo = jnp.sqrt(Wn.prod())

        p = jnp.exp(1j * (2 * jnp.arange(N) + N - 1) * jnp.pi / (2 * N))
        p = jnp.concatenate((p * wo / dw, p * wo * dw))

        if btype == 'stop':
            k = -k
        else:
            z = jnp.concatenate((z, -z))
    
    if not analog:
        z = (1 + z) / (1 - z)
        p = (1 + p) / (1 - p)
        k = k * jnp.prod(1 - p) / jnp.prod(1 - z)

    if output == 'zpk':
        return z, p, k
    else:
        num = jnp.poly(z)
        den = jnp.poly(p)
        num = num / den[0] * k
        return num, den


def __process_attack(array: Array, config: Config) -> (Array, Array):
    attack = ms_to_samples(config.limiter.attack, config.internal_sample_rate)

    slided_input = __sliding_window_fast(array, attack, mode="attack")
    
    coef = math.exp(config.limiter.attack_filter_coefficient / attack)
    b = jnp.array([1 - coef])
    a = jnp.array([1, -coef])
    n = jnp.max([a.shape[0],b.shape[0]])
    output = __filtfilt(b, a, slided_input)
    return output, slided_input


def __process_release(array: Array, config: Config) -> Array:
    hold = ms_to_samples(config.limiter.hold, config.internal_sample_rate)

    slided_input = __sliding_window_fast(array, hold, mode="hold")

    b, a = __butter(
        config.limiter.hold_filter_order,
        config.limiter.hold_filter_coefficient,
        output='ba'
    )

    hold_output = __lfilter(b, a, slided_input)
    hold_output = jnp.squeeze(hold_output)

    b, a = __butter(
        config.limiter.release_filter_order,
        config.limiter.release_filter_coefficient / config.limiter.release,
        output='ba'
    )

    release_output = __lfilter(b, a, np.maximum(slided_input, hold_output))

    return jnp.maximum(hold_output, release_output)


def limit(array: Array, config: Config) -> Array:
    debug("The limiter is started. Preparing the gain envelope...")
    rectified = rectify(array, config.threshold)

    if jnp.all(jnp.isclose(rectified, 1.0)):
        debug("The limiter is not needed!")
        return array

    gain_hard_clip = flip(1.0 / rectified)
    debug("Modifying the gain envelope: attack stage...")
    gain_attack, gain_hard_clip_slided = __process_attack(
        jnp.copy(gain_hard_clip), config
    )

    debug("Modifying the gain envelope: hold / release stage...")
    gain_release = __process_release(jnp.copy(gain_hard_clip_slided), config)

    debug("Finalizing the gain envelope...")
    gain = flip(max_mix(gain_hard_clip, gain_attack, gain_release))

    return array * gain[:, None]
