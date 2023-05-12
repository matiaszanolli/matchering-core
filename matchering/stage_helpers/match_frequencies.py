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
import jax
import jax.numpy as jnp
from scipy import interpolate
from jax.scipy import signal
from time import time

from ..log import debug
from .. import Config
from ..dsp import ms_to_lr, smooth_lowess


def __average_fft(
    loudest_pieces: np.ndarray, sample_rate: int, fft_size: int
) -> tuple:
    *_, specs = signal.stft(
        loudest_pieces,
        sample_rate,
        window="boxcar",
        nperseg=fft_size,
        noverlap=0,
        boundary=None,
        padded=False,
    )
    return jnp.abs(specs).mean((0, 2))


def __smooth_exponentially(matching_fft: np.ndarray, config: Config) -> np.ndarray:
    grid_linear = (
        config.internal_sample_rate * 0.5 * jnp.linspace(0, 1, config.fft_size // 2 + 1)
    )

    grid_logarithmic = (
        config.internal_sample_rate
        * 0.5
        * jnp.logspace(
            jnp.log10(4 / config.fft_size),
            0,
            (config.fft_size // 2) * config.lin_log_oversampling + 1,
        )
    )

    interpolator = interpolate.interp1d(grid_linear, matching_fft, "cubic")
    matching_fft_log = interpolator(grid_logarithmic)

    matching_fft_log_filtered = smooth_lowess(
        matching_fft_log, config.lowess_frac, config.lowess_it, config.lowess_delta
    )

    interpolator = interpolate.interp1d(
        grid_logarithmic, matching_fft_log_filtered, "cubic", fill_value="extrapolate"
    )
    matching_fft_filtered = interpolator(grid_linear)

    matching_fft_filtered = jnp.where(matching_fft_filtered < 0, 0, matching_fft_filtered)

    matching_fft_filtered = jnp.where(
        jnp.isnan(matching_fft_filtered), matching_fft_filtered[0], matching_fft_filtered
    )

    matching_fft_filtered = jnp.where(
        jnp.isinf(matching_fft_filtered), jnp.zeros_like(matching_fft_filtered), matching_fft_filtered
    )

    matching_fft_filtered = jnp.array(matching_fft_filtered)

    return matching_fft_filtered


def __hann_window(M):
    """
    Compute a Hann window of length M.
    Args:
    M (int): Length of the window.
    Returns:
    jnp.ndarray: The Hann window.
    """
    if M < 1:
        return jnp.array([])
    if M == 1:
        return jnp.ones(1)
    n = jnp.arange(0, M)
    return 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * n / (M - 1))


def get_fir(
    target_loudest_pieces: np.ndarray,
    reference_loudest_pieces: np.ndarray,
    name: str,
    config: Config,
) -> np.ndarray:
    debug(f"Calculating the {name} FIR for the matching EQ...")

    target_average_fft = __average_fft(
        target_loudest_pieces, config.internal_sample_rate, config.fft_size
    )
    reference_average_fft = __average_fft(
        reference_loudest_pieces, config.internal_sample_rate, config.fft_size
    )

    target_average_fft = np.maximum(config.min_value, target_average_fft)
    matching_fft = reference_average_fft / target_average_fft

    matching_fft_filtered = __smooth_exponentially(matching_fft, config)

    fir = np.fft.irfft(matching_fft_filtered)
    fir = np.fft.ifftshift(fir) * __hann_window(len(fir))

    return fir


@jax.jit
def convolve(
    target_mid: np.ndarray,
    mid_fir: np.ndarray,
    target_side: np.ndarray,
    side_fir: np.ndarray,
) -> (np.ndarray, np.ndarray):
    debug("Convolving the TARGET audio with calculated FIRs...")
    timer = time()
    result_mid = signal.convolve(target_mid, mid_fir, "same")
    result_side = signal.convolve(target_side, side_fir, "same")
    debug(f"The convolution is done in {time() - timer:.2f} seconds")

    debug("Converting MS to LR...")
    result = ms_to_lr(result_mid, result_side)

    return result, result_mid
