"""Some filter convenience functions."""
from typing import Sequence

import numpy as np
from scipy.signal import gaussian, fftconvolve


def gaussian_2d(matrix_size, std):
    if not isinstance(matrix_size, Sequence):
        gaussian_1d = gaussian(matrix_size, std)
        kernel = np.outer(gaussian_1d, gaussian_1d)
    else:
        x, y = (gaussian(matrix_size[i],
                         (std[i] if isinstance(std, Sequence) else std))
                for i in (1, 2))
        kernel = np.outer(x, y)
    return kernel / kernel.sum()


def goa(x: np.ndarray, gaussian_std=2, ave_size=-1):
    if ave_size < 0:
        ave_size = gaussian_std * 10 + 1
    kernel = gaussian_2d(ave_size, gaussian_std)
    kernel -= np.ones((ave_size, ave_size)) / (ave_size * ave_size)
    return fftconvolve(x, kernel, 'same')


def dog(x: np.ndarray, inner_size=2, outer_size=50):
    assert outer_size > inner_size
    kernel_size = outer_size * 10 + 1
    kernel = gaussian_2d(kernel_size, inner_size) - gaussian_2d(
        kernel_size, outer_size)
    return fftconvolve(x, kernel, 'same')


def apply_gaussian(x: np.ndarray, std=5) -> np.ndarray:
    kernel_size = std * 10 + 1
    kernel = gaussian(kernel_size, std)
    kernel /= kernel.sum()
    return fftconvolve(x, kernel, 'same')


def apply_gaussian_2d(x: np.ndarray, std=5):
    kernel = gaussian_2d(std * 10 + 1, std)
    return fftconvolve(x, kernel, 'same')

