from math import sqrt, pi

import numpy as np
from numba import jit
from .piece_lm import PieceLinear2

__all__ = ["PieceLinear2", "log_likelihood"]


@jit(nopython=True, nogil=True, cache=True)
def log_likelihood(y_hat, y, sd):
    return -np.log(np.exp(-((y - y_hat) / sd) ** 2 / 2) / sqrt(2 * pi) / sd).sum()
