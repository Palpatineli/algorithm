from typing import Tuple
from math import log, sqrt
import numpy as np
from numpy.linalg import pinv
from scipy.stats import t
from scipy.optimize import minimize, OptimizeResult
from numba import jit
from ..optimize import log_likelihood


@jit(nopython=True, nogil=True, cache=True)
def dog(params, x):
    k_c, k_s, w_c, w_s, f_0 = params
    x_2 = -x ** 2
    return k_c * np.exp(x_2 / w_c ** 2) - k_s * np.exp(x_2 / w_s ** 2) + f_0


@jit(nopython=True, nogil=True, cache=True)
def dog_ml(params, x, y, sd):
    y_hat = dog(params, x)
    return log_likelihood(y_hat, y, sd)


def dog_fit(response, diameter) -> OptimizeResult:
    """Follows Adesnik 2012, fit size tuning curve to difference of gaussian. Note the integration in paper is wrong:
        size differential is 2r * exp(-r^2/w^2) because change in circumference
        meaning that firing rate itself is a DOG
    Args:
        response: row is repeats, col is radius
        diameter: radius values
    Returns:
        k_c: amplitude of excitation
        k_s: amplitude of suppression
        w_c: size of excitatory center
        w_s: size of inhibitory surround
        f_0: baseline activity
    """
    y = np.nanmean(response, axis=1)
    sd = np.nanstd(response, axis=1) / np.sqrt(response.shape[1])
    max_loc = diameter[y.argmax()]
    y_range = y.max() - y.min()
    initial_values = np.array([y_range * 2, y_range * 2 - y[0], max_loc * 1.5, max_loc * 1.0, y.min()])
    bounds = [(0, None), (0, None), (None, None), (None, None), (None, None)]
    fit = minimize(dog_ml, initial_values, args=(diameter, y, sd), method='SLSQP', bounds=bounds)
    fit.evaluate = lambda a: dog(fit.x, a)
    fit.parameter_names = ['k_c', 'k_s', 'w_c', 'w_s', 'f_0']
    fit.df = response.shape[1] - len(fit.x)
    return fit


@jit(nopython=True, nogil=True, cache=True)
def max_dog(fit) -> float:
    """calculate the maximum of fit curve"""
    k_c, k_s, w_c, w_s, _ = fit
    if k_s == 0:
        return 0
    if k_c == 0:
        return np.inf
    return sqrt(log((w_s / w_c) ** 2 * k_c / k_s) / (1 / w_c ** 2 - 1 / w_s ** 2))


@jit(nopython=True, nogil=True, cache=True)
def h_ratio(params, contrast):
    r_max, c_50, n = params
    return r_max / (1.0 + (c_50 / contrast) ** n)


@jit(nopython=True, nogil=True, cache=True)
def h_ratio_ml(params, contrast, y, sd):
    y_hat = h_ratio(params, contrast)
    return log_likelihood(y_hat, y, sd)


def h_ratio_fit(response, contrast) -> Tuple[int, OptimizeResult]:
    if isinstance(response[0], np.ndarray):
        y, sd = response
    else:
        y = np.nanmean(response, axis=0)
        sd = np.nanstd(response, axis=0)
    initial_values = np.array([2.0 * y[-1], 0.25, 4.0])
    bounds = ((0, None), (0, 1), (1.0, 10.0))
    fit = minimize(h_ratio_ml, initial_values, args=(contrast, y, sd), method="SLSQP", bounds=bounds)
    fit.evaluate = lambda a: h_ratio(fit.x, a)
    fit.df = len(y) - len(initial_values)
    jac = fit.jac.reshape(1, -1)
    fit.var = np.diag(pinv(jac.T.dot(jac)))
    fit.parameter_names = ['r_max', 'c_50', 'n']
    fit.compare = lambda x: (1 - t.cdf(np.abs(fit.x - x.x) / np.sqrt(fit.var / fit.df + x.var / x.df),
                                       fit.df + x.df)) * 2
    return fit
