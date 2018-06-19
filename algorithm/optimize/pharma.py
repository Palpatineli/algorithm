import numpy as np
from scipy.optimize import minimize, OptimizeResult
from numba import jit
from optimize import log_likelihood


@jit(nopython=True, nogil=True, cache=True)
def logistic(params, x):
    x0, l_cap, k, a = params
    return l_cap / (1 + np.exp((x0 - x) * k)) + a


@jit(nopython=True, nogil=True, cache=True)
def logistic_target(params, x, y, sd):
    y_hat = [params[3]]
    y_hat_calc = logistic(params, x)
    for i in range(y_hat_calc.shape[0]):
        y_hat.append(y_hat_calc[i])
    y_hat.append(params[3] + params[1])
    y_hat = np.array(y_hat)
    return log_likelihood(y_hat, y, sd)


def ec50_log(data: np.ndarray, x: np.ndarray) -> OptimizeResult:
    """fit logistic curve given zero and saturation
    Args:
        data: an array with drug response data
            each column is for one dosage, data[:, 0] is zero value and data[:, -1] is saturation value
            nan is allowed if tests numbers are different for each dosage
        x: the dosage that correlates to center rows of data, must be in increasing order
            the params are x0, L, k, a. a is the zero level
    Return:
        the fit object, where fit.x is the parameters, has extra attribute fit.evaluate
    """
    y = np.nanmean(data, axis=0)
    sd = np.nanstd(data, axis=0)
    initial_values = np.array([x.mean(), y.max() - y.min(), 4 / (x.max() - x.min()), y.min()])
    fit = minimize(logistic_target, initial_values, args=(x, y, sd), options={'gtol': 1e-6})
    fit.evaluate = lambda a: logistic(fit.x, a)
    return fit
