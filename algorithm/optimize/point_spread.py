import numpy as np
from scipy.optimize import minimize, OptimizeResult
from numba import jit


@jit(nopython=True, nogil=True, cache=True)
def gaussian(params, x, y):
    x_0, y_0, a, rho, c = params
    return a * np.exp(-((x - x_0) ** 2 + (y - y_0) ** 2) / (2 * rho ** 2)) + c


@jit(nopython=True, nogil=True, cache=True)
def gaussian_target(params, x, y, z):
    return np.sum((z - gaussian(params, x, y)) ** 2)


def fit_gaussian(data: np.ndarray) -> OptimizeResult:
    """fit logistic curve given zero and saturation
    Args:
        data: an 2D array with image data
    Return:
        the fit object, where fit.x is the parameters, has extra attribute fit.evaluate
    """
    sorted_data = np.sort(data, None)
    baseline = sorted_data[len(sorted_data) // 2]
    amplitude = sorted_data[int(len(sorted_data) * 0.95)] - baseline
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    data_sum = data.sum()
    x_0 = int(round((x * data).sum() / data_sum))
    y_0 = int(round((y * data).sum() / data_sum))
    rho = data.shape[0] // 4
    initial_values = (x_0, y_0, amplitude, rho, baseline)
    fit = minimize(gaussian_target, initial_values, args=(x, y, data), options={'gtol': 1e-6})
    fit.grid = x, y
    return fit


def test_fit_gaussian():
    np.random.seed(12345)
    x, y = np.meshgrid(np.arange(15), np.arange(15))
    noisy = gaussian((5, 5, 2, 2, 1), x, y) + np.random.randn(15, 15) * 0.1
    fit = fit_gaussian(noisy.copy())
    correct = np.array([5, 5, 2, 2, 1])
    assert(np.all(np.abs(fit.x - correct) < 0.1))
