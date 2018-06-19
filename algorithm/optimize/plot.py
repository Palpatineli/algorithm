from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult
# noinspection PyPackageRequirements
from matplotlib import pyplot as plt
# noinspection PyPackageRequirements
import seaborn as sns


def plot_curve(fit_func: Callable[[np.ndarray, np.ndarray], OptimizeResult], y: np.ndarray,
               x: np.ndarray) -> OptimizeResult:
    sns.set()
    fit = fit_func(y, x)
    sim_x = np.linspace(x[0], x[-1], 100)
    sim_y = fit.evaluate(sim_x)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.plot(sim_x, sim_y, color="orange")
    ax.errorbar(x, np.nanmean(y, axis=1), np.nanstd(y, axis=1) / np.sqrt(y.shape[1]))
    return fit
