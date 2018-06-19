import numpy as np
from scipy.stats import norm, rv_continuous
from .alt_plot import save_chart

__all__ = ['save_chart', 'normalize']

def normalize(a: np.ndarray, dist: rv_continuous = norm, **kwargs) -> np.ndarray:
    """Assumes a is 1d."""
    indices, a = np.argsort(a), a.copy().astype(np.float)
    disc_dist = dist.ppf(np.linspace(0, 1, len(a) + 2, endpoint=True), **kwargs)
    a[indices] = disc_dist[1: -1]
    return a
