import numpy as np
from scipy.stats import norm, rv_continuous
try:
    from .alt_plot import save_chart
except ImportError:
    def save_chart(chart, file_path):
        raise NotImplementedError("install altair!")
from .first_x import first_gt, first_ge, first_lt, first_le
from .main import iter_tree, map_tree, map_tree_parallel, zip_tree, flatten, unflatten

__all__ = ['save_chart', 'normalize', 'first_gt', 'first_ge', 'first_lt', 'first_le',
           'iter_tree', 'map_tree', 'map_tree_parallel', 'zip_tree', 'flatten', 'unflatten',
           'quantize', 'one_hot']

def normalize(a: np.ndarray, dist: rv_continuous = norm, **kwargs) -> np.ndarray:
    """Assumes a is 1d."""
    indices, a = np.argsort(a), a.copy().astype(np.float)
    disc_dist = dist.ppf(np.linspace(0, 1, len(a) + 2, endpoint=True), **kwargs)
    a[indices] = disc_dist[1: -1]
    return a

def quantize(clusters: np.ndarray, groups: int = 1) -> np.ndarray:
    """given n clusters, and an array of cluster labels, isolate the largest {groups} clusters,
    and group the rest together. May return less than groups number of clusters if that's all there is
    Args:
        clusters: int array of cluster labels
        groups: how many large clusters to be isolated
    Returns:
        same format as clusters, except with fewer groups, and the number ordered by size,
        0 reserved for combined small groups
    """
    number, count = np.unique(clusters, return_counts=True)
    largests = number[np.argsort(-count)[0: groups]]
    result = np.zeros_like(clusters)
    for idx, large in enumerate(largests):
        indices = clusters == large
        result[indices] = idx + 1
    return result

def one_hot(array: np.ndarray) -> np.ndarray:
    """see tf.one_hot, convert 1d label array to 2d bool array where each row is a group"""
    return np.stack([array == number for number in np.unique(array)])
