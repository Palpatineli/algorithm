from typing import Sequence
import numpy as np
from scipy.stats import norm, rv_continuous
try:
    from .alt_plot import save_chart
except ImportError:
    def save_chart(chart, file_path):  # type: ignore
        raise NotImplementedError("install altair!")
from .first_x import first_gt, first_ge, first_lt, first_le
from .main import iter_tree, map_tree, map_tree_parallel, map_table, zip_tree, flatten, unflatten, rotate45
from .move import shift

__all__ = ['save_chart', 'normalize', 'first_gt', 'first_ge', 'first_lt', 'first_le',
           'iter_tree', 'map_tree', 'map_tree_parallel', 'map_table', 'zip_tree', 'flatten', 'unflatten',
           'quantize', 'one_hot', 'is_list', 'shift', 'rotate45']

def normalize(a: np.ndarray, dist: rv_continuous = norm, **kwargs) -> np.ndarray:
    """Assumes a is 1d."""
    indices, a = np.argsort(a), a.copy().astype(np.float)
    disc_dist = dist.ppf(np.linspace(0, 1, len(a) + 2, endpoint=True), **kwargs)
    a[indices] = disc_dist[1: -1]
    return a

def quantize(labels: np.ndarray, groups: int = 1) -> np.ndarray:
    """Given n clusters, and an array of cluster labels, isolate the largest {groups} clusters,
    and group the rest together. May return less than groups number of clusters if that's all there is.
    Args:
        clusters: either [1] 2D array [2 x sample_no] int array, 1st row = sample_id, 2nd row = sample_group;
            else [2] 1D array where index is kept and number is label
        groups: how many large clusters to be isolated
    Returns:
        1D array of int in order of sample_id, number ordered by group size, 0 reserved for combined small groups.
    """
    if labels.ndim == 2 and labels.shape[0] == 2:
        return _quantize2d(labels, groups)
    elif labels.ndim == 1 or labels.shape.max == labels.size:
        return _quantize1d(labels, groups)
    else:
        raise ValueError("you need either 1D or 2D label array")

def _quantize2d(labels: np.ndarray, groups: int = 1) -> np.ndarray:
    label, count = np.unique(labels[1], return_counts=True)
    largests = label[np.argsort(-count)[0: groups]]
    result = np.zeros_like(labels[1])
    for idx, large in enumerate(largests):
        indices = labels[1] == large
        result[indices] = idx + 1
    return result[labels[0].argsort()]

def _quantize1d(labels: np.ndarray, groups: int = 1) -> np.ndarray:
    label, count = np.unique(labels, return_counts=True)
    largests = label[np.argsort(-count)[0: groups]]
    result = np.zeros_like(labels)
    for idx, large in enumerate(largests):
        indices = labels == large
        result[indices] = idx + 1
    return result

def one_hot(array: np.ndarray) -> np.ndarray:
    """see tf.one_hot, convert 1d label array to 2d bool array where each row is a group"""
    return np.stack([array == number for number in np.unique(array)])

Sequence.register(np.ndarray)

def is_list(x) -> bool:
    return isinstance(x, Sequence) and not isinstance(x, str)
