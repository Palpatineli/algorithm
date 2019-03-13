import numpy as np
from ._utils import take_segment

__all__ = ['take_segment', 'splice', 'bool2index', 'rolling_sum']

def bool2index(series: np.ndarray) -> np.ndarray:
    starts = np.flatnonzero(np.logical_and(series[1:], np.logical_not(series[: -1]))) + 1
    ends = np.flatnonzero(np.logical_and(series[: -1], np.logical_not(series[1:]))) + 1
    if series[0]:
        starts = np.insert(starts, 0, 0)
    if series[-1]:
        ends = np.append(ends, series.shape[0])
    return np.vstack([starts, ends]).T
    # return _bool2index(series.astype(np.uint8))

def splice(series: np.ndarray, time_points: np.ndarray, axis: int = 0) -> np.ndarray:
    """The other way to take segments. Take variable length segments and splice together.
    Args:
        series: a N-d array with one axis ready to be cut
        time_points: 2-d array, each row is (start, end) in sample points
        axis: the axis to be cut in series
    Returns:
        a N-d array similar to series, with samples outside the time_points cut out
    """
    return np.concatenate(tuple(np.take(series, range(start, end), axis)
                                for start, end in time_points if start < series.shape[axis]), axis=axis)

def rolling_sum(x: np.ndarray, n: int) -> np.ndarray:
    if x.ndim == 1:
        result = x.cumsum(axis=0, dtype=float)
        result[n:] -= result[:-n]
        return result[n - 1:]
    elif x.ndim == 2:
        result = x.cumsum(axis=1, dtype=float)
        result[:, n:] -= result[:, :-n]
        return result[:, n - 1:]
    elif x.ndim == 3:
        result = x.cumsum(axis=2, dtype=float)
        result[:, :, n:] -= result[:, :, :-n]
        return result[:, :, n - 1:]
    else:
        raise NotImplementedError
