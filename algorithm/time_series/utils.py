import numpy as np
from numba import jit

@jit(nopython=True, nogil=True, cache=True)
def take_segment(trace: np.ndarray, events: np.ndarray, length: int) -> np.ndarray:
    result = np.empty((len(events), length), dtype=trace.dtype)
    total_length = trace.shape[0]
    for idx, (start, end) in enumerate(zip(events, events + length)):
        if start < 0:
            continue
        if end <= total_length:
            result[idx, :] = trace[start:end]
    return result

@jit(nopython=True, nogil=True, cache=True)
def bool2index(series: np.ndarray) -> np.ndarray:
    """From a 1d boolean array, take indices in a Nx2 array, where 1st column is
    True segment start and second column is True segment end."""
    series = series.astype(np.bool_)
    start = np.empty(series.shape[0], dtype=np.bool_)
    start[1:] = np.logical_and(series[1:], np.logical_not(series[0: -1]))
    start[0] = series[0]
    end = np.empty(series.shape[0], dtype=np.bool_)
    end[0:-1] = np.logical_and(np.logical_not(series[1:]), series[0: -1])
    end[-1] = series[-1]
    return np.vstack((np.nonzero(start)[0], np.nonzero(end)[0] + 1)).T

def splice(series: np.ndarray, time_points: np.ndarray, axis: int = 0) -> np.ndarray:
    """The other way to take segments. Take variable length segments and splice together.
    Args:
        series: a N-d array with one axis ready to be cut
        time_points: 2-d array, each row is (start, end) in sample points
        axis: the axis to be cut in series
    Returns:
        a N-d array similar to series, with samples outside the time_points cut out
    """
    return np.concatenate(tuple(np.take(series, range(start, end), axis) for start, end in time_points), axis=axis)
