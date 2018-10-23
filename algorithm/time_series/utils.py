import numpy as np
from ._utils import take_segment, _bool2index

__all__ = ['take_segment', 'splice', 'bool2index']

def bool2index(series: np.ndarray) -> np.ndarray:
    return _bool2index(series.astype(np.uint8))

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
