import numpy as np
from numba import jit

@jit(nopython=True, nogil=True, cache=True)
def take_segment(trace: np.ndarray, events: np.ndarray, length: int) -> np.ndarray:
    result = np.empty((len(events), length), dtype=trace.dtype)
    for idx, (start, end) in enumerate(zip(events, events + length)):
        result[idx] = trace[start:end]
    return result
