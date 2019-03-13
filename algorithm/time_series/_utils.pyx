import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

ctypedef fused numbers:
    int
    double
    long long

ctypedef fused integer:
    int
    long
    long long

def take_segment(numbers[:] trace, integer[:] events, integer length):
    cdef:
        Py_ssize_t x_size = events.shape[0]
        Py_ssize_t y_size = length
        Py_ssize_t total_length = trace.shape[0]
        Py_ssize_t start = 0
        Py_ssize_t end = 0
    if numbers is int:
        dtype = np.intc
    elif numbers is double:
        dtype = np.double
    elif numbers is cython.longlong:
        dtype = np.longlong
    result = np.zeros((x_size, y_size), dtype=dtype)
    cdef numbers[:, ::1] result_view = result
    for idx in range(x_size):
        start = events[idx]
        end = start + length
        if start >= 0 and start < total_length:
            if end <= total_length:
                result_view[idx, :] = trace[start: end]
            else:
                result_view[idx, : total_length - start] = trace[start: total_length]
    return result

cdef inline void append(long item, long **array, Py_ssize_t *size, Py_ssize_t *idx):
    cdef long *new_array
    if (idx[0]) >= (size[0]):
        size[0] = (size[0]) * 2
        new_array = <long*>malloc(size[0] * sizeof(long))
        for idx2 in range((size[0])):
            new_array[idx2] = array[0][idx2]
        free(array[0])
        array = &new_array
    array[0][idx[0]] = item
    idx[0] = idx[0] + 1

ctypedef unsigned char uint8
from numpy cimport ndarray, uint8_t

@cython.boundscheck(False)
@cython.wraparound(False)
def _bool2index(ndarray[uint8_t, cast=True] series):
    """From a 1d boolean array, take indices in a Nx2 array, where 1st column is
    True segment start and second column is True segment end."""
    cdef:
        Py_ssize_t result_size = 128
        Py_ssize_t result_idx = 0
        long *result = <long*>malloc(result_size * sizeof(long))
    cdef:
        uint8[:] series_view = series
        long size = series.shape[0]
        uint8 status = False
        long start = 0
        long idx
    for idx in range(size):
        if series_view[idx]:
            if not status:
                start = idx
                status = True
        else:
            if status:
                append(start, &result, &result_size, &result_idx)
                append(idx, &result, &result_size, &result_idx)
                status = False
    if status:
        append(start, &result, &result_size, &result_idx)
        append(size, &result, &result_size, &result_idx)
    result_array = np.zeros((result_idx // 2, 2), dtype=np.longlong)
    cdef long[:, ::1] result_view = result_array
    for idx in range(result_idx // 2):
        result_view[idx, 0] = result[idx * 2]
        result_view[idx, 1] = result[idx * 2 + 1]
    return result_array
