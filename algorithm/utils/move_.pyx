import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def shift_c(double[:, :] arr, int x, int y, double fill_value):
    cdef Py_ssize_t y_max = arr.shape[0]
    cdef Py_ssize_t x_max = arr.shape[1]

    result = np.empty((y_max, x_max), dtype=np.double)
    cdef double[:, :] result_view = result
    cdef Py_ssize_t x_idx, y_idx

    if y >= 0:
        if x >= 0:
            for y_idx in range(y):
                for x_idx in range(x_max):
                    result_view[y_idx, x_idx] = fill_value
            for y_idx in range(y, y_max):
                for x_idx in range(x):
                    result_view[y_idx, x_idx] = fill_value
            for y_idx in range(y_max - y):
                for x_idx in range(x_max - x):
                    result_view[y_idx + y, x_idx + x] = arr[y_idx, x_idx]
        else:
            for y_idx in range(y):
                for x_idx in range(x_max):
                    result_view[y_idx, x_idx] = fill_value
            for y_idx in range(y_max - y):
                for x_idx in range(x_max + x):
                    result_view[y_idx + y, x_idx] = arr[y_idx, x_idx - x]
            for y_idx in range(y, y_max):
                for x_idx in range(x_max + x, x_max):
                    result_view[y_idx, x_idx] = fill_value
    else:
        if x >= 0:
            for y_idx in range(y_max + y):
                for x_idx in range(x):
                    result_view[y_idx, x_idx] = fill_value
            for y_idx in range(y_max + y):
                for x_idx in range(x_max - x):
                    result_view[y_idx, x_idx + x] = arr[y_idx - y, x_idx]
            for y_idx in range(y_max + y, y_max):
                for x_idx in range(x_max):
                    result_view[y_idx, x_idx] = fill_value
        else:
            for y_idx in range(y_max + y):
                for x_idx in range(x_max + x):
                    result_view[y_idx, x_idx] = arr[y_idx - y, x_idx - x]
            for y_idx in range(y_max + y):
                for x_idx in range(x_max + x, x_max):
                    result_view[y_idx, x_idx] = fill_value
            for y_idx in range(y_max + y, y_max):
                for x_idx in range(x_max):
                    result_view[y_idx, x_idx] = fill_value
    return result

