import numpy as np
cimport cython

ctypedef fused integer:
    int
    long
    long long

cdef __find_first(integer item, integer[:] ar):
    """Find the first occurance of item in ar, returns the index"""
    cdef Py_ssize_t size = ar.shape[0]
    for idx in range(size):
        if item == ar[idx]:
            return idx
    return -1

def search_ar_int(integer[:] a, integer[:] b):
    """Find the index of all a elements in b"""
    if integer is int:
        dtype = np.intc
    elif integer is cython.long:
        dtype = np.long
    elif integer is cython.longlong:
        dtype = np.longlong
    result = np.zeros(a.shape[0], dtype=dtype)
    cdef:
        integer[:] result_view = result
        Py_ssize_t size = a.shape[0]
        integer value_a
    for idx_a in range(size):
        value_a = a[idx_a]
        result_view[idx_a] = __find_first(value_a, b)
    return result
