import numpy as np
cimport cython
cimport numpy as np

ctypedef fused integer:
    int
    long
    long long
    float
    double

def first_gt(integer[:] a, integer b):
    """Find the index of first item of a that is greater than b"""
    cdef Py_ssize_t size = a.shape[0]
    for idx in range(size):
        if a[idx] > b:
            return idx
    return -1

def first_ge(integer[:] a, integer b):
    """Find the index of first item of a that is greater than b"""
    cdef Py_ssize_t size = a.shape[0]
    for idx in range(size):
        if a[idx] >= b:
            return idx
    return -1

def first_lt(integer[:] a, integer b):
    """Find the index of first item of a that is less than b"""
    cdef Py_ssize_t size = a.shape[0]
    for idx in range(size):
        if a[idx] < b:
            return idx
    return -1

def first_le(integer[:] a, integer b):
    """Find the index of first item of a that is less than b"""
    cdef Py_ssize_t size = a.shape[0]
    for idx in range(size):
        if a[idx] <= b:
            return idx
    return -1
