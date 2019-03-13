import numpy as np

from algorithm.utils import quantize, one_hot, first_gt, first_ge, first_lt, first_le

def test_quantize():
    source = np.array([1, 2, 2, 3, 4, 5])
    assert(np.array_equal(quantize(source, 1), np.array([0, 1, 1, 0, 0, 0])))
    assert(np.array_equal(quantize(source, 2), np.array([2, 1, 1, 0, 0, 0])))
    source = np.array([1, 2, 2, 3, 4, 5, 3, 2, 4])
    assert(np.array_equal(quantize(source, 2), np.array([0, 1, 1, 2, 0, 0, 2, 1, 0])))
    source2 = np.array([[1, 2, 2, 3, 4], [5, 3, 2, 4, 6]])
    correct2 = np.array([[0, 1, 1, 2, 0], [0, 2, 1, 0, 0]])
    assert(np.array_equal(quantize(source2, 2), correct2))

def test_one_hot():
    source1 = np.array([0, 1, 1, 0, 0, 0])
    correct1 = np.array([[1, 0, 0, 1, 1, 1], [0, 1, 1, 0, 0, 0]], dtype=np.bool)
    assert(np.array_equal(one_hot(source1), correct1))
    source2 = np.array([0, 1, 3, 2, 1, 4])
    correct2 = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1]], dtype=np.bool)
    assert(np.array_equal(one_hot(source2), correct2))
    source3 = np.array([[0, 1, 3], [2, 1, 4]])
    correct3 = np.array([[[1, 0, 0], [0, 0, 0]],
                         [[0, 1, 0], [0, 1, 0]],
                         [[0, 0, 0], [1, 0, 0]],
                         [[0, 0, 1], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, 1]]], dtype=np.bool)
    assert(np.array_equal(one_hot(source3), correct3))

def test_first_gt():
    assert(first_gt(np.arange(100), 15) == 16)
    assert(first_gt(np.arange(100), 110) == -1)
    assert(first_gt(np.arange(-100, 100), 10) == 111)

def test_first_ge():
    assert(first_ge(np.arange(100), 15) == 15)
    assert(first_ge(np.arange(100), 110) == -1)
    assert(first_ge(np.arange(-100, 100), 10) == 110)

def test_first_lt():
    assert(first_lt(np.arange(100, 0, -1), 15) == 86)
    assert(first_lt(np.arange(100, 0, -1), -10) == -1)
    assert(first_lt(np.arange(100, -100, -1), -10) == 111)

def test_first_le():
    assert(first_le(np.arange(100, 0, -1), 15) == 85)
    assert(first_le(np.arange(100, 0, -1), -10) == -1)
    assert(first_le(np.arange(100, -100, -1), -10) == 110)
