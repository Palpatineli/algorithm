import numpy as np
from algorithm.time_series.utils import bool2index, splice, rolling_sum

def test_bool2index():
    a = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1], dtype=bool)
    correct_a = np.array([[0, 1],
                          [2, 5],
                          [7, 8],
                          [10, 11]])
    assert(np.array_equal(bool2index(a), correct_a))
    b = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    correct_b = np.array([[2, 5],
                          [7, 8]])
    assert(np.array_equal(bool2index(b), correct_b))

def test_splice():
    # 1-d
    assert(np.array_equal(splice(np.arange(5), np.array([[0, 1], [3, 5]])), np.array([0, 3, 4])))
    assert(np.array_equal(splice(np.arange(5), np.array([[0, 1], [3, 3]])), np.array([0])))
    # 2-d
    assert(np.array_equal(splice(np.arange(6).reshape(3, 2), np.array([[0, 1], [0, 2]])),
                          np.array([[0, 1], [0, 1], [2, 3]])))
    assert(np.array_equal(splice(np.arange(6).reshape(3, 2), np.array([[0, 1], [2, 3]])),
                          np.array([[0, 1], [4, 5]])))
    assert(np.array_equal(splice(np.arange(6).reshape(2, 3), np.array([[0, 1], [2, 3]]), 1),
                          np.array([[0, 2], [3, 5]])))
    # 3-d
    assert(np.array_equal(splice(np.arange(12).reshape(3, 2, 2), np.array([[0, 1], [2, 3]])),
                          np.array([[[0, 1], [2, 3]], [[8, 9], [10, 11]]])))
    assert(np.array_equal(splice(np.arange(12).reshape(2, 2, 3), np.array([[0, 1], [0, 2]]), 2),
                          np.array([[[0, 0, 1], [3, 3, 4]], [[6, 6, 7], [9, 9, 10]]])))

def test_rolling():
    # 1-d
    assert(np.array_equal(rolling_sum(np.arange(5), 3), np.array([3, 6, 9])))
    assert(np.array_equal(rolling_sum(np.arange(5), 4), np.array([6, 10])))
    # 2-d
    assert(np.array_equal(rolling_sum(np.arange(6).reshape(2, 3), 2), np.array([[1, 3], [7, 9]])))
    assert(np.array_equal(rolling_sum(np.arange(8).reshape(2, 4), 3), np.array([[3, 6], [15, 18]])))
    # 3-d
    assert(np.array_equal(rolling_sum(np.arange(16).reshape(2, 2, 4), 3), np.array([[[3, 6], [15, 18]],
                                                                                    [[27, 30], [39, 42]]])))
