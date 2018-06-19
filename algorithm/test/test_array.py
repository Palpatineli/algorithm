# pylint:disable=C0111
import numpy as np

from algorithm.array import DataFrame, search_ar


def test_group_mean():
    data = DataFrame(np.arange(81).reshape((3, 3, 3, 3)),
                     [np.arange(3), np.arange(3), np.arange(1, 4), np.arange(2, 5)])
    group1 = [np.array(['a', 'b', 'a']), np.array(['c', 'd', 'c'])]
    correct1 = np.array([[[[12, 13, 14], [12, 13, 14]],
                          [[12, 13, 14], [12, 13, 14]]],
                         [[[39, 40, 41], [39, 40, 41]],
                          [[39, 40, 41], [39, 40, 41]]],
                         [[[66, 67, 68], [66, 67, 68]],
                          [[66, 67, 68], [66, 67, 68]]]])
    axes1 = [np.arange(3), np.array(['a', 'b']), np.array(['c', 'd']), np.arange(2, 5)]
    assert data.group_by(group1, lambda x: np.mean(x, axis=1)) == DataFrame(correct1, axes1)

    group2 = [np.array(['a', 'b', 'a'])]
    correct2 = np.array([[[[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                          [[9, 10, 11], [12, 13, 14], [15, 16, 17]]],
                         [[[36, 37, 38], [39, 40, 41], [42, 43, 44]],
                          [[36, 37, 38], [39, 40, 41], [42, 43, 44]]],
                         [[[63, 64, 65], [66, 67, 68], [69, 70, 71]],
                          [[63, 64, 65], [66, 67, 68], [69, 70, 71]]]])
    axes2 = [np.arange(3), np.array(['a', 'b']), np.arange(1, 4), np.arange(2, 5)]
    result2 = data.group_by(group2, lambda x: np.mean(x, axis=1))
    assert result2 == DataFrame(correct2, axes2)

    correct2_5 = np.array([[[12, 13, 14], [12, 13, 14]],
                           [[39, 40, 41], [39, 40, 41]],
                           [[66, 67, 68], [66, 67, 68]]])
    axes2_5 = [np.arange(3), np.array(['a', 'b']), np.arange(2, 5)]
    result2_5 = data.group_by(group2, lambda x: np.mean(x, axis=(1, 2)))
    assert result2_5 == DataFrame(correct2_5, axes2_5)


def test_search_ar_normal():
    array_int1 = np.array([3, 1, 2, 0, 5, 4])
    array_int2 = np.array([-1, 5, 6, 0, 7, 4, 1, 3, 2])
    result1 = search_ar(array_int1, array_int2)
    correct1 = np.array([7, 6, 8, 3, 1, 5])
    assert(np.array_equal(result1, correct1))
    array_str3 = np.array(['this', 'crap', 'omg', 'b'])
    array_str4 = np.array(['crap', 'b', 'this', 'oops', 'omg'])
    result2 = search_ar(array_str3, array_str4)
    correct2 = np.array([2, 0, 4, 1])
    assert(np.array_equal(result2, correct2))


def test_enumerate():
    array1 = DataFrame(np.arange(24).reshape(2, 3, 4), [np.arange(2), np.arange(3), np.arange(4)])
    result1 = list(array1.enumerate())
    correct1 = [[[0, 0, 0], 0], [[0, 0, 1], 1], [[0, 0, 2], 2], [[0, 0, 3], 3],
                [[0, 1, 0], 4], [[0, 1, 1], 5], [[0, 1, 2], 6], [[0, 1, 3], 7],
                [[0, 2, 0], 8], [[0, 2, 1], 9], [[0, 2, 2], 10], [[0, 2, 3], 11],
                [[1, 0, 0], 12], [[1, 0, 1], 13], [[1, 0, 2], 14], [[1, 0, 3], 15],
                [[1, 1, 0], 16], [[1, 1, 1], 17], [[1, 1, 2], 18], [[1, 1, 3], 19],
                [[1, 2, 0], 20], [[1, 2, 1], 21], [[1, 2, 2], 22], [[1, 2, 3], 23]]
    assert(np.array_equal(result1, correct1))
