import numpy as np
from pytest import fixture
from algorithm.array.main import DataFrame

@fixture
def correct1():
    return np.array([[[10.5, 12.5], [18.5, 20.5]]])

def test_group_mean(correct1):
    a = np.arange(32).reshape(1, 4, 4, 2)
    groupings = [np.tile([0, 1], 2), np.tile([0, 1], 2)]
    data = DataFrame(a, [[1], np.arange(4), np.arange(4), np.arange(2)])
    result = data.group_by(groupings, lambda x: np.mean(x, axis=(1, 2)))
    assert np.allclose(result.values, correct1)
