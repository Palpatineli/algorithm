from typing import Dict, List, Callable, Any, Tuple, Generator, Union, Optional
from itertools import combinations
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def perm_test(dist1: np.ndarray, dist2: np.ndarray, iter_no: int = 1000) -> float:
    """from 2 of 1D arrays of data, test whether they are from different distributions.
    Without assumptions. Except that mean is unbiased estimation."""
    dist1, dist2 = np.asarray(dist1), np.asarray(dist2)
    size1, size2 = dist1.shape[0], dist2.shape[0]
    mean1 = [np.mean(np.random.choice(dist1, size1, True)) for _ in range(iter_no)]
    mean2 = [np.mean(np.random.choice(dist2, size2, True)) for _ in range(iter_no)]
    sorted1, sorted2 = np.sort(mean1), np.sort(mean2)
    indices = np.searchsorted(sorted1, sorted2)
    if sorted1[sorted1.shape[0] // 2] > sorted2[sorted2.shape[0] // 2]:  # compare median
        return indices.mean() / iter_no
    else:
        return 1 - indices.mean() / iter_no

def combine_test(data: Dict[str, np.ndarray], tests: List[Callable[[np.ndarray, np.ndarray], Any]]) -> str:
    result: List[str] = list()
    for x, y in combinations(data, 2):
        for test in tests:
            result.append(f"{test.__name__}: {x} vs. {y} {test(data[x], data[y])}")
    return '\n'.join(result)

import numpy as np

Bounds = Tuple[Tuple[float, float], Tuple[int, int], Tuple[int, int]]
Decoder = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

def scale_features(x: np.ndarray, axes: Union[List[int], int, None] = None) -> np.ndarray:
    return (x - x.mean(axes, keepdims=True)) / x.std(axes, keepdims=True)

def split_by_proportion(proportions: np.ndarray, total_count: int) -> np.ndarray:
    """
    Args:
        proportions: array of floats that should sum to 1.
        total_count: total number of items
    Returns:
        array of int, to have as close to proportions as possible
    """
    result_f64 = proportions * total_count
    result = result_f64.astype(np.int)
    fractions = result_f64 - result
    number_needed = int(round(fractions.sum()))
    result[np.argsort(-fractions)[0: number_needed]] += 1
    return result

def oversample(X, y, size):
    """Over-draft to reach size of sample numbers."""
    assert X.shape[0] == y.shape[0], "X and y should have the sample number of samples"
    labels, counts = np.unique(y, return_counts=True)
    sample_no = X.shape[0]
    extra_numbers = split_by_proportion(counts / sample_no, size - sample_no)
    indices = np.hstack([np.random.choice(np.nonzero(y == label)[0], extra_no, True)
                         for label, extra_no in zip(labels, extra_numbers)])
    return np.vstack([X, X[indices, :]]), np.hstack([y, y[indices]])

ValidateSet = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def split_data(X: np.ndarray, y: np.ndarray, repeats: int, train_size: int,
               test_size: int) -> Generator[ValidateSet, None, None]:
    """Create generator for training and testing datasets."""
    if (train_size + test_size) > X.shape[0]:
        X, y = oversample(X, y, np.int(np.ceil((train_size + test_size) / X.shape[0])) * X.shape[0])
    splitter = StratifiedShuffleSplit(repeats, test_size, train_size)
    for train_idx, test_idx in splitter.split(X, y):
        yield X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def split_time_series(X: np.ndarray, y: np.ndarray, fold: int, random: bool = False,
                      repeats: Optional[int] = None) -> Generator[ValidateSet, None, None]:
    """Create generator for random training and testing datasets from a timeseries where time
    is in the last dim.
    Args:
        X: predictor data, last dimension is time
        y: continuous label data, last dimension is time, must be same as that of X
        repeats: how many sets needed for the validation
        test_portion: the portion of test period between (0.0, 1.0)
    """
    full_length = y.shape[-1]
    try:
        assert X.shape[-1] == full_length
        assert full_length > fold > 0
        assert (not random) or (repeats is not None)
    except AssertionError as e:
        print(X.shape, y.shape, fold, random, repeats)
        raise e
    test_length = full_length // fold
    if random:
        starts = np.random.randint(0, (full_length - test_length), repeats)
    else:
        starts = np.arange(fold) * test_length
    for start in starts:
        end = test_length + start
        X_te, y_te = X.take(range(start, end), -1), y.take(range(start, end), -1)
        X_tr = np.concatenate([X.take(range(0, start), -1), X.take(range(end, full_length), -1)], axis=-1)
        y_tr = np.concatenate([y.take(range(0, start), -1), y.take(range(end, full_length), -1)], axis=-1)
        yield X_tr, y_tr, X_te, y_te
