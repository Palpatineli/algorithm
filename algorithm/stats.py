import numpy as np

def perm_test(dist1: np.ndarray, dist2: np.ndarray, iter_no: int = 1000) -> float:
    """from 2 of 1D arrays of data, test whether they are from different distributions.
    Without assumptions. Except that mean is unbiased estimation."""
    size1, size2 = dist1.shape[0], dist2.shape[0]
    mean1 = [np.mean(np.random.choice(dist1, size1, True)) for _ in range(iter_no)]
    mean2 = [np.mean(np.random.choice(dist2, size2, True)) for _ in range(iter_no)]
    sorted1, sorted2 = np.sort(mean1), np.sort(mean2)
    indices = np.searchsorted(sorted1, sorted2)
    if sorted1[sorted1.shape[0] // 2] > sorted2[sorted2.shape[0] // 2]:  # compare median
        return indices.mean() / iter_no
    else:
        return 1 - indices.mean() / iter_no
