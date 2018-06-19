import numpy as np
from scipy.optimize import minimize, OptimizeResult
from .rf import dog, dog_ml


def dog_inner(x: np.ndarray, y: np.ndarray, sd: np.ndarray) -> OptimizeResult:
    max_loc = x[y.argmax()]
    y_range = y.max() - y.min()
    initial_values = np.array([y_range * 2, y_range * 2 - y[0], max_loc * 1.5, max_loc * 1.0, y.min()])
    bounds = [(0, None), (0, None), (None, None), (None, None), (None, None)]
    return minimize(dog_ml, initial_values, args=(x, y, sd), method='SLSQP', bounds=bounds)


def bootstrap_normal(original_data: np.ndarray, repeat: int=100) -> np.ndarray:
    """assume normal spread of recording values generate simulated date set
    Args:
        original_data: row is each condition, column for repetitions
        repeat: size of simulated dataset
    Returns:
        numpy array; 1st dimension: datasets; 2nd dimension: conditions; 3rd dimension repeats
    """
    y = np.nanmean(original_data, axis=1).reshape((1, -1, 1))
    noise = np.random.randn(repeat, *original_data.shape)
    return noise * np.nanstd(original_data, axis=1).reshape((1, -1, 1)) + y


def bootstrap_sample(original_data: np.ndarray, repeat: int=100) -> np.ndarray:
    """generate simulated date set, by sample with replacement
    Args:
        original_data: row is each condition, column for repetitions
        repeat: size of simulated dataset
    Returns:
        numpy array; 1st dimension: datasets; 2nd dimension: conditions; 3rd dimension repeats
    """
    index = np.random.randint(0, original_data.shape[1], (repeat, *original_data.shape))
    _, y, _ = np.meshgrid(np.arange(repeat), np.arange(original_data.shape[0]), np.arange(original_data.shape[1]),
                          indexing='ij')
    return original_data[y, index]


def dog_interval(response, diameter, feature_func, bootstrap_repeat: int=100, p: float=0.05) -> OptimizeResult:
    y = np.nanmean(response, axis=1)
    sd = np.nanstd(response, axis=1) / np.sqrt(15)
    fit = dog_inner(diameter, y, sd)
    fit.evaluate = lambda a: dog(fit.x, a)
    fit.parameter_names = ['k_c', 'k_s', 'w_c', 'w_s', 'f_0']
    fit.df = len(y) - len(fit.parameter_names)
    parameter_dist = list()
    random_data = bootstrap_sample(response, bootstrap_repeat)
    for random_case in random_data:
        y = np.nanmean(random_case, axis=1)
        sd = np.nanstd(random_case, axis=1) / np.sqrt(15)
        random_fit = dog_inner(diameter, y, sd)
        if random_fit.success:
            parameter_dist.append(feature_func(random_fit))
    parameter_dist = np.sort(np.array(parameter_dist), 0)
    fit.parameter_dist = parameter_dist
    bounds = (int(round(len(parameter_dist) * p / 2.0)), int(round(len(parameter_dist) * (1.0 - p / 2.0))))
    fit.parameter_ci = (parameter_dist[bounds[0], :], parameter_dist[bounds[1], :])
    return fit
