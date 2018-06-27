from typing import Tuple
import numpy as np
from numpy import logical_and
from scipy.signal import convolve
from .recording import Recording
from .utils import take_segment

def find_deviate(trace: np.ndarray, quiet_var: float = 0.00001, window_size: int = 200,
                 event_thres: float = 0.004) -> np.ndarray:
    """Find events where trace stays quiet then move over a threshold.
    Default parameters is configured for regular lever pushes after a 0.125 s gaussian filter.
    Args:
        trace: 1d time series
        quiet_var: max variance during resting state
        window_size: window size in samples for both resting state variance and mean calculation
        event_thres: how much does it need to deviate from windowed mean to be an event
    Returns:
        (indices of events, bool array for these indices being positive crossings)
    """
    mean_kernel = np.ones(window_size) / window_size
    square = convolve(trace ** 2, mean_kernel, mode='valid')
    mean = convolve(trace, mean_kernel, mode='valid')
    var = square - mean ** 2
    deviate = trace[window_size - 1:] - mean
    cross = logical_and(var[0: -1] < quiet_var,
                        logical_and(np.abs(deviate[0: -1]) <= event_thres, np.abs(deviate[1:] > event_thres)))
    cross_idx = np.nonzero(cross)[0]
    return cross_idx + window_size - 1, (deviate[0: -1] >= 0)[cross_idx]

def find_response_onset(data: Recording, quite_var: float = 0.0001, window_size: int = 200,
                        event_thres: float = 0.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    onsets = np.asarray(data.stimulus['timestamps'])
    pre_period = take_segment(data.values[0, :], onsets - window_size, window_size)
    quiet_trials = pre_period.var(1) < quite_var
    pre_means = pre_period.mean(1)
    post_period = take_segment(data.values[0, :], onsets, int(round(data.sample_rate * data.stim_time)))
    post_period -= pre_means.reshape(-1, 1)
    event_onset = np.argmax(np.logical_and(post_period[:, 1:] > event_thres,
                                           post_period[:, 0:-1] <= event_thres), axis=1)
    correct_trials = np.any(post_period > event_thres, axis=1)
    return (event_onset + onsets)[correct_trials], correct_trials, quiet_trials
