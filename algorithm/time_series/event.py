from typing import Tuple
import numpy as np
from numpy import logical_and
from scipy.signal import convolve
from .recording import Recording
from .utils import take_segment

def find_deviate(trace: Recording, quiet_var: float = 0.00001, window_size: int = 200,
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
    trace_val = trace.values  # type: np.ndarray
    if trace_val.ndim > 1 and trace_val.shape[0] == 1:
        trace_val = trace_val[0, :]
    mean_kernel = np.ones(window_size) / window_size
    square = convolve(trace_val ** 2, mean_kernel, mode='valid')
    mean = convolve(trace_val, mean_kernel, mode='valid')
    var = square - mean ** 2
    deviate = trace_val[window_size - 1:] - mean
    cross = logical_and(var[0: -1] < quiet_var,
                        logical_and(np.abs(deviate[0: -1]) <= event_thres, np.abs(deviate[1:] > event_thres)))
    cross_idx = np.nonzero(cross)[0]
    return cross_idx + window_size - 1, (deviate[0: -1] >= 0)[cross_idx]

def find_response_onset(data: Recording, quiet_var: float = 0.0001, window_size: int = 200,
                        event_thres: float = 0.4)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """calculate response onset after stimulus
    Args:
        data: the lever data read into a Recording
        quiet_var: the limit of pre_period to be considered quiet
        window_size: how many samples are included in the pre-period
        event_thres: the threshold to cross for correct trial
    Returns:
        event_onset: in sample id
        trial_onset: in sample id
        post_period: event reading in the [window_size] post stimulus/trial onset
        correct_trials: bool mask for trials with response post-stimulus
        quiet_trials: trials with no response pre-stimulus
    """
    onsets = np.asarray(data.stimulus['timestamps'], dtype=np.int)
    pre_period = take_segment(data.values[0, :], onsets - window_size, window_size)
    quiet_trials = pre_period.var(1) < quiet_var
    length = int(round(data.sample_rate * data.stim_time))
    post_period = take_segment(data.values[0, :], onsets, length)
    post_period_1 = post_period.copy()
    post_period = post_period - pre_period.mean(1).reshape(-1, 1)
    event_onset = np.argmax((post_period[:, 1:] > event_thres) & (post_period[:, 0:-1] <= event_thres), axis=1)
    correct_trials = (post_period.max(axis=1) > event_thres)
    return (event_onset + onsets)[correct_trials], onsets, post_period_1, correct_trials, quiet_trials

def _find_response_onset(data: Recording, quiet_var: float = 0.0001, window_size: int = 200,
                         event_thres: float = 0.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """calculate response onset after stimulus
    Returns:
        event_onset in samples, bool mask for trials with response post-stimulus,
        trials with no response pre-stimulus
    """
    onsets = np.asarray(data.stimulus['timestamps'])
    post_period = take_segment(data.values[0, :], onsets, int(round(data.sample_rate * data.stim_time))) -\
        take_segment(data.values[0, :], onsets - window_size, window_size).mean(1).reshape(-1, 1)
    event_onset = np.argmax((post_period[:, 1:] > event_thres) & (post_period[:, 0:-1] <= event_thres), axis=1)
    correct_trials = post_period.max(axis=1) > event_thres
    return (event_onset + onsets)[correct_trials]
