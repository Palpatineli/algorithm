from typing import Dict
import numpy as np


def period2phase(starts: np.ndarray, ends: np.ndarray, record_length: int,
                 sample_rate: float=20.0) -> (np.ndarray, int, int):
    """
    Returns:
        phases, start_index, end_index (+1)
    """
    frames = np.arange(0.5, record_length + 0.5) * (1.0 / sample_rate)
    start_index = np.searchsorted(frames, starts[0])
    end_index = np.searchsorted(frames, ends[-1]) if ends[-1] < frames[-1] else len(frames)
    frames = frames[start_index: end_index]
    indices = np.searchsorted(ends, frames)
    phases = (frames - starts[indices]) / (ends - starts)[indices] * (2 * np.pi)
    return phases, start_index, end_index


def circular_sum(spike_series: Dict[str, np.ndarray], starts: np.ndarray, period: float) -> np.ndarray:
    """calculate the circular sum of a group of spike series"""
    spike_range = np.searchsorted(spike_series['y'], [starts[0], starts[-1] + period])
    spike_segment = spike_series['data'][:, spike_range[0]: spike_range[1]]
    spike_index = spike_series['y'][spike_range[0]: spike_range[1]]
    phases = np.exp(-1j * (spike_index - starts[np.searchsorted(starts, spike_index) - 1]) / period * (2 * np.pi))
    leaked_f0 = phases.sum() * spike_segment.mean(axis=1)
    f1 = (spike_segment * phases).sum(axis=1)
    return f1 - leaked_f0
