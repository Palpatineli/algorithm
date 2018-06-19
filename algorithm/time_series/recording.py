from typing import Union, TypeVar, List, Tuple
from functools import reduce
from copy import deepcopy
import numpy as np
from noformat import File
from ..array import DataFrame
from .stimulus import Stimulus
from .utils import take_segment
from .resample import resample

T = TypeVar('T', bound='Recording')
K = TypeVar('K', bound='Recording')

def _fix_to_length(array: np.ndarray, length: int) -> np.ndarray:
    """extend or cut the second axis of a 2D array to length """
    if array.shape[1] >= length:
        return array[:, 0:length]
    new_array = np.zeros((array.shape[0], length), dtype=array.dtype)
    new_array[:, :array.shape[1]] = array
    return new_array


class Recording(DataFrame, Stimulus):
    def __init__(self, data: DataFrame, stimulus: dict, sample_rate: int) -> None:
        self.converted = False
        DataFrame.__init__(self, data.values, data.axes)
        Stimulus.__init__(self, stimulus)
        self.sample_rate = sample_rate
        self.onset = int(self.blank_time * sample_rate)
        self.trial_samples = int(self.trial_time * sample_rate)

    def create_like(self: T, values: np.ndarray, axes: List[np.ndarray] = None) -> T:
        return self.__class__(DataFrame(values, (axes if axes else self.axes)),
                              deepcopy(self.stimulus), self.sample_rate)

    @classmethod
    def load(cls, data_file: Union[str, File], data_target='spike') -> "Recording":
        if isinstance(data_file, str):
            data_file = File(data_file)
        sample_rate = data_file.attrs.get('spike_resolution', data_file.attrs['frame_rate'])
        return cls(DataFrame.load(data_file[data_target]), data_file['stimulus'], sample_rate)

    def fold_trials(self: T) -> T:
        """convert x, a data frame with columns of neurons and rows of samples, to a 3D array with x=neuron_id,
        y=trial_id, and z=frame_no_within_trial
        """
        trial_time = np.arange(0, self.trial_time, 1 / self.sample_rate)
        trial_no, trial_length = len(self), len(trial_time)
        self.values: np.ndarray = _fix_to_length(self.values, trial_length * trial_no).\
            reshape(self.values.shape[0], trial_no, trial_length)
        self.axes: List[np.ndarray] = [self.axes[0], np.arange(trial_no), trial_time]
        self.converted = True
        return self

    def fold_by(self: T, other: K) -> T:
        """Fold traces to other's trial starts and ends.
        Here trial timepoints can be anything, including motion onset and stimulus onset.
        Args:
            other: another recording, such as sprase trial recording
        Returns:
            a copy of folded recording, that has the same sample rate as other, and using
                the other's trial starts and ends
        """
        if self.converted:
            raise ValueError("cannot fold already folded recording")
        segments, trial_length = other._segments()
        full_trace = resample(self.values, self.sample_rate, other.sample_rate, axis=1)
        folded = take_segment(full_trace, *segments)
        axes = [self.axes[0].copy(), np.arange(segments.shape[0]),
                np.arange(0, other.trial_time, 1 / other.sample_rate)]
        result = self.create_like(folded, axes)
        result.sample_rate = other.sample_rate
        return result

    def _segments(self) -> Tuple[np.ndarray, int]:
        trial_time = np.arange(0, self.trial_time, 1 / self.sample_rate)
        trial_no, trial_length = len(self), len(trial_time)
        return np.arange(trial_no) * trial_length, trial_length

    def filter_by(self: T, **kwargs) -> T:
        """filter by specific features of the stimulus, for example: filter_by(direction=180.0)"""
        sequence = self.stimulus['sequence']
        filter_keys = set(kwargs.keys())
        all_keys = sequence.keys()
        mask = reduce(np.logical_and, (np.equal(sequence[key], value) for key, value in kwargs.items()))
        result = self.create_like(self.values.compress(mask, 1))
        result.stimulus['sequence'] = {key: np.array(sequence[key])[mask] for key in all_keys - filter_keys}
        return result
