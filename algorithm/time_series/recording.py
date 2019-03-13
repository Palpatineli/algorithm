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
L = TypeVar('L', bound='DataFrame')

def _fix_to_length(array: np.ndarray, length: int) -> np.ndarray:
    """extend or cut the second axis of a 2D array to length """
    if array.shape[1] >= length:
        return array[:, 0:length]
    new_array = np.zeros((array.shape[0], length), dtype=array.dtype)
    new_array[:, :array.shape[1]] = array
    return new_array


class Recording(DataFrame, Stimulus):
    def __init__(self, data: Union[DataFrame, Tuple], stimulus: dict, sample_rate: int) -> None:
        self.converted = False
        if isinstance(data, DataFrame):
            DataFrame.__init__(self, data.values, data.axes)
        else:
            DataFrame.__init__(self, data[0], data[1])
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
        if self.converted:
            return self
        trial_time = np.arange(0, self.trial_time, 1 / self.sample_rate)
        trial_no, trial_length = len(self), len(trial_time)
        self.values: np.ndarray = _fix_to_length(self.values, trial_length * trial_no).\
            reshape(self.values.shape[0], (trial_no, trial_length))
        self.axes: List[np.ndarray] = [self.axes[0], np.arange(trial_no), trial_time]
        self.converted = True
        return self

    def fold_by(self: T, other: K, resample_to_self: bool = False) -> T:
        """Fold traces to other's trial starts and ends.
        Here trial time points can be anything, including motion onset and stimulus onset.
        Args:
            other: another recording, such as sparse trial recording
            resample_to_self: if true, self retains original sampling,
                if false, self is resampled to other's sample rate
        Returns:
            a copy of folded recording, that has the same sample rate as other, and using
                the other's trial starts and ends
        """
        # noinspection PyProtectedMember
        segments, trial_length = other._segments()
        if resample_to_self:
            segments = np.rint(segments * (self.sample_rate / other.sample_rate))
            trial_length = np.rint(trial_length * (self.sample_rate / other.sample_rate))
            full_trace = self.values
        else:
            full_trace = resample(self.values, self.sample_rate, other.sample_rate, axis=1)
        folded = np.stack(take_segment(trace, segments, trial_length) for trace in full_trace)
        sample_rate = self.sample_rate if resample_to_self else other.sample_rate
        axes = [self.axes[0].copy(), np.arange(segments.shape[0]), np.arange(0, other.trial_time, 1 / sample_rate)]
        result = self.create_like(folded, axes)
        result.sample_rate = sample_rate
        return result

    def _segments(self) -> Tuple[np.ndarray, int]:
        trial_time = np.arange(0, self.trial_time, 1 / self.sample_rate)
        trial_length = len(trial_time)
        trial_no = self.values.shape[1] if self.converted else (self.values.shape[1] / trial_length)
        return np.arange(trial_no) * trial_length, trial_length

    def filter_by(self: T, **kwargs) -> T:
        """filter by specific features of the stimulus, for example: filter_by(direction=180.0)"""
        sequence = self.stimulus['sequence']
        filter_keys = set(kwargs.keys())
        all_keys = sequence.keys()
        # noinspection PyTypeChecker
        mask = reduce(np.logical_and, (np.equal(sequence[key], value) for key, value in kwargs.items()))
        result = self.create_like(self.values.compress(mask, 1))
        result.stimulus['sequence'] = {key: np.array(sequence[key])[mask] for key in all_keys - filter_keys}
        return result

def fold_by(self: L, other: K, sample_rate: float, resample_to_self: bool = False) -> L:
    """Fold traces to other's trial starts and ends. Same as Recording::fold_by except that
    this one can be used with isinstance(self, DataFrame) == True"""
    # noinspection PyProtectedMember
    segments, trial_length = other._segments()
    if resample_to_self:
        segments = np.rint(segments * (sample_rate / other.sample_rate)).astype(np.int_)
        trial_length = np.rint(trial_length * (sample_rate / other.sample_rate)).astype(np.int_)
        full_trace = self.values
    else:
        full_trace = resample(self.values, sample_rate, other.sample_rate, axis=1)
    folded = np.stack([take_segment(trace, segments, trial_length) for trace in full_trace])
    sample_rate = sample_rate if resample_to_self else other.sample_rate
    axes = [self.axes[0].copy(), np.arange(segments.shape[0]),
            np.arange(0, other.trial_time, 1 / sample_rate)]
    result = self.create_like(folded, axes)
    return result
