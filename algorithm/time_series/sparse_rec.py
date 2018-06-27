from typing import TypeVar, List, Tuple
from copy import deepcopy
import numpy as np
from ..array import DataFrame
from .stimulus import Stimulus
from .recording import Recording
from .utils import take_segment

T = TypeVar("T", bound="SparseRec")

class SparseRec(Recording):
    onset = None  # type: int
    trial_samples = None  # type: np.ndarray
    trial_anchors = None  # type: np.ndarray
    pre_time = None  # type: float
    post_time = None  # type: float

    def __init__(self, data: DataFrame, stimulus: dict, sample_rate: int) -> None:
        self.initialized = False  # type: bool
        self.converted = False
        DataFrame.__init__(self, data.values, data.axes)
        Stimulus.__init__(self, stimulus)
        self.sample_rate = sample_rate

    def set_trials(self, trials: np.ndarray, pre_time: float, post_time: float) -> None:
        self.trial_anchors = trials
        self.pre_time = pre_time
        self.post_time = post_time
        self._pre = int(round(pre_time * self.sample_rate))
        self._post = int(round(post_time * self.sample_rate))
        self.onset = self._pre
        self.trial_samples = self._pre + self._post
        self.initialized = True

    def create_like(self: T, values: np.ndarray, axes: List[np.ndarray] = None) -> T:
        new_obj = self.__class__(DataFrame(values, (axes if axes else self.axes)), deepcopy(self.stimulus),
                                 self.sample_rate)
        new_obj.set_trials(self.trial_anchors, self.pre_time, self.post_time)
        return new_obj

    def fold_trials(self: T) -> T:
        result = np.stack([take_segment(value, self.trial_anchors - self._pre, self.trial_samples)
                           for value in self.values])
        self.values = result
        self.axes = [self.axes[0], np.arange(len(self.trial_anchors)),
                     np.linspace(self.pre_time, self.post_time, self.trial_samples)]
        self.converted = True
        return self

    def _segments(self) -> Tuple[np.ndarray, int]:
        """returns timepoints and lengths in samples"""
        return self.trial_anchors - self._pre, self.trial_samples
