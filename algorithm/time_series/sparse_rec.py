from typing import TypeVar, List, Tuple, Optional
from copy import deepcopy
import numpy as np
from algorithm.array import DataFrame
from .stimulus import Stimulus
from .recording import Recording
from .utils import take_segment
from .event import find_deviate, find_response_onset

T = TypeVar("T", bound="SparseRec")

class SparseRec(Recording):
    onset = None  # type: int
    trial_samples = None  # type: np.ndarray
    trial_anchors = None  # type: np.ndarray
    pre_time = None  # type: float
    post_time = None  # type: float
    mask = None  # type: np.ndarray

    def __init__(self, data: DataFrame, stimulus: dict, sample_rate: int) -> None:
        self.initialized = False  # type: bool
        self.converted = False
        DataFrame.__init__(self, data.values, data.axes)
        Stimulus.__init__(self, stimulus)
        self.sample_rate = sample_rate

    def set_trials(self, trials: np.ndarray, pre_time: float, post_time: float) -> None:
        if self.initialized:
            raise ValueError("already initialized!")
        self.trial_anchors = trials
        self.pre_time = pre_time
        self.post_time = post_time
        self._pre = None if pre_time is None else int(round(pre_time * self.sample_rate))
        self._post = None if post_time is None else int(round(post_time * self.sample_rate))
        self.onset = self._pre
        if self._pre is not None and self._post is not None:
            self.trial_samples = self._pre + self._post
        self.initialized = True

    def center_on(self: T, mode: str = 'motion', **kwargs) -> T:
        if self.initialized:
            return self
        pre_time = kwargs.pop("pre_time", self.stimulus['config']['blank_time'])
        post_time = kwargs.pop("post_time", self.stimulus['config']['stim_time'])
        if mode == 'motion':
            onset, _, _, mask, _ = find_response_onset(self, **kwargs)
            self.set_trials(onset, pre_time, post_time)
            self.mask = mask
        elif mode == 'stim':
            self.set_trials(find_deviate(self, **kwargs), pre_time, post_time)
        return self

    def create_like(self: T, values: np.ndarray, axes: Optional[List[np.ndarray]] = None) -> T:
        new_obj = self.__class__(DataFrame(values, (axes if axes is not None else self.axes)), deepcopy(self.stimulus),
                                 self.sample_rate)
        new_obj.set_trials(self.trial_anchors, self.pre_time, self.post_time)
        new_obj.initialized = self.initialized
        return new_obj

    def fold_trials(self: T) -> T:
        result = np.stack([take_segment(value, self.trial_anchors - self._pre, self.trial_samples)
                           for value in self.values])
        self.values: np.ndarray = result
        self.axes: List[np.ndarray] = [self.axes[0], np.arange(len(self.trial_anchors)),
                                       np.linspace(-self.pre_time, self.post_time, self.trial_samples)]
        self.converted = True
        return self

    def _segments(self) -> Tuple[np.ndarray, int]:
        """returns timepoints and lengths in samples"""
        return self.trial_anchors - self._pre, self.trial_samples

    def scale(self):
        values = self.values
        pre = values[(slice(None),) * (values.ndim - 1) + (slice(None, self.onset),)]
        post = values[(slice(None),) * (values.ndim - 1) + (slice(self.onset, None),)]
        post_max = post.max(axis=-1, keepdims=True)
        pre_mean = pre.mean(axis=-1, keepdims=True)
        self.values = (values - pre_mean) / (post_max - pre_mean)
