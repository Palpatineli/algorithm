from typing import List
from collections import namedtuple
import numpy as np
STIM_FRAME_RATE = 60
UniqueSequence = namedtuple('UniqueSequence', ['names', 'order', 'indices', 'features'])


class Stimulus(object):
    stim_time: float = 0
    blank_time: float = 0
    trial_time: float = 0

    def __init__(self, stimulus: dict) -> None:
        self.stimulus = stimulus
        self._parse(stimulus)

    def _parse(self, stimulus: dict) -> None:
        stim_config = stimulus['config']
        stim_time = stim_config.get('stim_time', None)
        self.stim_time = stim_time if stim_time else stim_config['movie_length'] / STIM_FRAME_RATE
        self.blank_time = stim_config['blank_time']
        self.trial_time = self.stim_time + self.blank_time

    def __len__(self) -> int:
        return len(self.stimulus['timestamps'])

    def sequence(self, feature: str) -> dict:
        return np.asarray(self.stimulus['sequence'][feature])

    def unique_trials(self, features: List[str] = None) -> UniqueSequence:
        """find unique trials from stimulus sequence
        Args:
            sequence: stim_seq['sequence']
        Returns:
            names, order, indices, features
        """
        sequence = self.stimulus['sequence']
        if features:
            feature_name, feature_seq = features, np.array([sequence[x] for x in features]).T
        else:
            feature_name, feature_seq = list(sequence.keys()), np.array(list(sequence.values())).T
        unique_order, unique_indices = np.unique(feature_seq, axis=0, return_inverse=True)
        unique_features = [np.unique(x) for x in unique_order.T]
        return UniqueSequence(feature_name, unique_order, unique_indices, unique_features)
