# pylint:disable=W0621
from os.path import join
import json
from pkg_resources import resource_filename, Requirement
from pytest import fixture
import numpy as np

# noinspection PyProtectedMember
from algorithm.time_series.stimulus import Stimulus
from algorithm.time_series.utils import take_segment
from algorithm.time_series.resample import resample
from algorithm.time_series.recording import Recording, DataFrame

TEST_DATA_FOLDER = 'algorithm/test/data/contra-test'


@fixture
def stim_seq():
    json_file = resource_filename(Requirement.parse('algorithm'), join(TEST_DATA_FOLDER, 'stimulus.log'))
    with open(json_file) as fp:
        yield json.load(fp)


# noinspection PyShadowingNames
def test_unique(stim_seq: dict):
    unique_seq = Stimulus(stim_seq).unique_trials()
    assert np.allclose(unique_seq.order[unique_seq.indices[2]][unique_seq.names.index('contrast')], 0.01930698)
    assert np.allclose(unique_seq.order[unique_seq.indices[-12]][unique_seq.names.index('size_x')], 1920.0)

def test_take_segment():
    trace = np.arange(5, 1005)
    points = np.array([3, 18, 518, 998])
    result = take_segment(trace, points - 3, 5)
    assert(np.array_equal(result, np.array(
        [[5, 6, 7, 8, 9],
         [20, 21, 22, 23, 24],
         [520, 521, 522, 523, 524],
         [1000, 1001, 1002, 1003, 1004]])))

def test_resample():
    result = resample(np.arange(10), 5, 3)
    correct1 = np.linspace(0, 10, 6, endpoint=False)
    assert(np.allclose(result, correct1))

# noinspection PyShadowingNames
def test_recording(stim_seq: dict):
    record = Recording(DataFrame(np.arange(20).reshape(2, 10), [np.array(['a', 'b']), np.arange(10)]), stim_seq, 2.5)
    record.stim_time = 1.0
    record.blank_time = 1.0
    record.trial_time = 2.0
    record2 = Recording(DataFrame(np.arange(24).reshape(3, 8), [np.array(['a', 'b', 'c']), np.arange(8)]),
                        stim_seq, 2)
    record2.stim_time = 1.0
    record2.blank_time = 1.0
    record2.trial_time = 2.0
    folded = record2.fold_by(record)
    correct2 = np.array([
        [[0, 0.8, 1.6, 2.4, 3.2],
         [4, 4.8, 5.6, 6.4, 7.2]],
        [[8, 8.8, 9.6, 10.4, 11.2],
         [12, 12.8, 13.6, 14.4, 15.2]],
        [[16, 16.8, 17.6, 18.4, 19.2],
         [20, 20.8, 21.6, 22.4, 23.2]]])
    assert(np.allclose(folded.values, correct2))
