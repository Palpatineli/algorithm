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

TEST_DATA_FOLDER = 'plptn/algorithm/test/data/contra-test'


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

def test_recording():

