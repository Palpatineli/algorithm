from os.path import join
from pkg_resources import resource_filename, Requirement
import numpy as np
from pytest import fixture
from scipy.io import loadmat
from reader.object_array2dict import convert as cell2dict

from algorithm.array import DataFrame
from algorithm.time_series.event import find_deviate
from algorithm.time_series.utils import take_segment

TEST_DATA_FOLDER = 'algorithm/test/data/'

@fixture
def lever_file():
    mat_file = resource_filename(Requirement.parse('algorithm'),
                                 join(TEST_DATA_FOLDER, '20170619_trackball_2072_2P.mat'))
    return cell2dict(loadmat(mat_file)['data'])

@fixture
def correct_last_trial():
    trial_file = resource_filename(Requirement.parse('algorithm'),
                                   join(TEST_DATA_FOLDER, '20170619_trackball_2072_2P.npy'))
    with open(trial_file, 'rb') as fp:
        last_trial = np.load(fp)
    return last_trial

def test_find_deviate(lever_file):
    raw_movement = lever_file['response']['mvmtdata'].ravel()
    calibration_factor = lever_file['params']['lev_cal']
    trace = raw_movement / calibration_factor
    recording = DataFrame(trace, [np.arange(len(trace))])
    find_deviate(recording)
    assert(False)

def test_take_segments(lever_file, correct_last_trial):
    raw = lever_file['response']['mvmtdata'].ravel()
    onset = np.array(lever_file['response']['samples_start'] + [156000])
    segments = take_segment(raw, onset, 1280)
    print(segments)
    assert(np.allclose(segments[-1, :], correct_last_trial))
