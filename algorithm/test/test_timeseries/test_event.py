from os.path import join
from pkg_resources import resource_filename, Requirement
import numpy as np
from pytest import fixture
from scipy.io import loadmat
from reader.object_array2dict import convert as cell2dict

from algorithm.array import DataFrame
from algorithm.time_series.event import find_deviate

TEST_DATA_FOLDER = 'algorithm/test/data/'

@fixture
def lever_file():
    mat_file = resource_filename(Requirement.parse('algorithm'),
                                 join(TEST_DATA_FOLDER, '20170619_trackball_2072_2P.mat'))
    return cell2dict(loadmat(mat_file)['data'])

def test_find_deviate(lever_file):
    raw_movement = lever_file['response']['mvmtdata'].ravel()
    calibration_factor = lever_file['params']['lev_cal']
    trace = raw_movement / calibration_factor
    recording = DataFrame(trace, [np.arange(len(trace))])
    events = find_deviate(recording)
    print(events)
