from os.path import join
from pkg_resources import Requirement, resource_filename
from pytest import fixture
import numpy as np
# noinspection PyPackageRequirements
from noformat import File

from algorithm.time_series import Recording

TEST_DATA_FOLDER = 'plptn/algorithm/test/data'


@fixture
def data():
    data_file = resource_filename(Requirement.parse('algorithm'), join(TEST_DATA_FOLDER, 'contra-test'))
    return File(data_file)


# noinspection PyShadowingNames
def test_fold_trials(data):
    record = Recording.load(data)
    result = record.fold_trials()
    original = np.rollaxis(np.stack([data['spike']['data'][0:3, 0:3], data['spike']['data'][:3, 20:23],
                                     data['spike']['data'][:3, 40:43]]), 1)
    assert np.allclose(result[:3, :3, :3].values, original)
