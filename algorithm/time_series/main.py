"""generic functions manipulating time series recordings """
import numpy as np
from scipy.stats import ttest_ind
from sklearn.neighbors import KDTree

from ..array import DataFrame
from .recording import Recording

FRAME_RATE = 60


def erase_noise(series: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """eliminate peaks with areas smaller than threshold
    Args:
        series: 1-D time series to process, should be sparse
        threshold: peaks smaller than threshold will be eliminated
    Returns:
        a cleaned series
    """
    nonzero = np.greater(series.ravel(), 0.001)
    nonzero[0] = nonzero[-1] = False
    # noinspection PyTypeChecker,PyUnresolvedReferences
    rise = np.nonzero(np.logical_and(nonzero[1:], np.logical_not(nonzero[:-1])))[0] + 1
    # noinspection PyTypeChecker,PyUnresolvedReferences
    fall = np.nonzero(np.logical_and(nonzero[:-1], np.logical_not(nonzero[1:])))[0] + 1
    for start, end in zip(rise, fall):
        if series[start:end].sum() < threshold:
            series[start:end] = 0
    return series


def fix_to_length(array: np.ndarray, length: int) -> np.ndarray:
    """extend or cut the second axis of a 2D array to length """
    if array.shape[1] >= length:
        return array[:, 0:length]
    new_array = np.zeros((array.shape[0], length), dtype=array.dtype)
    new_array[:, :array.shape[1]] = array
    return new_array


def ori_tuning(x: Recording) -> DataFrame:
    """extract ori_tuning from spiking time series data
    Args:
        x: recorded data of spike count, where x: cells, y: trials, z: samples
    Returns:
        a dataframe where columns are cells and columns are mean firing rate on each orientation
            following the orientations in index
    """
    return x[:, :, x.onset:].group_by([x.sequence('direction')], lambda x: np.mean(x, (1, 2)))


def ori_selectivity(x: DataFrame) -> DataFrame:
    """circular summation version, see @banerjee2015 and @castro2014
    Args:
        x: dataframe, where y are angles in degrees, x are cells
    Returns:
        the index, 0 for no orientation selectivity, and 1 for maximum
    """
    cell_id, orientations = x.axes[0], np.exp(np.deg2rad(x.axes[1]) * (2 * 1j))
    return x.create_like(np.abs((x.values * orientations).sum(1)) / x.values.sum(1), [cell_id])


def reliability(data: Recording) -> Recording:
    r"""calculate reliability defined as correlation between repeats of external impact
    .. math:: Reliability = \frac{2}{T^2 - T} \sum_{i=1}^T{\sum_{j=i+1}^T{\rho(R_i, R_j)}}
    Args:
        data: 3d DataFrame, with axes [cell, trials, time/sample]
        stim_seq: stimulus dictionary
        sample_rate: sample_rate of data
    Returns:
        reliability, with axes [cell, movie_id]
    """
    def _movie_id(movie_name: str) -> int:
        return int(movie_name[3:movie_name.find('.')])

    def _reliability(plane: np.ndarray) -> np.ndarray:
        trial_no = plane.shape[1]
        coef = 2 / (trial_no**2 - trial_no)
        return np.array([np.triu(np.corrcoef(cell, rowvar=False)).sum() for cell in plane]) * coef

    movie_ids = np.array([_movie_id(x) for x in data.sequence('movie_id')])
    return data[:, :, :data.onset].group_by([movie_ids], _reliability)  # type: ignore


def sparseness(data: Recording) -> Recording:
    r"""Sparseness as defined by inter-frame variation binned through trials.
    .. math:: Sparsenss = \frac{N - \frac{(\sum_i{R_i})^2}{\sum_i(R_i^2)}}{N - 1}
    whith ith frame and N the number of frames (use data sample rate as that's slower
    than stimulus framerate)
    Args:
        data: 3D DataFrame, dims: [cell-id, trial, timecourse], with stimulus and sample rate
    Returns:
        2D DataFrame of sparseness (a.u.), dims: [cell-id, condition]
    """
    activity = data[:, :, data.onset:].group_by([np.array(data.sequence('movie_id'))], lambda x: np.mean(x, 1))
    activity_mat, activity_axes = activity.values, activity.axes
    value = (data.trial_samples - activity_mat.sum(-1) ** 2 / (activity_mat ** 2).sum(-1)) / (data.trial_samples - 1)
    return data.create_like(value, activity_axes[0: 2])


def signal_noise_ratio_rvr(data: Recording) -> Recording:
    """from Rajeev Rikhye, but doesn't fit the data shown in Banerjee2016, which claims to use Rajeev's algorithm"""
    def z_activity(y: np.ndarray) -> np.ndarray:
        spontaneous = y[:, :, 0: data.onset].mean(1)
        evoked = y[:, :, data.onset:].mean(1)
        return (evoked.mean(1) - spontaneous.mean(1)) / (np.sqrt(evoked.std(1) * spontaneous.std(1)))

    return data.group_by([data.unique_trials().indices], z_activity)


def signal_noise_ratio_kpl(data: Recording) -> Recording:
    """From Keji Li, not a good definition either."""
    def _activity(y: np.ndarray) -> np.ndarray:
        spontaneous = y[:, :, 0: data.onset]
        evoked = y[:, :, data.onset:]
        spontaneous_var = spontaneous.var(-1)
        evoked_var = ((evoked - spontaneous.mean(-1)[:, :, None]) ** 2).mean(-1)
        return np.nanmean(np.sqrt(evoked_var / spontaneous_var - 1), 1)

    return data.group_by([data.unique_trials().indices], _activity)


def responsiveness(data: Recording) -> Recording:
    def _response(y: np.ndarray) -> np.ndarray:
        return y[:, :, data.onset:].mean((1, 2)) / y[:, :, :data.onset].mean((1, 2))

    return data.group_by([data.unique_trials().indices], _response)


def filter_responsive(data: Recording, alpha: float = 0.01) -> DataFrame:
    """give a boolean array on whether the activity of a cells at preferred (maximum) stimulus exceeds alpha
    based on k-s test with Bonferroni correction
    """
    spontaneous = data.values[:, :, 0:data.onset].reshape(data.shape[0], -1).T

    def _responsiveness(plane: np.ndarray) -> np.ndarray:
        """From 3d ndarray [cell_no, repeat, timecourse] to 1d [cell_no]"""
        flattened = plane.reshape(plane.shape[0], -1).T
        return ttest_ind(flattened, spontaneous).pvalue

    indices = data.unique_trials().indices
    responsive_p = data.group_by([indices], _responsiveness)
    return DataFrame(responsive_p.values.min(1) < (alpha / len(indices)), [data.axes[0]])

def mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    """estimate mutual information in two continuously distributed serieses using kNN"""
    pass
