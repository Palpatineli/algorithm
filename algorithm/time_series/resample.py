"""functions to resample time series, and (potentially) correction bias"""
from typing import Union
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
from ..array import DataFrame as _df

DataFrame = Union[np.ndarray, pd.DataFrame, _df]

def resample(series: np.ndarray, source_freq: float, target_freq: float) -> np.ndarray:
    """resample the time series from timestamp1 to timestamp2
    Args:
        series: the input series or a bunch of series
        source_freq: the sampling frequency of the original series in Hz
        target_freq: the target sampling frequency
    Returns:
        new series in target sampling frequency
    """
    source_timestamp = np.arange(series.size) * (1.0 / source_freq)
    target_timestamp = np.arange(series.size * target_freq / source_freq) * (1.0 / target_freq)
    spline = InterpolatedUnivariateSpline(source_timestamp, series, ext=0)
    return spline(target_timestamp)

def resample_pd(series: pd.DataFrame, source_freq: float, target_freq: float) -> pd.DataFrame:
    return pd.DataFrame({col_id: resample(col, source_freq, target_freq)
                         for col_id, col in series.iteritems()})
