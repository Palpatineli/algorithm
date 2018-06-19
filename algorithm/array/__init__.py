"""Replace limited pandas functionality.
define a DataFrame with a data tensor and named coordinates
define functions performed on the data
"""
from .main import DataFrame, stack, common_axis, search_ar

__all__ = ['DataFrame', 'stack', 'common_axis', 'search_ar']
