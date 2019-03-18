"""Define the array class, hide implementation"""
from typing import Any, Callable, Dict, Tuple, List, TypeVar, Union, Optional, Iterator
from functools import reduce
import numpy as np
from .array_mixin import OpDelegatorMixin
from ._main import search_ar_int

T = TypeVar("T", bound="DataFrame")
NpzFile = Dict[str, np.ndarray]


def _is_1d(array: np.ndarray) -> bool:
    if array.ndim == 1:
        return True
    return len(np.flatnonzero(array.shape > 1)) <= 1


def _name_axes(axes: List[np.ndarray], copy: bool = False) -> Dict[str, np.ndarray]:
    """generate axis name in the sequence of x, y, z, xx, yy, zz, xxx, ..."""
    order = np.arange(len(axes))
    return {chr(x) * y: (axis.copy() if copy else axis) for x, y, axis in zip(order % 3 + 120, order // 3 + 1, axes)}


def _order_axes(named_axes: Dict[str, np.ndarray], copy: bool = True) -> List[np.ndarray]:
    """generate axis name in the sequence of x, y, z, xx, yy, zz, xxx, ..."""
    order = np.arange(len(named_axes.keys()) - (1 if 'data' in named_axes else 0))
    return [(named_axes[chr(x) * y].copy() if copy else named_axes[chr(x) * y])
            for x, y in zip(order % 3 + 120, order // 3 + 1)]

def _order_axes_old(named_axes: Dict[str, np.ndarray], copy: bool = True) -> List[np.ndarray]:
    """generate axis name in the sequence of row, column, z, xx, yy, zz, xxx, ..."""
    result = []
    for key in ['row', 'column']:
        if key in named_axes:
            result.append(named_axes[key])
    axes_no = len(named_axes.keys()) - (1 if 'data' in named_axes else 0)
    if axes_no > 2:
        order = np.arange(2, axes_no)
        result.extend([(named_axes[chr(x) * y].copy() if copy else named_axes[chr(x) * y])
                       for x, y in zip(order % 3 + 120, order // 3 + 1)])
    return result


class InvalidDataFrame(ValueError):
    pass


class DataFrame(OpDelegatorMixin):
    """Handles n dimensional array with marginal labels, faster than pandas/xarray."""
    def __init__(self, values: np.ndarray, axes: List[np.ndarray], validate: bool = False) -> None:
        if validate:
            self.validate(values, axes)
        self.values = values
        self.axes = axes  # type: List[np.ndarray]

    def create_like(self: T, values: np.ndarray, axes: List[np.ndarray] = None) -> T:
        if not axes:
            axes = self.axes
        return self.__class__(values, axes, validate=False)

    @staticmethod
    def validate(values: np.ndarray, axes: List[np.ndarray]) -> None:
        if values.ndim != len(axes):
            raise InvalidDataFrame("incompatible dimensions")
        for idx, (i, j) in enumerate(zip(axes, values.shape)):
            if not _is_1d(i):
                raise InvalidDataFrame("axes not 1d")
            if i.size != j:
                raise InvalidDataFrame(f"{idx}th axis lengths doesn't match")

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def shape(self) -> List[int]:
        return self.values.shape

    def astype(self, dtype):
        self.values = self.values.astype(dtype)
        return self

    def __getitem__(self: T, indices) -> T:
        if self.values.ndim == 1:
            return self.create_like(
                self.values.__getitem__(indices), [self.axes[0][indices]])
        else:
            return self.create_like(
                self.values.__getitem__(indices), [axis[index] for index, axis in zip(indices, self.axes)])

    def group_by(self: T, groupings: List[np.ndarray], func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 extra_axes: Optional[List[np.ndarray]] = None) -> T:
        """group data by groupings starting from the axis 1, and take the means on those axes
        Args:
            groupings: a list of groups, len(groupings[i]) == data['data'].shape[i + 1]
                np.unique(groupings[0]) will be returns['y'], np.unique(groupings[1]) will be returns['z']
            func: a function to operate on the groups, takes a 2d array and returns a 1d array with the same
                1st dimension
            extra_axes: ignored axes
        Returns:
            a n-dim DataFrame that is the mean by group from axes 1-m
        """
        source = self.values
        uniques, unique_indices = zip(*[np.unique(x, return_inverse=True) for x in groupings])
        group_seq = np.vstack([x.ravel() for x in np.meshgrid(*[np.arange(len(x)) for x in uniques], indexing='ij')]).T
        indices = np.vstack([x.ravel() for x in np.meshgrid(*[x for x in unique_indices], indexing='ij')]).T
        flat_source = source.reshape((source.shape[0], -1, *source.shape[len(uniques) + 1:]))
        mask = np.all(np.expand_dims(group_seq, 1) == np.expand_dims(indices, 0), -1)
        if func is None:
            flat_target = np.array([flat_source.compress(row, 1) for row in mask])
        else:
            flat_target = np.array([func(flat_source.compress(row, 1)) for row in mask])
        target_dims = [source.shape[0], *map(len, uniques), *flat_target.shape[2:]]
        target = np.swapaxes(flat_target, 0, 1).reshape(target_dims)
        if not extra_axes:
            extra_axes = self.axes[len(uniques) + 1:flat_target.ndim - 1 + len(uniques)]
        axes = [self.axes[0], *uniques, *extra_axes]
        return self.create_like(target, axes)

    def save(self, file_path: str) -> None:
        np.savez_compressed(file_path, **{'data': self.values, **_name_axes(self.axes)})

    @classmethod
    def load(cls, data: NpzFile, copy: bool = True) -> "DataFrame":
        if isinstance(data, str):
            data = np.load(data)  # noqa
        axes = _order_axes_old(data, copy=copy) if 'row' in data else _order_axes(data, copy=copy)
        return cls(data['data'].copy() if copy else data['data'], axes)

    def copy(self) -> "DataFrame":
        axes = [axis.copy() for axis in self.axes]
        return DataFrame(self.values.copy(), axes)

    def search(self, search_term: np.ndarray, axis_no: int = 0) -> "DataFrame":
        indices = search_ar(search_term, self.axes[axis_no])
        return self.create_like(self.values.take(indices, axis_no), self.axes[axis_no][indices])

    def append(self: T, other: T, axis: int = 0) -> T:
        try:
            result_value = np.concatenate(self.values, other.values)
        except ValueError:
            raise ValueError(f"dimensions do not match on axis {axis}")
        axes = [x.copy() for x in self.axes]
        axes[axis] = np.concatenate([axes[axis], other.axes[axis]], axis=0)
        return self.create_like(result_value, axes)

    def remove(self: T, indices: np.ndarray, axis: int = 0) -> T:
        mask = np.ones(self.values.shape[axis], dtype=np.bool_)
        mask[indices] = False
        self.values = self.values.compress(mask, axis)
        self.axes[axis] = self.axes[axis][mask]
        return self

    def take(self: T, indices: np.ndarray, axis: int = 0) -> T:
        axes = [*self.axes[0: axis], self.axes[axis][indices], *self.axes[axis + 1:]]
        if np.issubdtype(indices.dtype, np.bool_):
            return self.create_like(self.values.compress(indices, axis), axes)
        else:
            return self.create_like(self.values.take(indices, axis), axes)

    def mean(self: T, axis: int = -1) -> T:
        return self.create_like(self.values.mean(axis), [*self.axes[0: axis], *self.axes[axis + 1:]])

    def enumerate(self: T) -> Iterator[Tuple[List[Any], np.number]]:
        axes, ndim, values = self.axes, self.values.ndim, self.values
        indices = np.zeros(ndim + 1, dtype=np.int)
        max_shape = np.concatenate([[0], self.values.shape])
        top_index = ndim
        while indices[0] == 0:
            yield [axes[n][indices[n + 1]] for n in range(ndim)], values[tuple(indices[1:])]
            indices[ndim] += 1
            while indices[top_index] == max_shape[top_index]:
                indices[top_index] = 0
                top_index -= 1
                indices[top_index] += 1
                if indices[top_index] != max_shape[top_index]:
                    top_index = ndim


def search_ar(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """Find the locations of array1 elements in array2"""
    if array1.dtype == np.integer:
        return search_ar_int(array1, array2)
    arg2 = np.argsort(array2)
    arg1 = np.argsort(array1)
    rev_arg1 = np.argsort(arg1)
    sorted2_to_sorted1 = np.searchsorted(array2[arg2], array1[arg1])
    return arg2[sorted2_to_sorted1[rev_arg1]]


def stack(data_frames: List[T], axis: int = 0) -> T:
    try:
        result_value = np.concatenate([x.values for x in data_frames], axis)
    except ValueError:
        raise ValueError(f"dimensions do not match on axis {axis}")
    axes = [x.copy() for x in data_frames[0].axes]
    axes[axis] = np.concatenate([data.axes[axis] for data in data_frames], axis=0)
    return data_frames[0].create_like(result_value, axes)


def common_axis(data_frames: List[T], ax_id: int = 0) -> List[T]:
    """Filter an axis to get common planes.
    Args:
        data_frames: list of dataframes with the {ax_id}th axis having common element, all other dimensions
            need to be the same
        ax_id: the id of axis for repeats
    Return:
        the same dataframe list with the {ax_id}th axis masked to get the common elements
    """
    other_axes = tuple(range(ax_id)) + tuple(range(ax_id + 1, data_frames[0].values.ndim))
    axes = (x.axes[ax_id] for x in data_frames)
    # filter out planes with nan
    masks = (np.any(np.isnan(data.values), axis=other_axes) for data in data_frames)
    masked_axes = [(axis[~mask] if np.any(mask) else axis) for axis, mask in zip(axes, masks)]
    common_ids = reduce(np.intersect1d, masked_axes)
    indices = (search_ar(common_ids, x) for x in masked_axes)
    result = list()
    for index, data in zip(indices, data_frames):
        new_axes = [*data.axes[0:ax_id], data.axes[ax_id][index], *data.axes[ax_id + 1:]]
        result.append(data.create_like(data.values.take(index, ax_id), new_axes))
    return result

try:  # monkey patch conversion to pd if pandas is installed
    import pandas as pd

    def to_pd(self: DataFrame) -> Union[pd.DataFrame, pd.Series]:
        ndim = self.values.ndim
        if ndim == 1:
            return pd.Series(self.values, index=self.axes[0])
        elif ndim == 2:
            return pd.DataFrame(self.values.T, columns=self.axes[0], index=self.axes[1])
        else:
            raise ValueError('only dataframes with dimensions of 2 can be converted to pandas')

    DataFrame.to_pd = to_pd  # type:ignore
except ImportError:
    pass
