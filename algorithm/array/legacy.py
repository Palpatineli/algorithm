"""Legacy function for handling custome DataFrame."""
from typing import Dict, Any, Sequence, List, Iterable, Union
import numpy as np

DataFrame = Dict[str, np.ndarray]


def generate_axis_name(idx: int) -> str:
    """generate axis name in the sequence of x, y, z, xx, yy, zz, xxx, ..."""
    return chr(idx % 3 + 120) * (idx // 3 + 1)


def group_mean(data: DataFrame, groupings: np.ndarray) -> DataFrame:
    """group data by groupings starting from the axis 1, and take the means on those axes
    Args:
        data: DataFrame with 'data' a n-dim array
        groupings: a list of groups, len(groupings[i]) == data['data'].shape[i + 1]
            np.unique(groupings[0]) will be returns['y'], np.unique(groupings[1]) will be returns['z']
    Returns:
        a n-dim DataFrame that is meand by group from axes 1-m
    """
    source = data['data']
    uniques, unique_indices = zip(*[np.unique(x, return_inverse=True) for x in groupings])
    target_dims = [source.shape[0], *map(len, uniques), *source.shape[len(uniques) + 1:]]
    group_seq = np.vstack([x.ravel() for x in np.meshgrid(*[np.arange(len(x)) for x in uniques], indexing='ij')]).T
    indices = np.vstack([x.ravel() for x in np.meshgrid(*[x for x in unique_indices], indexing='ij')]).T
    flat_source = source.reshape((source.shape[0], -1, *source.shape[len(uniques) + 1:]))
    mask = np.all(np.expand_dims(group_seq, 1) == np.expand_dims(indices, 0), -1)
    target = np.swapaxes(np.array([flat_source.compress(row, 1).mean(1) for row in mask]), 0, 1).reshape(target_dims)
    return {**data, 'data': target, **dict(zip([generate_axis_name(x) for x in range(1, len(uniques) + 1)], uniques))}


def group_stack(data: DataFrame, groupings: np.ndarray) -> DataFrame:
    uniques = [np.sort(np.unique(grouping)) for grouping in groupings]
    grids = np.meshgrid(*[np.arange(len(x)) for x in uniques], indexing='ij')
    """:type: Tuple[np.ndarray]"""
    sequence = np.vstack([x.ravel() for x in grids])
    values = np.array([x[y] for x, y in zip(uniques, sequence)])
    mask = np.all(np.expand_dims(values, 2) == np.expand_dims(groupings, 1), 0)
    # axis_0 = condition axis_1 = sequence
    lengths = [len(data['x'])] + [len(x) for x in uniques] + [-1]
    stack = np.swapaxes(np.stack([data['data'][:, plane] for plane in mask]), 0, 1).reshape(lengths)
    extra_dimensions = dict(zip([chr(x) for x in 121 + np.arange(len(uniques))], uniques))
    return {'x': data['x'], 'data': stack, **extra_dimensions}


def group_by(data: DataFrame, grouping: np.ndarray) -> Dict[Any, DataFrame]:
    new_ys = np.sort(np.unique(grouping))
    y, value = data['y'], data['data']
    result = dict()
    for key in new_ys:
        mask = grouping == key
        result[key] = {**data, 'y': y[mask], 'data': value[:, mask]}
    return result


def concatenate(df_list: List[DataFrame]) -> DataFrame:
    assert all_equal(a['y'] for a in df_list)
    return {'x': np.concatenate([a['x'] for a in df_list]), 'y': df_list[0]['y'],
            'data': np.vstack([a['data'] for a in df_list])}


def all_equal(iterable: Iterable) -> bool:
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(np.array_equal(first, rest) for rest in iterator)


__order_dict, __name_dict = {'x': 0, 'y': 1, 'z': 2, 't': 3}, ('x', 'y', 'z', 't')


def df_remove(data: DataFrame, idx: Any, axis: Union[int, str] = 0) -> DataFrame:
    """remove hyperplanes from data, by the given idx and axis (order or name) """
    if isinstance(axis, str):
        axis_order = __order_dict[axis]
    else:
        axis_order, axis = axis, __name_dict[axis]
    if not isinstance(idx, (Sequence, np.ndarray)):
        numerical_idx = [data[axis].index(idx)]
    else:
        idx_dict = {x: idx2 for idx2, x in enumerate(data[axis])}
        numerical_idx = [idx_dict[x] for x in idx]
    mask = np.ones(data[axis].shape[0], dtype=np.bool_)
    mask[numerical_idx] = False
    return {**data, axis: data[axis][mask], 'data': data['data'].compress(mask, axis=axis_order)}
##
