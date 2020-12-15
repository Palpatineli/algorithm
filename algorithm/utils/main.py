from typing import Dict, List, TypeVar, Callable, Iterator, Tuple
from itertools import product
import numpy as np
from joblib import Parallel, delayed

def iter_tree(case_ids: Dict[str, List[Dict[str, str]]]) -> Iterator[Dict[str, str]]:
    for group_str, group in case_ids.items():
        for session in group:
            session['group'] = group_str
            yield {"group": group_str, **session}

T = TypeVar("T")
E = TypeVar("E")

def map_tree(fn: Callable[[T], E], tree: Dict[str, List[T]]) -> Dict[str, List[E]]:
    output: Dict[str, List[E]] = dict()
    for group_str, group in tree.items():
        output[group_str] = [fn(x) for x in group]
    return output

def map_tree_parallel(fn: Callable[[T], E], tree: Dict[str, List[T]], verbose: int = 0) -> Dict[str, List[E]]:
    output: Dict[str, List[Tuple[int, E]]] = dict()
    pool = Parallel(-1, prefer="threads", verbose=verbose)
    flattened: List[Tuple[str, int, T]] = [(group_str, idx, load) for group_str, group in tree.items()
                                           for idx, load in enumerate(group)]

    results = pool((delayed(lambda x: (x[0], x[1], fn(x[2])))(value)) for value in flattened)
    for group_str, idx, result in results:
        if group_str in output:
            output[group_str].append((idx, result))
        else:
            output[group_str] = [(idx, result)]
    return {group_str: [value for _, value in sorted(group)] for group_str, group in output.items()}

def map_table(fn: Callable[..., E], *axes: List[List[T]], n: int = 10) -> np.ndarray:
    results = Parallel(n, prefer="threads")((delayed(fn)(*value)) for value in product(*axes))
    return np.asarray(results).reshape(*(len(axe) for axe in axes))

def zip_tree(*trees: Dict[str, List[T]]) -> Dict[str, List[List[T]]]:
    output: Dict[str, List[List[T]]] = dict()
    for group_str, group in trees[0].items():
        output[group_str] = [[x, *y] for x, *y in zip(group, *[tree[group_str] for tree in trees[1:]])]
    return output

def flatten(groups: Dict[str, List[T]]) -> Dict[str, T]:
    return {f"{group}-{idx}": x for group, y in groups.items() for idx, x in enumerate(y)}

def unflatten(flat: Dict[str, T]) -> Dict[str, List[T]]:
    numbers = [key.split('-') for key in flat.keys()]
    groups, group_counts = np.unique(next(iter(zip(*numbers))), return_counts=True)
    results: Dict[str, List[T]] = dict()
    for group, no in zip(groups, group_counts):
        results[group] = list()
        for idx in range(no):
            results[group].append(flat[f"{group}-{idx}"])
    return results

def ctake(ar: np.ndarray, indices: np.ndarray, axis=None) -> np.ndarray:
    if axis is None:
        mask = np.ones(ar.size, dtype=np.bool_)
        mask[indices] = False
        return ar.take(mask)
    else:
        mask = np.ones_like(ar, dtype=np.bool_)
        np.put_along_axis(mask, indices, False, axis)
        return ar.take(mask, axis)

def rotate45(x):
    size = x.shape[0]
    symmetry = np.allclose(x, x.T)
    res = list()
    for diff in range(size - 1, -1, -1):
        off_diagonal = [x[idx, idx + diff] for idx in range(size - diff)]
        res.append(off_diagonal)
    if symmetry:
        return res + res[-2::-1]
    else:
        for diff in range(1, size):
            res.append([x[idx, idx - diff] for idx in range(diff, size)])
        return res

