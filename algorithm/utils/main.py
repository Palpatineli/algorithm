from typing import Dict, List, TypeVar, Callable, Iterator, Tuple
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

def zip_tree(tree0: Dict[str, List[T]], tree1: Dict[str, List[E]]) -> Dict[str, List[Tuple[T, E]]]:
    output: Dict[str, List[Tuple[T, E]]] = dict()
    for group_str, group in tree0.items():
        output[group_str] = [(x, y) for x, y in zip(group, tree1[group_str])]
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
