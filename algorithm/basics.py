from typing import List, Dict, Any


def remove_duplicate(array: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    current = [array[0][key] for key in keys]
    result = [array[0]]
    for elem in array[1:]:
        if [elem[key] for key in keys] != current:
            result.append(elem)
            current = [elem[key] for key in keys]
    return result
