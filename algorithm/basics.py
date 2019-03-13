from typing import List, Dict, Any, Optional

import numpy as np
from scipy.stats import norm

def remove_duplicate(array: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    current = [array[0][key] for key in keys]
    result = [array[0]]
    for elem in array[1:]:
        if [elem[key] for key in keys] != current:
            result.append(elem)
            current = [elem[key] for key in keys]
    return result

def d_prime(prediction: np.ndarray, truth: Optional[np.ndarray] = None) -> float:
    if truth is None:
        truth = np.ones_like(prediction, dtype=np.bool)
    hits = (prediction & truth).sum()
    misses = ((~prediction) & truth).sum()
    false_alarms = (prediction & (~truth)).sum()
    correct_rejections = (~(prediction | truth)).sum()

    Z = norm.ppf
    # Calculate hitrate and avoid d' infinity
    hitRate = hits / (hits + misses)
    if hitRate == 1:
        # Floors an ceilings are replaced by half hits and half FA's
        hitRate = 1 - 0.5 / (hits + misses)
    elif hitRate == 0:
        hitRate = 0.5 / (hits + misses)
    # Calculate false alarm rate and avoid d' infinity
    false_alarm_rate = false_alarms / (false_alarms + correct_rejections)
    if false_alarm_rate == 1:
        # Floors an ceilings are replaced by half hits and half FA's
        false_alarm_rate = 1 - 0.5 / (false_alarms + correct_rejections)
    elif false_alarm_rate == 0:
        false_alarm_rate = 0.5 / (false_alarms + correct_rejections)

    print(hitRate, false_alarm_rate)
    return Z(hitRate) - Z(false_alarm_rate)
