from typing import Dict, Tuple

import numpy as np
from numpy import linalg as LA


def get_points(user1: str, user2: str, prefs: Dict[str, Dict]) -> Tuple[nd.array]:
    common = set(prefs[user1]).intersection(prefs[user2])
    if not common:
        return 0

    # points
    p1 = np.array([prefs[user1][item] for item in common])
    p2 = np.array([prefs[user2][item] for item in common])
    return p1, p2


def similarity_distance(user1: str, user2: str, prefs: Dict[str, Dict]) -> float:

    p1, p2 = get_points(user1, user2, prefs)

    distance = np.sqrt(((p2 - p1) ** 2).sum())
    return 1.0 / (1.0 + distance)


def similarity_pearson(user1: str, user2: str, prefs: Dict[str, Dict]) -> float:

    p1, p2 = get_points(user1, user2, prefs)
    n = len(p1)

    def _var(x: nd.array, y: nd.array) -> float:
        return (x * y).sum() - (x.sum() * y.sum() / n)

    # how much variables change together
    cross = _var(p1, p2)
    # product of individual variations
    individual = np.sqrt(_var(p1, p1) * _var(p2, p2))
    return cross / individual
