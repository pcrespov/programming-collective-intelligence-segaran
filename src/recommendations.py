from operator import itemgetter
from typing import Dict, Tuple, List

import numpy as np
from numpy import linalg as LA


def get_points(user1: str, user2: str, prefs: Dict[str, Dict]) -> Tuple[np.array]:
    common = sorted(set(prefs[user1]).intersection(prefs[user2]))
    if not common:
        return 0

    # points
    p1 = np.array([prefs[user1][item] for item in common])
    p2 = np.array([prefs[user2][item] for item in common])
    return p1, p2


def similarity_distance(user1: str, user2: str, *, prefs: Dict[str, Dict]) -> float:

    p1, p2 = get_points(user1, user2, prefs)

    distance = np.sqrt(((p2 - p1) ** 2).sum())
    return 1.0 / (1.0 + distance)


def similarity_pearson(user1: str, user2: str, *, prefs: Dict[str, Dict]) -> float:

    p1, p2 = get_points(user1, user2, prefs)
    n = len(p1)

    def _var(x: np.array, y: np.array) -> float:
        return (x * y).sum() - (x.sum() * y.sum() / n)

    # how much variables change together
    cross = _var(p1, p2)
    # product of individual variations
    individual = np.sqrt(_var(p1, p1) * _var(p2, p2))

    return cross / individual


SIMILARITY_INDEX = 0


def eval_top_matches(
    user: str, *, count: int = 5, similarity=similarity_pearson, prefs: Dict[str, Dict]
) -> List[Tuple[float, str]]:
    """ Returns top matches for a given user sorted by similarity """
    scores = [
        (similarity(user, other, prefs=prefs), other)
        for other in prefs
        if other != user
    ]

    scores.sort(key=itemgetter(SIMILARITY_INDEX), reverse=True)
    return scores[:count]
