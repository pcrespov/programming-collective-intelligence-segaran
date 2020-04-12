from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Tuple

import numpy as np

# from numpy import linalg as LA


def get_points(user1: str, user2: str, prefs: Dict[str, Dict]) -> Tuple[np.array]:
    if user1 not in prefs or user2 not in prefs:
        raise ValueError

    common = sorted(set(prefs[user1]).intersection(prefs[user2]))
    if not common:
        raise ValueError

    # points
    p1 = np.array([prefs[user1][item] for item in common])
    p2 = np.array([prefs[user2][item] for item in common])
    return p1, p2


def similarity_distance(user1: str, user2: str, *, prefs: Dict[str, Dict]) -> float:

    try:
        p1, p2 = get_points(user1, user2, prefs)
    except ValueError:
        return 0

    distance = np.sqrt(((p2 - p1) ** 2).sum())
    return 1.0 / (1.0 + distance)


def similarity_pearson(user1: str, user2: str, *, prefs: Dict[str, Dict]) -> float:

    try:
        p1, p2 = get_points(user1, user2, prefs)
    except ValueError:
        return 0

    dim = len(p1)
    if dim <= 1:  # TODO: checkcase dim == 1! => 0/0
        return 0

    def _var(x: np.array, y: np.array) -> float:
        return (x * y).sum() - (x.sum() * y.sum() / dim)

    # how much variables change together
    cross = _var(p1, p2)
    # product of individual variations
    individual = np.sqrt(_var(p1, p1) * _var(p2, p2))

    return cross / individual


SCORE_INDEX = 0


def eval_top_matches(
    user: str, *, count: int = 5, similarity=similarity_pearson, prefs: Dict[str, Dict]
) -> List[Tuple[float, str]]:
    """ Returns top matches for a given user sorted by similarity """
    scores = [
        (similarity(user, other, prefs=prefs), other)
        for other in prefs
        if other != user
    ]

    scores.sort(key=itemgetter(SCORE_INDEX), reverse=True)
    return scores[:count]


def get_recommendations(
    user: str, *, count: int = 5, similarity=similarity_pearson, prefs: Dict[str, Dict]
) -> List[Tuple[float, str]]:
    """  Returns a list of recommendations of items the user has not scored """

    def _nill():
        return 0

    rating = defaultdict(_nill)
    totals = defaultdict(_nill)

    user_rating = prefs.get(user, {})

    for other in prefs:
        if other != user:
            sim = similarity(user, other, prefs=prefs)

            if sim > 0:
                for name, score in prefs[other].items():
                    if name not in user_rating:
                        rating[name] += sim * score
                        totals[name] += sim

    # normalize and encapsulate
    scores = [(rating[name] / totals[name], name) for name in rating]

    scores.sort(key=itemgetter(SCORE_INDEX), reverse=True)

    return scores[:count]
