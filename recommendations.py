from typing import Dict

import numpy as np
from numpy import linalg as LA

critics = {
    "Lisa Rose": {
        "Lady in the Water": 2.5,
        "Snakes on a Plane": 3.5,
        "Just My Luck": 3.0,
        "Superman Returns": 3.5,
        "You, Me and Dupree": 2.5,
        "The Night Listener": 3.0,
    },
    "Gene Seymour": {
        "Lady in the Water": 3.0,
        "Snakes on a Plane": 3.5,
        "Just My Luck": 1.5,
        "Superman Returns": 5.0,
        "The Night Listener": 3.0,
        "You, Me and Dupree": 3.5,
    },
    "Michael Phillips": {
        "Lady in the Water": 2.5,
        "Snakes on a Plane": 3.0,
        "Superman Returns": 3.5,
        "The Night Listener": 4.0,
    },
    "Claudia Puig": {
        "Snakes on a Plane": 3.5,
        "Just My Luck": 3.0,
        "The Night Listener": 4.5,
        "Superman Returns": 4.0,
        "You, Me and Dupree": 2.5,
    },
    "Mick LaSalle": {
        "Lady in the Water": 3.0,
        "Snakes on a Plane": 4.0,
        "Just My Luck": 2.0,
        "Superman Returns": 3.0,
        "The Night Listener": 3.0,
        "You, Me and Dupree": 2.0,
    },
    "Jack Matthews": {
        "Lady in the Water": 3.0,
        "Snakes on a Plane": 4.0,
        "The Night Listener": 3.0,
        "Superman Returns": 5.0,
        "You, Me and Dupree": 3.5,
    },
    "Toby": {
        "Snakes on a Plane": 4.5,
        "You, Me and Dupree": 1.0,
        "Superman Returns": 4.0,
    },
}


names = set(critics.keys())
movies = set().union(*(critics[n].keys() for n in names))


def get_points(user1: str, user2: str, prefs: Dict[str, Dict]) -> Tuple[nd.array]:
    common = set(prefs[user1]).intersection(prefs[user2])
    if not common:
        return 0

    # points
    p1 = np.array([prefs[user1][item] for item in common])
    p2 = np.array([prefs[user2][item] for item in common])
    return p1, p2


def similarity_distance(
    user1: str, user2: str, prefs: Dict[str, Dict] = critics
) -> float:

    p1, p2 = get_points(user1, user2, prefs)

    distance = np.sqrt(((p2 - p1) ** 2).sum())
    return 1.0 / (1.0 + distance)

def similarity_pearson(
    user1: str, user2: str, prefs: Dict[str, Dict] = critics
) -> float:

    p1, p2 = get_points(user1, user2, prefs)
    n = len(p1)

    def _var(x: nd.array, y: nd.array) -> float:
        return (x * y).sum() - (x.sum() * y.sum() / n)

    # how much variables change together
    cross = _var(p1, p2)
    # product of individual variations
    individual = np.sqrt(_var(p1, p1) * _var(p2, p2))
    return cross / individual
