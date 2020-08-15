import numpy as np


def eucledian_distance(p1: np.array, p2: np.array, *, normalize=False) -> float:
    distance = np.sqrt(((p2 - p1) ** 2).sum())
    if normalize:
        # [0,1]
        return 1.0 / (1.0 + distance)
    return distance


def pearson_distance(p1: np.array, p2: np.array) -> float:
    dim = p1.size
    assert dim >= 1

    def _var(x: np.array, y: np.array) -> float:
        return (x * y).sum() - (x.sum() * y.sum() / dim)

    # product of individual variations
    self_vars = np.sqrt(_var(p1, p1) * _var(p2, p2))

    if self_vars == 0.0:
        return 0

    # how much variables change together
    cross_vars = _var(p1, p2)

    # match = 1, no-relation=0, opposite relation<0
    similar = cross_vars / self_vars

    # match = distance = 0
    distance = 1.0 - similar
    return distance
