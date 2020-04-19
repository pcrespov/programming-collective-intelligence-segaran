from __future__ import annotations

# SEE https://www.python.org/dev/peps/pep-0563/

import logging

from typing import Any, List, Optional

import attr
import numpy as np

from metrics import pearson_distance

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@attr.s(auto_attribs=True, frozen=True)
class BiNode:
    uid: int
    vec: np.array = attr.ib(converter=np.array)

    # binary tree children
    left: Optional[BiNode] = None
    right: Optional[BiNode] = None
    distance: Optional[float] = None  # distance(left, right)

    @classmethod
    def from_nodes(cls, uid: int, left: BiNode, right: BiNode, distance: float):
        return BiNode(uid, 0.5 * (left.vec + right.vec), left, right, distance)

    def is_leaf(self):
        return self.uid >= 0


def hcluster(table: List[List[Any]], distance_fun=pearson_distance) -> BiNode:

    cached_distance = {}

    def get_distance(src: BiNode, dst: BiNode) -> float:
        nonlocal cached_distance
        # simple cache commutative
        key = tuple(sorted([src.uid, dst.uid]))
        try:
            result = cached_distance[key]
        except KeyError:
            result = distance_fun(src.vec, dst.vec)
            cached_distance[key] = result
        return result

    # original nodes identified with positive uids
    items: List[BiNode] = [BiNode(i, row[:]) for i, row in enumerate(table)]

    cluster_counter: int = 0
    while len(items) > 1:
        i0, j0 = 0, 1
        dmin = get_distance(items[i0], items[j0])

        for i, src in enumerate(items):
            for j, dst in enumerate(items[i + 1 :], start=i + 1):
                dij = get_distance(src, dst)
                if dij < dmin:
                    i0, j0, dmin = i, j, dij

        # cluster identified with negavite uid
        cluster_counter += 1
        cluster = BiNode.from_nodes(
            uid=-cluster_counter, left=items[i0], right=items[j0], distance=dmin,
        )
        del items[j0]  # first largest
        del items[i0]
        items.append(cluster)

    return items[0] if items else None
