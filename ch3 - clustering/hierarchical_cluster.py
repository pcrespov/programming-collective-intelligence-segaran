from __future__ import annotations

import logging
import sys
from typing import Any, List, Optional

import attr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generate_feed_vector import load_data
from metrics import pearson_distance

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# SEE https://www.python.org/dev/peps/pep-0563/


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
        return self.uid < 0


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


def print_cluster(node: BiNode, labels: List[str], deep: int = 0):
    indent = " " * deep
    if node.is_leaf():
        print(indent, "+", f"[{node.distance:3.2f}]")
        print_cluster(node.left, labels, deep + 1)
        print_cluster(node.right, labels, deep + 1)
    else:
        print(indent, labels[node.uid], deep + 1)


WHITE = (255,) * 3
RED = (255, 0, 0)
BLACK = (0,) * 3

def draw_dendrogram(root: BiNode, labels=List[str]):
    def get_height(node: BiNode) -> int:
        if not node.left and not node.right:
            return 1
        return get_height(node.left) + get_height(node.right)

    def get_depth(node: BiNode) -> int:
        if not node.left and not node.right:
            return 0
        return node.distance + max(get_depth(node.left), get_depth(node.right))

    def draw_node(node, x, y, scaling, labels, draw) -> None:
        if node.uid < 0:
            h1 = get_height(node.left) * 20
            h2 = get_height(node.right) * 20

            top = y - (h1 + h2) / 2
            bottom = y + (h1 + h2) / 2

            ll = node.distance * scaling

            y1 = top + h1 / 2
            y2 = bottom - h2 / 2

            # vertical
            draw.line((x, y1, x, y2), fill=RED)

            # horizonal for for left
            draw.line((x, y1, x + ll, y1), fill=RED)

            # horizontal line for right
            draw.line((x, y2, x + ll, y2), fill=RED)

            draw_node(node.left, x + ll, y1, scaling, labels, draw)
            draw_node(node.right, x + ll, y2, scaling, labels, draw)
        else:
            font = ImageFont.truetype('Arial')
            args = [(x + 5, y - 7), labels[node.uid], BLACK]
            try:
                draw.text(*args, font=font)
            except UnicodeEncodeError as ee:
                print(ee, " with ", args[1])
                args[1] = str(node.uid)
                draw.text(*args)


    h = get_height(root) * 20
    w = 1200
    depth = get_depth(root)

    scaling = float(w - 150) / depth  # so it fits in width

    img: Image.Image = Image.new("RGB", (w, h), WHITE)
    draw = ImageDraw.Draw(img)

    draw.line((0, h / 2, 10, h / 2), fill=RED)

    draw_node(root, 10, h / 2, scaling, labels, draw)

    return img


def main(revert: bool = False, verbose: bool = True):
    data = load_data()
    words = data["words"]
    blogs = [blog["name"] for blog in data["blogs"]]
    table = np.array([blog["wc"] for blog in data["blogs"]])

    log.info(" # words: %3d", len(words))
    log.info(" # blogs: %3d", len(blogs))

    labels = blogs
    if revert:
        table = table.T
        labels = words

    root = hcluster(table)
    if verbose:
        print_cluster(root, labels)
    return root, labels


if __name__ == "__main__":
    main(revert=len(sys.argv) > 1)
