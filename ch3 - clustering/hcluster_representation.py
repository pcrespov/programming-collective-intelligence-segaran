import logging
from typing import List

from PIL import Image, ImageDraw, ImageFont

from hcluster_algoritms import BiNode

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

WHITE = (255,) * 3
RED = (255, 0, 0)
BLACK = (0,) * 3


def print_cluster(node: BiNode, labels: List[str], deep: int = 0):
    indent = " " * deep
    if not node.is_leaf():
        print(indent, "+", f"[{node.distance:3.2f}]")
        print_cluster(node.left, labels, deep + 1)
        print_cluster(node.right, labels, deep + 1)
    else:
        print(indent, labels[node.uid], deep + 1)


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
            font = ImageFont.truetype("Arial")
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
