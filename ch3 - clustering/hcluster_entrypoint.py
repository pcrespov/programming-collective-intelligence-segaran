import logging
import sys

import numpy as np

from generate_feed_vector import load_data
from hcluster_algoritms import hcluster
from hcluster_representation import print_cluster

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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
