# pylint:disable=unused-variable
# pylint:disable=unused-argument
# pylint:disable=redefined-outer-name

import numpy as np
import pytest

from hierarchical_cluster import BiNode, hcluster, print_cluster
from metrics import pearson_distance


@pytest.fixture()
def blogs_words_subset():
    blogs = ["Gothamist", "GigaOM", "Quick Online Tips"]
    words = ["china", "kids", "music", "yahoo"]
    table = [[0, 3, 3, 0], [6, 0, 0, 2], [0, 2, 2, 22]]

    return blogs, words, table


def test_bnodes():
    n0 = BiNode(0, [0] * 3)

    assert isinstance(n0.vec, np.ndarray)
    assert n0.is_leaf()


def test_hcluster(blogs_words_subset):
    blogs, words, table = blogs_words_subset

    btree: BiNode = hcluster(table, pearson_distance)

    assert not btree.is_leaf()
    assert btree.left
    assert btree.right

    # TODO: this hsould be automatic... with mipy??
    print()
    print_cluster(btree, labels=blogs, deep=0)
