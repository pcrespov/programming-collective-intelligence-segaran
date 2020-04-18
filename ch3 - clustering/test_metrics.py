# pylint:disable=unused-variable
# pylint:disable=unused-argument
# pylint:disable=redefined-outer-name


import numpy as np

from metrics import eucledian_distance, pearson_distance


def test_collinear():
    p0 = np.array([0.0] * 3)
    p1 = np.array([1.0] * 3)

    assert pearson_distance(p0, p1) == 0.0
    assert eucledian_distance(p0, p1) == np.sqrt(p1.size)
