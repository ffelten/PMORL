import numpy as np
import pytest
from PQL.utils.QSet import QSet

qset = QSet()
qset.set = [np.array([-4., 1.]), np.array([-2., 1.]), np.array([-3., 1.])]

test_data = [
    ([np.array([-4., 1.]), np.array([-2., 1.]), np.array([-3., 1.])], [-3., 1.]),
    ([np.array([-2.5, 1.5]), np.array([-2., 1.]), np.array([-3., 2.])], [-2.5, 1.5]),
    ([np.array([-4., 2.]), np.array([-2., 1.])], [-3., 1.5])
]

@pytest.mark.parametrize("qset_data, expected_means", test_data)
def test_means(qset_data, expected_means):
    qset = QSet()
    qset.set = qset_data
    np.testing.assert_array_almost_equal(qset.compute_means(), expected_means)



test_data = [
    ([np.array([-4., 1.]), np.array([-2., 1.]), np.array([-3., 1.])], [-4., 1.]),
    ([], []),
    ([np.array([-2.5, 1.5]), np.array([-2., 1.]), np.array([-3., 2.])], [-3., 1.]),
    ([np.array([-4., 2.]), np.array([-2., 1.])], [-4., 1.])
]

@pytest.mark.parametrize("qset_data, expected_mins", test_data)
def test_mins(qset_data, expected_mins):
    qset = QSet()
    qset.set = qset_data
    np.testing.assert_array_almost_equal(qset.compute_mins(), expected_mins)

test_data = [
    ([np.array([-4., 1.]), np.array([-2., 1.]), np.array([-3., 1.])], [-2., 1.]),
    ([], []),
    ([np.array([-2.5, 1.5]), np.array([-2., 1.]), np.array([-3., 2.])], [-2., 2.]),
    ([np.array([-4., 2.]), np.array([-2., 1.])], [-2., 2.])
]

@pytest.mark.parametrize("qset_data, expected_maxs", test_data)
def test_maxs(qset_data, expected_maxs):
    qset = QSet(qset_data)
    np.testing.assert_array_almost_equal(qset.compute_maxs(), expected_maxs)