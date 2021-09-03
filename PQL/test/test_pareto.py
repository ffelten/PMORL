import numpy as np
import pytest
from PQL.utils import pareto

test_data = [
    ([[1], [2], [0]], [[2]]),
    ([[1, 3], [2, 4], [0, 5]], [[2, 4], [0, 5]]),
    ([[1, 3, 10], [2, 4, 2], [0, 5, 4]], [[1, 3, 10], [2, 4, 2], [0, 5, 4]]),
    ([[1, -1], [1, -2]], [[1, -1]]),
    ([[2, -1], [1, -2]], [[2, -1]]),
    ([[1.0, 3.2, 10.1222], [2.332, 4.321, 2.432], [0, 5, 4]], [[1.0, 3.2, 10.1222], [2.332, 4.321, 2.432], [0, 5, 4]])
]
@pytest.mark.parametrize("points, expected_front", test_data)
def test_pareto_efficient(points, expected_front):
    np.testing.assert_array_almost_equal(pareto.pareto_efficient(points), expected_front)

