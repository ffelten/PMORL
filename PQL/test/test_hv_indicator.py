import numpy as np
import pytest

from PQL.utils.hv_indicator import MaxHVHeuristic

test_data = [
    (MaxHVHeuristic(ref_point=np.array([10.0, 0.0])), np.array([[[1, -0.3]], [[2, -0.2]], [[0.2, -0.5]]]), 2),
    (MaxHVHeuristic(ref_point=np.array([10.0, 10.0])), np.array([[[1, 2]], [[2, 1], [1, 1]], [[0, 2]]]), 2),
    (MaxHVHeuristic(ref_point=np.array([10.0, 10.0, 10.0])), np.array([[[1, 2, 2]], [[2, 1, 2]], [[0, 2, 2]]]), 2)
]

@pytest.mark.parametrize("hv, qsets, expected_action", test_data)
def test_hv(hv, qsets, expected_action):
    assert hv.compute(qsets) == expected_action