import numpy as np
import pytest

from reconstruct_climate_indices.statistics import my_mean


# Test data
data = [[1, 2, 3, 4], [4, 3, np.nan, 1]]
weights = [[1, 1, 1, 1], [2, 2, 2, 2]]


@pytest.fixture
def example_array_01():
    data = np.array(
        [[1, 2, 3, 4], [4, 3, np.nan, 1]],
    )
    return data


def test_my_mean(example_array_01):
    result = my_mean(example_array_01)
    np.testing.assert_allclose(np.array([np.nan]), result)
