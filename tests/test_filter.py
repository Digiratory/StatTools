import numpy as np
import pytest

from StatTools.generators.base_filter import Filter

testdata = {
    "target_mean": [0.5, 0.6, 0.7, 0.8, 0.9],
    "target_std": [0.5, 0.6, 0.7, 0.8, 0.9],
    "h": [0.5, 0.6, 0.7, 0.8, 0.9],
    "length": [4000],
}

SCALES = np.array([2**i for i in range(3, 9)])
TARGET_LEN = 2000


@pytest.mark.parametrize("h", testdata["h"])
@pytest.mark.parametrize("length", testdata["length"])
@pytest.mark.parametrize("target_std", testdata["target_std"])
@pytest.mark.parametrize("target_mean", testdata["target_mean"])
def test_filter_mean_std(h, length, target_std, target_mean):
    """
    Test that the generated data has the specified mean and standard deviation.
    """
    generator = Filter(h, length, set_mean=target_mean, set_std=target_std)
    trajectory = generator.generate(n_vectors=1)

    actual_mean = np.mean(trajectory)
    actual_std = np.std(trajectory, ddof=1)

    assert (
        abs(actual_mean - target_mean) < 0.001
    ), f"Mean deviation too large: expected {target_mean}, got {actual_mean}"
    assert (
        abs(actual_std - target_std) < 0.001
    ), f"Std deviation too large: expected {target_std}, got {actual_std}"
