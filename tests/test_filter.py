import numpy as np
import pytest

from StatTools.generators.base_filter import Filter
from StatTools.analysis.dfa import DFA

testdata = {
    "target_mean": [0.5, 0.7, 0.9],
    "target_std": [0.5, 0.7, 0.9],
    "h": [0.5, 0.6, 0.7, 0.8, 0.9],
    "length": [6000],
}

SCALES = np.array([2**i for i in range(3, 9)])
TARGET_LEN = 2000


@pytest.mark.parametrize("h", testdata["h"])
@pytest.mark.parametrize("length", testdata["length"])
@pytest.mark.parametrize("target_std", testdata["target_std"])
@pytest.mark.parametrize("target_mean", testdata["target_mean"])
def test_filter_generator(h, length, target_std, target_mean):
    """
    Test that the generated data has the specified mean and standard deviation.
    """
    generator = Filter(h, length, set_mean=target_mean, set_std=target_std)
    trajectory = list(generator)

    actual_mean = np.mean(trajectory)
    actual_std = np.std(trajectory, ddof=1)
    actual_h = DFA(trajectory).find_h()

    assert (
        abs(actual_mean - target_mean) < 0.001
    ), f"Mean deviation too large: expected {target_mean}, got {actual_mean}"
    assert (
        abs(actual_std - target_std) < 0.001
    ), f"Std deviation too large: expected {target_std}, got {actual_std}"
    assert abs(actual_h - h) < (
        h * 0.15
    ), f"Hurst deviation too large: expected {h}, got {actual_h}"


@pytest.mark.parametrize("h", testdata["h"])
@pytest.mark.parametrize("length", testdata["length"])
@pytest.mark.parametrize("target_std", testdata["target_std"])
@pytest.mark.parametrize("target_mean", testdata["target_mean"])
def test_filter(h, length, target_std, target_mean):
    """
    Test that the generated data has the specified mean and standard deviation.
    """
    generator = Filter(h, length, set_mean=target_mean, set_std=target_std)
    trajectory = generator.generate(n_vectors=1)

    actual_mean = np.mean(trajectory)
    actual_std = np.std(trajectory, ddof=1)
    actual_h = DFA(trajectory).find_h()

    assert (
        abs(actual_mean - target_mean) < 0.001
    ), f"Mean deviation too large: expected {target_mean}, got {actual_mean}"
    assert (
        abs(actual_std - target_std) < 0.001
    ), f"Std deviation too large: expected {target_std}, got {actual_std}"
    assert abs(actual_h - h) < (
        h * 0.15
    ), f"Hurst deviation too large: expected {h}, got {actual_h}"


@pytest.mark.parametrize("h", testdata["h"])
@pytest.mark.parametrize("length", testdata["length"])
@pytest.mark.parametrize("target_std", testdata["target_std"])
@pytest.mark.parametrize("target_mean", testdata["target_mean"])
def test_filter_2d(h, length, target_std, target_mean):
    """
    Test that the generated data has the specified mean and standard deviation.
    """
    generator = Filter(h, length, set_mean=target_mean, set_std=target_std)
    trajectories = generator.generate(n_vectors=3)

    for i in range(3):
        trajectory = trajectories[i]

        actual_mean = np.mean(trajectory)
        actual_std = np.std(trajectory, ddof=1)
        actual_h = DFA(trajectory).find_h()

        assert (
            abs(actual_mean - target_mean) < 0.001
        ), f"Mean deviation too large: expected {target_mean}, got {actual_mean}"
        assert (
            abs(actual_std - target_std) < 0.001
        ), f"Std deviation too large: expected {target_std}, got {actual_std}"
        assert abs(actual_h - h) < (
            h * 0.15
        ), f"Hurst deviation too large: expected {h}, got {actual_h}"
