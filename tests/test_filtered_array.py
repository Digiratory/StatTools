import numpy as np
import pytest

from StatTools.analysis.dfa import DFA
from StatTools.generators.base_filter import Filter, FilteredArray

testdata = {
    "h_list": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.0],
    "length_list": [4000],
}

SCALES = np.array([2**i for i in range(3, 9)])
TARGET_LEN = 2000


def calculate_hurst_exponent(trajectory: np.ndarray) -> float:
    """
    Calculates the Hurst exponent from a given trajectory using the DFA algorithm.

    Args:
        trajectory (np.ndarray): The input trajectory to calculate the Hurst exponent for.

    Returns:
        float: The calculated Hurst exponent.
    """
    return DFA(trajectory).find_h()


def get_test_h(h: float, length: int) -> float:
    """
    Calculates the Hurst exponent for the generated trajectory.

    Parameters:
        h: The specified Hurst exponent
        length: Length of the generated trajectory

    Returns:
        Calculated Hurst exponent (h_gen)
    """
    generator = FilteredArray(h, length)
    trajectory = generator.generate(n_vectors=1, threads=1, h_limit=0.05)

    return calculate_hurst_exponent(trajectory)


@pytest.mark.parametrize("h", testdata["h_list"])
@pytest.mark.parametrize("length", testdata["length_list"])
def test_filtered_array_generator(h: float, length: int):
    """
    Test the FilteredArray class for compliance with the specified Hurst exponent.

    Parameters:
        h: The specified Hurst exponent
        length: Length of the generated trajectory
    """
    threshold = 0.10
    times = 10
    mean_difference = 0

    for _ in range(times):
        h_gen = get_test_h(h, length)
        mean_difference += abs(h_gen - h) / h

    mean_difference /= times
    assert (
        mean_difference <= threshold
    ), f"Diff between h and h_gen exceeds {threshold * 100}%: h={h}, mean diff={mean_difference * 100:.2f}%"


if __name__ == "__main__":
    for h in testdata["h_list"]:
        for length in testdata["length_list"]:
            test_filtered_array_generator(h, length)
