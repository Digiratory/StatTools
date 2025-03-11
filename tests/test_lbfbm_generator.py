from itertools import islice
import math
from scipy import stats
import numpy as np
import pytest

from StatTools.analysis.dpcca import dpcca
from StatTools.generators.lbfbm_generator import LBFBmGenerator, normalize

testdata = {
    "h_list": [i * 0.01 for i in range(50, 150, 10)],
    "base_list": [1.2],
}

SCALES = np.array([2**i for i in range(3, 9)])
STEP = 1
TARGET_LEN = 4000


def generate_trajectory(generator: LBFBmGenerator, target_len: int) -> np.ndarray:
    """
    Generates a trajectory of a specified length from the given generator.

    Args:
        generator (LBFBmGenerator): The generator to use for generating the trajectory.
        target_len (int): The desired length of the trajectory.

    Returns:
        np.ndarray: The generated trajectory as an array.
    """
    trj = []
    for value in islice(generator, target_len):
        trj.append(value)
    return np.array(trj)


def calculate_hurst_exponent(
    trajectory: np.ndarray, scales: np.ndarray, step: float
) -> float:
    """
    Calculates the Hurst exponent from a given trajectory using the DPCCA algorithm.

    Args:
        trajectory (np.ndarray): The input trajectory to calculate the Hurst exponent for.
        scales (np.ndarray): An array of scales to use in the calculation.
        step (float): The step size used in the DPCCA algorithm.

    Returns:
        float: The calculated Hurst exponent.

    Notes:
        This function uses the dpcca module from StatTools.analysis.dpcca to perform the
        calculations. It normalizes the input trajectory, applies the DPCCA algorithm,
        and then calculates the Hurst exponent using linear regression.
    """
    signal_z = normalize(trajectory)
    _, _, f_z, s_z = dpcca(signal_z, 2, step, scales, processes=1, n_integral=0)
    f_z = np.sqrt(f_z)
    f_z /= f_z[0]
    res = stats.linregress(np.log(s_z), np.log(f_z)).slope
    return res


def get_test_h(
    base: float,
    filter_len: int,
    h: float,
    scales: np.ndarray,
    step: int,
    target_len: int,
) -> float:
    """
    Calculates the Hurst exponent for the generated trajectory.

    Parameters:
        base: The base of the number system for bins
        filter_len: Filter length
        h: The specified Hurst exponent
        scales: Scales for analysis
        step: The step for analysis

    Returns:
        Calculated Hurst exponent (h_gen)
    """
    generator = LBFBmGenerator(h, filter_len, base)
    trj = generate_trajectory(generator, target_len)
    res = calculate_hurst_exponent(trj, scales, step)
    return res


@pytest.mark.parametrize("h", testdata["h_list"])
@pytest.mark.parametrize("base", testdata["base_list"])
def test_lbfbm_generator(h: float, base: float):
    """
    It tests the generator for compliance with the specified Hurst exponent.

    Parameters:
        h: The specified Hurst exponent
        base: The base of the number system for bins
    """
    threshold = 0.10
    times = 10
    filter_len = int(math.log(TARGET_LEN, base))
    mean_difference = 0
    for i in range(times):
        h_gen = get_test_h(base, filter_len, h, SCALES, STEP, TARGET_LEN)

        mean_difference += abs(h_gen - h) / h
    mean_difference /= times
    assert (
        mean_difference <= threshold
    ), f"Diff between h and h_gen exceeds {threshold * 100}%: h={h}, mean diff={mean_difference * 100:.2f}%"
