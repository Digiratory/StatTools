from itertools import islice
import math
from scipy import stats
import numpy as np
import pytest

from StatTools.analysis.dpcca import dpcca
from StatTools.generators.hurst_generator import HurstGenerator, normalize

testdata = {
    "h_list": [i * 0.01 for i in range(50, 150, 10)],
    "base_list": [1.2],
}

S = np.array([2**i for i in range(3, 9)])
STEP = 1
TARGET_LEN = 4000


def get_test_h(
    base: float, filter_len: int, h: float, S: np.ndarray, step: int
) -> float:
    """
    Calculates the Hurst exponent for the generated trajectory.

    Parameters:
        base: The base of the number system for bins
        filter_len: Filter length
        h: The specified Hurst exponent
        S: Scales for analysis
        step: The step for analysis

    Returns:
        Calculated Hurst exponent (h_gen)
    """
    generator = HurstGenerator(h, filter_len, base)
    trj = []
    for value in islice(generator, TARGET_LEN):
        trj.append(value)

    Z = normalize(np.array(trj[::-1]))
    _, _, f_z, s_z = dpcca(Z, 2, step, S, processes=1, n_integral=0)
    f_z = np.sqrt(f_z)
    f_z /= f_z[0]
    res = stats.linregress(np.log(s_z), np.log(f_z)).slope
    return res


@pytest.mark.parametrize("h", testdata["h_list"])
@pytest.mark.parametrize("base", testdata["base_list"])
def test_hurst_generator(h: float, base: float):
    """
    It tests the generator for compliance with the specified Hurst indicator.

    Parameters:
        h: The specified Hurst exponent
        base: The base of the number system for bins
    """
    threshold = 0.10
    times = 10
    filter_len = int(math.log(TARGET_LEN, base))
    mean_difference = 0
    for i in range(times):
        h_gen = get_test_h(base, filter_len, h, S, STEP)

        mean_difference += abs(h_gen - h) / h
    mean_difference /= times
    assert (
        mean_difference <= threshold
    ), f"Diff between h and h_gen exceeds {threshold * 100}%: h={h}, mean diff={mean_difference * 100:.2f}%"
