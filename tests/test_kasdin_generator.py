import numpy as np
import pytest

from StatTools.analysis.dfa import DFA
from StatTools.generators.kasdin_generator import KasdinGenerator

testdata = {
    "h_list": [i * 0.01 for i in range(50, 200, 20)],
    "rate_list": [14],
}


def get_test_h(
    h: float,
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
    generator = KasdinGenerator(h, length=target_len)
    signal = generator.get_full_sequence()
    dfa = DFA(signal)
    return dfa.find_h()


@pytest.mark.parametrize("h", testdata["h_list"])
@pytest.mark.parametrize("rate", testdata["rate_list"])
def test_kasdin_generator(h: float, rate: int):
    """
    It tests the generator for compliance with the specified Hurst exponent.

    Parameters:
        h: The specified Hurst exponent
        base: The base of the number system for bins
    """
    threshold = 0.10
    times = 3
    mean_difference = 0
    length = 2**rate
    for _ in range(times):
        h_gen = get_test_h(h, length)
        mean_difference += abs(h_gen - h) / h
    mean_difference /= times
    assert (
        mean_difference <= threshold
    ), f"Diff between h and h_gen exceeds {threshold * 100}%: h={h}, h_gen={h_gen}, mean diff={mean_difference * 100:.2f}%"
