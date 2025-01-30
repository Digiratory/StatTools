import pytest

import numpy as np
from scipy import signal, stats

from StatTools.analysis import fa
testdata = [
    (0.25),
    (0.5),
    (0.75),
    (0.95),
]


@pytest.fixture(scope="module")
def sample_signal():
    """ Generate sample signal using Kasdin method

    Returns:
        Dict[float:np.ndarray]: Dict of sample signals with requested H
    """
    length = 2 ** 14
    signals = {}
    for h in testdata:
        z = np.random.normal(size=length*2)
        beta = 2 * h - 1
        L = length*2
        A = np.zeros(length*2)
        A[0] = 1
        for k in range(1, L):
            A[k] = (k - 1 - beta/2) * A[k - 1] / k

        if h == 0.5:
            Z = z
        else:
            Z = signal.lfilter(1, A, z)
        signals[h] = Z

    return signals


def test_fa_default_gaussian_noise():
    s = [2 ** i for i in range(3, 20)]
    step = 0.01
    length = 2 ** 16
    sig = np.random.standard_normal(length)

    f1, s1 = fa(sig, step, s, n_integral=1)
    f1 = np.sqrt(f1)

    res = stats.linregress(np.log(s1), np.log(f1)).slope
    assert res == pytest.approx(0.5, 0.15)

    f1, s1 = fa(sig, step, s, n_integral=2)
    f1 = np.sqrt(f1)

    res = stats.linregress(np.log(s1), np.log(f1)).slope
    assert res == pytest.approx(1.0, 0.15)


@pytest.mark.parametrize("h", testdata)
def test_fa_default(sample_signal, h):
    s = [2 ** i for i in range(3, 20)]
    step = 0.01
    sig = sample_signal[h]

    f1, s1 = fa(sig, step, s)
    f1 = np.sqrt(f1)

    res = stats.linregress(np.log(s1), np.log(f1)).slope
    assert res == pytest.approx(h, 0.2)


@pytest.mark.parametrize("h", testdata)
def test_fa_default_multiple_signals(sample_signal, h):
    s = [2 ** i for i in range(3, 20)]
    step = 0.01
    sig = sample_signal[h]
    sig = np.repeat(sig[np.newaxis, :], repeats=10, axis=0)

    f1, s1 = fa(sig, step, s)
    f1 = np.sqrt(f1)
    res = []
    for f in f1:
        res.append(stats.linregress(np.log(s1), np.log(f)).slope)
    res = np.array(res)
    assert res.mean() == pytest.approx(h, 0.2)
