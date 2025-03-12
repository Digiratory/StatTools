import warnings
from typing import List, Iterator, Optional
import itertools
import numpy as np
from scipy.signal import lfilter

def signed_power(base: float, degree: float) -> float:
    """
    Calculates the degree of a number while preserving the sign.

    Parameters:
        base: The base of the degree
        degree: An indicator of the degree

    Returns:
        sign(base) * |base|^degree for base ≠ 0
        0 for base = 0
    """
    if base == 0:
        return 0.0
    sign = np.sign(base)
    return sign * np.abs(base) ** degree


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalizes the data to a zero mean and a single standard deviation.

    Parameters:
        data: Input data for normalization

    Returns:
        Normalized data array
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std if std != 0 else (data - mean)


def get_adaptive_filter_coefficients(
    slices_lengths: List[int], data: np.ndarray, threshold: int = 1024
) -> List[float]:
    """
    Calculates adaptive filter coefficients by weighted segment averaging.

    Parameters:
        slices_lengths: The length of the segments to process
        data: The source data of the filter
        threshold: Length threshold for switching between averaging methods

    Returns:
        List of adaptive filter coefficients
    """
    coefficients = []
    index = 0
    for length in slices_lengths:
        if index + length > len(data):
            break

        segment = data[index : index + length]
        if length > threshold:
            weights = np.linspace(1, 0.1, length)
            coeff = np.dot(segment, weights) / weights.sum()
        else:
            coeff = segment[len(segment) // 2]

        coefficients.append(coeff)
        index += length

    return coefficients


class LBFBmGenerator:
    """
    Generates a sequence of numbers based on the Hurst exponent.
    The Hurst exponent is a measure of long-term memory of time series.

    Args:
        h (float): Hurst exponent (0 < H < 2)
        filter_len (int): Filter length
        base (int, optional): Base of the number system for bins. Defaults to 2.
        random_generator (Iterator[float], optional): Iterator providing random values.
            Defaults to None, in which case np.random.randn() is used.
        length (Optional[int], optional): Maximum length of the sequence.
            Defaults to None for unlimited sequence.

    Raises:
        ValueError: If Hurst exponent is not in a range (0, 2)
        ValueError: If filter length is not positive.
        StopIteration('Sequence exhausted') : If maximum sequence length has been reached.

    Example usage:
    >>> generator = LBFBmGenerator(h, filter_len, base)
    >>> trj = list(generator)  # Get sequence of specified length
    """

    def __init__(
        self,
        h: float,
        filter_len: int,
        base: int = 2,
        random_generator: Optional[Iterator[float]] = None,
        length: Optional[int] = None
    ) -> None:
        if not 0 < h <= 2:
            raise ValueError("Hurst exponent must be in (0, 2)")
        if filter_len < 1:
            raise ValueError("Filter length must be positive")
        self.h = h
        self.filter_len = filter_len
        self.base = base
        self.current_time = 0
        self.bins = []
        self.max_steps = None
        self.length = length
        self.random_generator = random_generator or (n for n in iter(np.random.randn, None))

        self._init_bins()
        self._init_filter()

    def __iter__(self):
        return self

    def _init_bins(self):
        """Initializes the structure of bins and their boundaries."""
        self.bin_sizes: List[int] = [1] + [int(self.base**n) for n in range(self.filter_len - 1)]
        self.bins = [0.0] * self.filter_len
        self.bin_limits = list(itertools.accumulate(self.bin_sizes))
        self.max_steps = sum(self.bin_sizes)

    def _init_filter(self):
        """Initializes the filter coefficients based on the Hurst exponent."""
        beta = 2 * self.h - 1

        # Calculating the length of the initial filter
        orig_len = 1
        for i in range(self.filter_len - 1):
            orig_len += int(self.base**i)

        # Generating the initial coefficients
        A = np.zeros(orig_len)
        A[0] = 1.0
        for k in range(1, orig_len):
            A[k] = (k - 1 - beta / 2) * A[k - 1] / k

        # Optimize filter
        self.A = get_adaptive_filter_coefficients(self.bin_sizes, A)

    def _update_bins(self, new_value: float) -> None:
        """Updates the beans with a new value."""
        updated = []
        for i, curr_bin in enumerate(self.bins):
            if i == 0:
                updated.append(new_value)
                continue
            if self.current_time <= self.bin_limits[i]:
                # incomplete bin
                # We do not subtract from the curr if there is no transition through bin.
                prev = signed_power(self.bins[i - 1], (1 / self.bin_sizes[i - 1]))
                bin_upd = curr_bin + prev
                updated.append(bin_upd)
                # other = 0
                updated += [0.0] * (len(self.bins) - len(updated))
                break
            curr = signed_power(curr_bin, (1 / self.bin_sizes[i]))
            prev = signed_power(self.bins[i - 1], (1 / self.bin_sizes[i - 1]))
            bin_upd = curr_bin - curr + prev
            updated.append(bin_upd)
        self.bins = updated

    def _calculate_step(self) -> float:
        """Applies a filter."""
        return lfilter(np.ones(self.filter_len), self.A, self.bins[::-1])[-1]

    def __next__(self):
        """Generates the next signal value."""
        if self.length is not None and self.current_time >= self.length:
            raise StopIteration('Sequence exhausted')
            
        self.current_time += 1
        if self.current_time >= self.max_steps:
            warnings.warn(f"Sequence length {self.current_time} exceeded the maximum allowed length {self.max_steps}", RuntimeWarning)
        new_val = next(self.random_generator)
        self._update_bins(new_val)
        return self._calculate_step()

    def __len__(self) -> int:
        """Returns the length of the sequence if specified."""
        if self.length is None:
            raise TypeError("Length is not defined for unlimited generator")
        return self.length

    @property
    def current_bins(self) -> np.ndarray:
        """Returns the current bin values."""
        return self.bins

    def get_filter_coefficients(self) -> np.ndarray:
        """Returns the current filter coefficients."""
        return self.A

    def get_bin_sizes(self) -> List[int]:
        """Returns the bin sizes."""
        return self.bin_sizes

    def get_signal_from_bins(self) -> List[float]:
        """Returns the sum of the values for each bin."""
        return [np.sum(bin) for bin in self.bins]
