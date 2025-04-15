import numpy as np


class AutoregressiveFilter:
    """
    Filter for autoregressive (AR) model
    """

    def __init__(self, h: float, length: int) -> None:
        self.h = h
        self.length = length
        self.coeff_vector = np.zeros(self.length, dtype=np.float64)
        self.coeff_vector[0] = 1.0
        self._init_filter()

    def _init_filter(self) -> None:
        beta = 2 * self.h - 1
        k = np.arange(1, self.length)
        self.coeff_vector[1:] = np.cumprod((k - 1 - beta / 2) / k)

    def set_h(self, h: float) -> None:
        """
        Update the Hurst exponent
        """
        self.h = h
        self._init_filter()

    def get_vector(self) -> np.array:
        """
        Get vector of filter coefficients
        """
        return self.coeff_vector
