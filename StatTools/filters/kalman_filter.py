import numpy as np
from filterpy.kalman import KalmanFilter
from numpy.typing import NDArray

from StatTools.analysis.dfa import DFA
from StatTools.generators.kasdin_generator import KasdinGenerator


class EnhancedKalmanFilter(KalmanFilter):
    """
    Advanced Kalman filter based on filterpy.kalman.KalmanFilter
    with methods for automatic calculation of transition matrix (F)
    and measurement covariance matrix (R).
    """

    def get_R(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculates the measurement covariance matrix (R) for the Kalman filter.

        Parameters:
            signal (NDArray[np.float64]): Input signal (noise)

        Returns:
            NDArray[np.float64]: A 1x1 dimension covariance matrix R
        """
        return np.std(signal) ** 2

    def _get_filter_coefficients(
        self, signal: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Helper method to get filter coefficients."""
        dfa = DFA(signal)
        h = dfa.find_h()
        generator = KasdinGenerator(h, length=signal.shape[0])
        return generator.get_filter_coefficients()

    def get_F(
        self, signal: NDArray[np.float64], dt: float, order: int = 2
    ) -> NDArray[np.float64]:
        """
        Calculates the transition matrix F for the Kalman filter.

        Parameters:
            signal (NDArray[np.float64]): Input signal
            dt (float): Time step
            order (int): Order of the filter

        Returns:
            NDArray[np.float64]: transition matrix F

        Raises:
            ValueError: If the filter order is not supported
                        (only orders 1, 2, 3 are currently implemented)
        """
        dfa = DFA(signal)
        h = dfa.find_h()
        generator = KasdinGenerator(h, length=signal.shape[0])
        A = generator.get_filter_coefficients()
        if order == 1:
            return np.array([[1, dt], [0, 1]])
        if order == 2:
            return np.array(
                [[-A[0] - A[1], A[1] * dt], [(-1 - A[0] - A[1]) / dt, A[1]]]
            )
        # TODO: add dt for order 3
        if order == 3:
            return np.array(
                [
                    [-A[0] - A[1] - A[2], A[1] + 2 * A[2], -A[2]],
                    [-1 - A[0] - A[1] - A[2], A[1] + 2 * A[2], -A[2]],
                    [-1 - A[0] - A[1] - A[2], -1 + A[1] + 2 * A[2], -A[2]],
                ]
            )
        raise ValueError(f"Order {order} is not supported")

    def auto_configure(
        self,
        signal: NDArray[np.float64],
        noise: NDArray[np.float64],
        dt: float,
        order: int = 2,
    ):
        """
        Automatically adjusts R, F based on the input data.

        Parameters:
            signal (NDArray[np.float64]): Original signal
            noise (NDArray[np.float64]): Noise signal
            dt (float): Time interval between measurements
            ar_vector(NDArray[np.float64]): Autoregressive filter coefficients
        """
        # TODO: add Q matrix auto configuration
        self.R = self.get_R(noise)
        self.F = self.get_F(signal, dt, order)
