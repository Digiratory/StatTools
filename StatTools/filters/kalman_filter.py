import numpy as np
from filterpy.kalman import KalmanFilter


class EnhancedKalmanFilter(KalmanFilter):
    """
    Advanced Kalman filter with methods for automatic calculation
    covariance matrices of the process (Q) and measurements (R).
    """

    def get_Q(self, signal: np.array, dt: float) -> np.array:
        """
        Calculates the process covariance matrix (Q) for the Kalman filter.

        Parameters:
            signal (np.array): Input signal (observations)
            dt (float): Time interval between measurements

        Returns:
            np.array: A 2x2 process covariance matrix Q
        """
        velocity = np.diff(signal)
        accelerations = np.diff(velocity)
        sigma_a_squared = np.nanvar(accelerations)
        return np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * sigma_a_squared

    def get_R(self, signal: np.array) -> np.array:
        """
        Calculates the measurement covariance matrix (R) for the Kalman filter.

        Parameters:
            signal (np.array): Input signal (observations)

        Returns:
            np.array: A 1x1 dimension covariance matrix R
        """
        signal = np.diff(signal)
        return np.array([[np.nanvar(signal)]])

    def get_F(self, ar_v: np.array) -> np.array:
        """
        Calculates the F for the Kalman filter.

        Parameters:
            ar_v (np.array): Autoregressive filter coefficients

        Returns:
            np.array: matrix F
        """
        # matrix = np.zeros((self.dim_x, self.dim_x))
        if self.dim_x != 3 or ar_v is None:
            print("Use simple matrix")
            return np.eye(self.dim_x)
        matrix = [
            [-ar_v[0] - ar_v[1] - ar_v[2], ar_v[1] + 2 * ar_v[2], -ar_v[2]],
            [-1 - ar_v[0] - ar_v[1] - ar_v[2], ar_v[1] + 2 * ar_v[2], -ar_v[2]],
            [-1 - ar_v[0] - ar_v[1] - ar_v[2], -1 + ar_v[1] + 2 * ar_v[2], -ar_v[2]],
        ]
        return np.array(matrix)

    def auto_configure(self, signal: np.array, dt: float, ar_vector: np.array = None):
        """
        Automatically adjusts Q, R, F based on the input data.

        Parameters:
            signal (np.array): Input signal (observations)
            dt (float): Time interval between measurements
            ar_vector(np.array): Autoregressive filter coefficients
        """
        self.Q = self.get_Q(signal, dt)
        self.R = self.get_R(signal)
        self.F = self.get_F(ar_vector)
