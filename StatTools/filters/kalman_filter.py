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

    def auto_configure(self, signal: np.array, dt: float):
        """
        Automatically adjusts Q and R based on the input signal.

        Parameters:
            signal (np.array): Input signal (observations)
            dt (float): Time interval between measurements
        """
        self.Q = self.get_Q(signal, dt)
        self.R = self.get_R(signal)
