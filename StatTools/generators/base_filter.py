from contextlib import closing
from functools import partial
from multiprocessing import Pool, cpu_count
from random import gauss
import numpy as np
from numpy.fft import rfft, irfft
from numpy.random import normal
from StatTools.analysis.dfa import DFA
from tqdm import tqdm
import platform


class Filter:
    """
    A class for generating long-term correlated data by passing a dataset through a filter.
    This implementation processes data sequentially. For better performance, use FilteredArray
    which supports parallel processing.

    Args:
        h (float): Hurst parameter
        length (int): Length of input vectors
        set_mean (float, optional): Target mean for output vectors. Defaults to 0.
        set_std (float, optional): Target standard deviation for output vectors. Defaults to 1.

    Note:
        All input vectors must have the same length along the first dimension.

    Example:
        >>> filter = Filter(0.8, 1440)
        >>> # Get all vectors at once
        >>> correlated_vectors = filter.generate(n_vectors=10)

        # Or iterate through values one by one
        >>> for value in filter:
        ...     print(value)

        To filter your own data:
        >>> filter = Filter(0.8, 1440, set_mean=10, set_std=3)
        >>> filter.data = np.random.normal(0, 1, (100, 1440 * 3)) # use your data instead of numpy.normal()
        >>> correlated_vectors = filter.generate(n_vectors=100)
    """

    def __init__(self, h: float, length: int, set_mean: float = 0, set_std: float = 1):
        if h < 0:
            raise ValueError("Hurst parameter cannot be negative!")
        if length < 0:
            raise ValueError("Length cannot be negative!")

        self.h = h
        self.length = length
        self.set_mean = set_mean
        self.set_std = set_std
        self._data = None
        self._current_index = 0
        self._trajectory = None

        beta = 1 - 2 + 2 * self.h
        self.total_length = int(self.length * 3 / 2 + 1)
        self.beta_coefficients = np.fromiter(
            (pow(n, (-beta / 2)) for n in range(1, self.total_length)), dtype=np.float64
        )

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if value is not None:
            if not isinstance(value, np.ndarray):
                try:
                    value = np.array(value)
                except Exception:
                    raise ValueError("Cannot cast input to numpy array!")
        self._data = value

    def generate(self, n_vectors: int = 1, progress_bar: bool = False) -> np.ndarray:
        """
        Generate correlated vectors by passing data through the filter.

        Args:
            n_vectors (int): Number of vectors to generate
            progress_bar (bool): Whether to show progress bar

        Returns:
            np.ndarray: Array of correlated vectors

        Raises:
            ValueError: If n_vectors is less than 1
        """
        if n_vectors < 1:
            raise ValueError("Number of vectors must be at least 1!")

        basis = self.data if self.data is not None else self._get_basis(n_vectors)

        if basis.ndim == 1:
            return self._pass_through_filter(basis)

        result_array = np.array([], dtype=np.float64)
        for vector in tqdm(
            basis, desc=f"Filter[{self.h}|{self.length}]", disable=not progress_bar
        ):
            passed_through = self._pass_through_filter(vector)
            result_array = (
                passed_through
                if result_array.size < 1
                else np.vstack((result_array, passed_through))
            )
        return result_array

    def __iter__(self):
        """Initialize the iterator."""
        if self._trajectory is None:
            basis = self._get_basis(1)
            self._trajectory = self._pass_through_filter(basis)
        self._current_index = 0
        return self

    def __next__(self):
        """Get the next value from the trajectory."""
        if self._trajectory is None:
            raise StopIteration("Trajectory is not initialized!")
        if self._current_index >= len(self._trajectory):
            raise StopIteration("Length of trajectory is exceeded!")
        value = self._trajectory[self._current_index]
        self._current_index += 1
        return value

    def __len__(self):
        """Return the length of the trajectory."""
        if self._trajectory is None:
            return 0
        return len(self._trajectory)

    def _pass_through_filter(self, vector: np.ndarray) -> np.ndarray:
        """
        Pass input data through the filter using FFT.

        Args:
            vector (np.ndarray): Input vector to filter

        Returns:
            np.ndarray: Filtered vector with specified mean and standard deviation
        """
        full_spec = np.abs(rfft(vector))[:-1]
        phase_spec = np.angle(full_spec)

        self.spec = full_spec
        self.angle = phase_spec

        output = full_spec * self.beta_coefficients * np.exp(phase_spec * 1j)
        self.output = output

        reversed_fft = irfft(output, n=self.length * 3)
        result = reversed_fft[
            int(len(reversed_fft) / 3) : int(len(reversed_fft) * 2 / 3)
        ]

        # Normalize to desired mean and standard deviation
        result = result * (self.set_std / np.std(result, ddof=1))
        result = result + (self.set_mean - np.mean(result))

        return result

    def _get_basis(self, n_vectors: int) -> np.ndarray:
        """
        Generate basis vectors with normally distributed values.

        Args:
            n_vectors (int): Number of vectors to generate

        Returns:
            np.ndarray: Array of normally distributed vectors
        """
        if platform.system() == "Linux":
            if n_vectors == 1:
                return np.array([gauss(0, 1) for _ in range(self.length * 3)])
            return np.array(
                [gauss(0, 1) for _ in range(n_vectors * self.length * 3)]
            ).reshape(n_vectors, -1)
        else:
            if n_vectors == 1:
                return normal(0, 1, self.length * 3)
            return normal(0, 1, (n_vectors, self.length * 3))


class FilteredArray(Filter):
    """
    An optimized version of Filter that supports parallel processing and ensures
    generated vectors have the specified Hurst parameter within a predefined range.

    Args:
        h (float): Hurst parameter
        length (int): Length of input vectors
        set_mean (float, optional): Target mean for output vectors. Defaults to 0.
        set_std (float, optional): Target standard deviation for output vectors. Defaults to 1.

    Note:
        1. All input vectors must have the same length along the first dimension
        2. Uses all available CPU threads by default
        3. When filtering custom data, use length * 3 to account for FFT edge effects
        4. Uses DFA method to ensure generated vectors have the specified Hurst parameter

    Example:
        >>> filter = FilteredArray(0.8, 1440, set_mean=10, set_std=3)
        >>> vectors = filter.generate(n_vectors=len(x), threads=12, progress_bar=True)
    """

    def generate(
        self,
        n_vectors: int = 1,
        progress_bar: bool = False,
        threads: int = cpu_count(),
        h_limit: float = 0.05,
    ) -> np.ndarray:
        """
        Generate correlated vectors with parallel processing.

        Args:
            n_vectors (int): Number of vectors to generate
            progress_bar (bool): Whether to show progress bar
            threads (int): Number of parallel threads to use
            h_limit (float): Maximum allowed deviation from target Hurst parameter

        Returns:
            np.ndarray: Array of correlated vectors
        """
        self.h_limit = h_limit
        self.progress_bar = progress_bar
        self.n_vectors = n_vectors

        threads = min(threads, n_vectors)
        indices = np.array_split(
            np.linspace(0, n_vectors - 1, n_vectors, dtype=int), threads
        )

        if self.data is not None:
            basis = self.data
            return self.__create_pool(
                partial(self._iterate_through_chunk, basis=basis), threads, indices
            )

        if threads <= 1 or n_vectors == 1:
            return self._get_valid_vector(indices)

        return self.__create_pool(self._get_valid_vector, threads, indices)

    def _get_valid_vector(self, indices: np.ndarray) -> np.ndarray:
        """Generate vectors that meet the Hurst parameter criteria."""
        result_array = np.array([], dtype=np.float64)

        for _ in tqdm(
            range(len(indices)), desc="Filtering", disable=not self.progress_bar
        ):
            while True:
                base_vector = super()._get_basis(1)
                filtered_base = super()._pass_through_filter(base_vector)

                if abs(DFA(filtered_base).find_h() - self.h) <= self.h_limit:
                    result_array = (
                        filtered_base
                        if result_array.size < 1
                        else np.vstack((result_array, filtered_base))
                    )
                    break

        return result_array

    def __create_pool(
        self, func: callable, threads: int, indices: np.ndarray
    ) -> np.ndarray:
        """Create and manage a process pool for parallel processing."""
        with closing(Pool(processes=threads)) as pool:
            results = pool.map(func, indices)

        return np.vstack(results)

    def _iterate_through_chunk(
        self, v_range: np.ndarray, basis: np.ndarray
    ) -> np.ndarray:
        """Process a chunk of vectors in parallel."""
        result_array = np.array([], dtype=np.float64)
        for vector in tqdm(
            np.take(basis, v_range, axis=0),
            desc="Filtering",
            disable=not self.progress_bar,
        ):
            corr_vector = super()._pass_through_filter(vector)
            result_array = (
                corr_vector
                if result_array.size < 1
                else np.vstack((result_array, corr_vector))
            )
        return result_array
