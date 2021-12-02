import time
from contextlib import closing
from functools import partial
from multiprocessing import Pool, cpu_count
from random import gauss
from numpy import array, array_split, ndarray, linspace, zeros, fromiter, vstack, angle, exp, std, mean, \
    abs, take
from numpy.fft import rfft, irfft
from numpy.random.mtrand import normal
from StatTools.analysis.dfa import DFA
from tqdm import tqdm
from StatTools.auxiliary import CheckNumpy
import platform


class Filter:
    """
    Passes a given dataset through filter thus producing a long-term correlated data.
    'Filter' does so sequentially. For better performance use 'FilteredArray' which
    allows you to pass a data through filter in parallel mode.

    Inputs: h - given Hurst parameter
            length - length of input vectors
            set_mean, set_std - output mean and st.dev. of vectors

    Note that all input vectors are supposed to have same length along first dimension!

    Usage:

        correlated_vectors = Filter(0.8, 1440).generate(n_vectors=10)   # creating filter itself

        In order to filter you own data first you need to assign your array to Filter.data:

            filter = Filter(0.8, 1440, set_mean = 10, set_std = 3)      # creating the filter
            filter.data = numpy.random.normal(0, 1, (100, 1440 * 3))    # use your data instead of
                                                                        numpy.normal()
            correlated_vectors = filter.generate(n_vectors = 100)       # now you can pass through

    """

    data = CheckNumpy()

    def __init__(self, h: float, length: int, set_mean=0, set_std=1):

        if h < 0:
            raise ValueError("H < 0!")
        if length < 0:
            raise ValueError("Length cannot be < 0!")

        self.h = h
        self.length = length
        self.set_mean = set_mean
        self.set_std = set_std

        beta = 1 - 2 + 2 * self.h
        self.total_length = int(self.length * 3 / 2 + 1)

        self.beta_coefficients = fromiter((pow(n, (-beta / 2)) for n in range(1, self.total_length)), dtype=float)

    def generate(self, n_vectors=1, progress_bar=False) -> ndarray:
        """
        This is primary method to use.
        Here I create vectors with normally distributed values. Then I
        pass them through filter_core method which is defined below.

        :param progress_bar:
        :param n_vectors: number of total vectors to be generated
        :return: long-term correlated vectors passed through filter_core
        """
        if n_vectors < 1:
            raise ValueError("No vectors to generate!")

        if 'data' in self.__dict__:
            basis = self.__dict__['data']
        else:
            basis = self._get_basis(n_vectors)

        if basis.ndim == 1:
            return self._pass_through_filter(basis)
        else:
            result_array = array([], dtype=float)
            for vector in tqdm(basis, desc=f"Filter[{self.h}|{self.length}]", disable=not progress_bar):
                passed_through = self._pass_through_filter(vector)
                result_array = passed_through if result_array.size < 1 else vstack((result_array, passed_through))
            return result_array

    def _pass_through_filter(self, vector: ndarray) -> ndarray:
        """
        This is where I pass input data through filter.
        First I get magnitude and phase spectrum.
        Then you have to multiply each magnitude component by betta-values
        leaving the same phase components.
        Finally, revers FFT is taken and the middle part of the vector is cut out
        in order to avoid FFT edge effects.

        :param vector: input basis vector defined in generate() method.
        :return: vector that has
        """

        full_spec = abs(rfft(vector))[:-1]
        phase_spec = angle(full_spec)

        self.spec = full_spec
        self.angle = phase_spec

        output = zeros(len(full_spec), dtype=complex)

        # for spec_val, phase_val, beta, n in zip(full_spec, phase_spec, self.beta_coefficients,
        #                                         range(self.total_length)):
        #     output[n] = spec_val * beta * exp(phase_val * 1j)

        output = full_spec * self.beta_coefficients * exp(phase_spec * 1j)

        self.output = output
        reversed_fft = irfft(output, n=self.length * 3)
        result = reversed_fft[int(len(reversed_fft) / 3):int(len(reversed_fft) * 2 / 3)]

        result = result * (self.set_std / std(result, ddof=1))
        result = result + (self.set_mean - mean(result))

        return result

    def _get_basis(self, n_vectors: int) -> ndarray:
        """
        Generates basis vectors for the filter

        :param n_vectors: total num of vectors to be generated
        :return: given number of vectors filled by normally distributed values
        """

        if platform.system() == "Linux":
            basis = array([], dtype=float)
            for i in range(n_vectors):
                random_base = array([gauss(0, 1) for k in range(self.length * 3)])
                if n_vectors == 1:
                    return random_base
                basis = random_base if basis.size < 1 else vstack((basis, random_base))
        else:
            if n_vectors == 1:
                return normal(0, 1, self.length * 3)
            basis = normal(0, 1, (n_vectors, self.length * 3))
        return basis


class FilteredArray(Filter):
    """
    FilteredArray is an extension for main class Filter. It provides faster
    computation and Hurst parameter control.

    Inputs: h - given Hurst parameter
            length - length of input vectors
            set_mean, set_std - output mean and st.dev. of vectors

    Note :
        1. All input vectors are supposed to have same length along first dimension!
        2. By default all available threads are going to be in use while generating!
        3. WHEN FILTERING YOUR OWN DATA USE LENGTH OF VECTORS * 3 (FFT edge effects)
        4. This class is different from just a 'Filter'. It uses DFA method to ensure
        that all realizations generated have given Hurst parameter with a predefined
        range. Except the case when you use your own input data.

    Usage:

        correlated_vectors = Filter(0.8, 1440).generate(n_vectors=1000, threads=4, progress_bar=True)

        In order to filter you own data first you need to assign your array to Filter.data:

            filter = Filter(0.8, 1440, set_mean = 10, set_std = 3)      # creating the filter
            filter.data = numpy.random.normal(0, 1, (100, 1440 * 3))       # use your data instead of
                                                                        numpy.normal()
            correlated_vectors = filter.generate(n_vectors = 100, threads=4, progress_bar=True)

    """

    def __init__(self, h, length, set_mean=0, set_std=1):

        super().__init__(h, length, set_mean, set_std)

    def generate(self, n_vectors=1, progress_bar=False, threads=cpu_count(), h_limit=0.05) -> ndarray:
        """
        :param n_vectors: Total number of vectors. Even if you want to process your own data
                          you should define total num.
        :param progress_bar: Just a flag that let you check a progress.
        :param threads: Total threads (interpreter's processes) for computation. Default is
                        total num that is available in a system.
        :param h_limit: Hurst parameter max deviation from target.
        :return:
        """

        self.__dict__['h_limit'] = h_limit
        self.__dict__['progress_bar'] = progress_bar
        self.__dict__['n_vectors'] = n_vectors

        threads = n_vectors if threads > n_vectors else threads

        if threads > 1 and n_vectors > 1:
            indices = array_split(linspace(0, n_vectors - 1, n_vectors, dtype=int), threads)
        else:
            indices = linspace(0, n_vectors - 1, n_vectors, dtype=int)

        if 'data' in self.__dict__:
            basis = self.__dict__['data']
            return self.__create_pool(partial(self._iterate_through_chunk, basis=basis), threads, indices)
        else:
            if threads <= 1 or n_vectors == 1:
                return self._get_valid_vector(indices)
            else:
                return self.__create_pool(self._get_valid_vector, threads, indices)

    def _get_valid_vector(self, indices: ndarray) -> ndarray:
        result_array = array([], dtype=float)

        for n in tqdm(range(len(indices)), desc=f"Filtering", disable=not self.__dict__['progress_bar']):
            while True:
                base_vector = super(FilteredArray, self)._get_basis(1)
                filtered_base = super(FilteredArray, self)._pass_through_filter(base_vector)

                if abs(DFA(filtered_base).find_h() - self.h) <= self.__dict__['h_limit']:
                    result_array = filtered_base if result_array.size < 1 else vstack((result_array, filtered_base))
                    break

        return result_array

    def __create_pool(self, partial_func_ref, threads: int, indices: ndarray) -> ndarray:
        with closing(Pool(processes=threads)) as pool:
            result = pool.map(partial_func_ref, indices)

        result_array = array([], dtype=float)
        for res in result:
            result_array = res if result_array.size < 1 else vstack((result_array, res))
        return result_array

    def _iterate_through_chunk(self, v_range: ndarray, basis: ndarray) -> ndarray:
        result_array = array([], dtype=float)
        for vector in tqdm(take(basis, v_range, axis=0), desc=f"Filtering", disable=not self.__dict__['progress_bar']):
            corr_vector = super(FilteredArray, self)._pass_through_filter(vector)
            result_array = corr_vector if result_array.size < 1 else vstack((result_array, corr_vector))

        return result_array


if __name__ == '__main__':
    x = normal(0, 1, (1000, 1440 * 3))

    t1 = time.perf_counter()
    filter_ = FilteredArray(0.8, 1440, set_mean=10, set_std=3)
    vectors = filter_.generate(n_vectors=len(x), threads=12, progress_bar=True)
    print("ALL DONE!")