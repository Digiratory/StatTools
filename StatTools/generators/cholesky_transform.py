from multiprocessing import cpu_count
from typing import Union
from collections.abc import Iterable

from StatTools.analysis.dfa import DFA
from StatTools.generators.base_filter import FilteredArray
from numpy import full, matmul, mean, ndarray
from numpy.linalg import cholesky
import pandas as pd


class CorrelatedArray:
    """
    It produces mutually correlated vectors using Cholesky transform.

    Basic usage:

    1. Having your own dataset you can produces correlated data out of it:
        d = numpy.random.normal(0, 1, (100, 1024))
        corr_target = 0.8
        corr_vectors = CorrelatedArray(data=d, threads=4).create(corr_target, h_control=False)

    2. When you need to get mutually correlated vectors quickly and you don't have your own
    dataset:

        corr_target = 0.8

        corr_vectors = CorrelatedArray(h=1.5, quantity=100, length=1024, set_mean=0,
         set_std = 1, threads=4).create(corr_target, h_control=False)

    Returns pandas.Dataframe with correlated data inside.

    Notes:
        1. h_control - creates new column in pd.Dataframe inserted as first. This column is
        DFA estimation of Hurst parameter for corresponding vector.

        2. You can use multiple targets for the call:
            corr_target = [0.7, 0.9]

            This way output will consist of multiple concatenated arrays where the index
            represents input corr.
        3. This module (CorrelatedArray) doesn't guarantee that all output vectors are
        going to have given mutual corr. coefficient. You have to select correlated groups
        by your own!
    """

    def __init__(self, h=None, quantity=None, length=None, data=None, set_mean=0, set_std=1,
                 threads=cpu_count()):

        if data is not None:
            self.length = length if length is not None else len(data[0])
            self.quantity = quantity if quantity is not None else len(data)
            self.dataset = data
        else:
            if length is None or quantity is None:
                raise TypeError(f"Didn't specify {'length' if length is None else 'quantity'} . . .")

            self.dataset = FilteredArray(h, length, set_mean, set_std).generate(n_vectors=quantity, progress_bar=False,
                                                                                threads=threads)
            self.quantity, self.length = quantity, length

        self.set_mean, self.set_std, self.threads = set_mean, set_std, threads

    def create(self, corr_target: Union[float, Iterable, ndarray], h_control=False) -> pd.DataFrame:
        """
        Main method.
        1. For each input corr coefficient I create Rxx matrix which consist of
        corr_coefficient except diagonal is ones.
        2. I produce Cholesky matrix using numpy.linalg.cholesky
        3. Chol. matrix is multiplied by input array of vectors

        In case h_control is True we need to call DFA on the output.
        """

        result = pd.DataFrame()

        if not isinstance(corr_target, Iterable):
            self.corr_target = [corr_target]
        else:
            self.corr_target = corr_target

        for corr in self.corr_target:
            indices = [corr for i in range(self.quantity)]

            if isinstance(self.corr_target, ndarray):
                Rxx = self.corr_target
            else:
                Rxx = full((self.quantity, self.quantity), corr, dtype=float)
                for i, j in zip(range(self.quantity), range(self.quantity)):
                    Rxx[i][j] = 1

            chol_transform_matrix = cholesky(Rxx)
            correlated_vectors = pd.DataFrame(matmul(chol_transform_matrix, self.dataset), index=indices)

            if h_control:
                if self.threads != 1:
                    h_estimated = DFA(correlated_vectors.to_numpy()).parallel_2d(threads=self.threads)
                else:
                    h_estimated = DFA(correlated_vectors.to_numpy()).find_h()

                correlated_vectors.insert(0, 'H_est', h_estimated)

            if isinstance(self.corr_target, ndarray):
                return correlated_vectors

            result = correlated_vectors if result.size == 0 else result.append(correlated_vectors)

        return result


if __name__ == '__main__':
    "Simple test. Here I create (100, 1024) array with given H = 1.5 then transform it" \
    "using Cholesky distribution"

    d = FilteredArray(1.5, 1024, set_mean=10, set_std=3).generate(n_vectors=100)

    x = CorrelatedArray(data=d, threads=1).create(0.7, h_control=True).to_numpy()

