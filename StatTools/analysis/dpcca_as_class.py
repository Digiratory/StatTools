from collections.abc import Iterable
from ctypes import c_double
from functools import partial
from multiprocessing import Pool
from numpy import array, ndarray, array_split, arange, polyfit, polyval, zeros, mean, sqrt, \
    cumsum, concatenate
from numpy.linalg import inv
from typing import Union
from contextlib import closing
from StatTools.auxiliary import SharedBuffer
import gc


class DPCCA:

    def __init__(self, arr: Union[ndarray, SharedBuffer], pd: int, step: float, s: Union[int, Iterable]):

        if isinstance(arr, ndarray):
            self.arr = [arr] if arr.ndim == 1 else arr
        else:
            self.arr = arr
        self.pd, self.step, self.s = pd, step, s
        self.shape = arr.shape

        if not 0 < step <= 1:
            raise ValueError("0 < step <= 1 !")

    def forward(self, processes: int = 1, force_gc: Union[bool, tuple] = False):
        if force_gc:
            force_gc = (2, 2)

        if isinstance(self.s, (tuple, list, ndarray)):
            init_s_len = len(self.s)

            s = list(filter(lambda x: x <= self.shape[1] / 4, self.s))
            if len(s) < 1:
                raise ValueError("All input S values are larger than vector shape / 4 !")

            if len(s) != init_s_len:
                print(f"\tDPCAA warning: only following S values are in use: {s}")

            processes = len(s) if processes > len(s) else processes

            S = array(s, dtype=int) if not isinstance(self.s, ndarray) else s
            S_by_workers = array_split(S, processes)

            if processes == 1:
                return self._dpcca_worker(s) + s

            if isinstance(self.arr, ndarray):
                chunk = SharedBuffer(self.shape, c_double)
                chunk.write(self.arr)
                chunk.apply_in_place(cumsum, by_1st_dim=True)
                self.arr = chunk
            elif isinstance(self.arr, SharedBuffer):
                self.arr.apply_in_place(cumsum, by_1st_dim=True)

            with (closing(Pool(processes=processes, initializer=self.arr.buffer_init,
                               initargs=({'ARR': self.arr},)))) as pool:
                result = pool.map(partial(self._dpcca_worker, force_gc=force_gc), S_by_workers)

        elif isinstance(self.s, int):
            if self.s > self.shape[1] / 4:
                raise ValueError("Cannot use S > L / 4")
        else:
            raise TypeError("Input S values could be : int, tuple, list or numpy.ndarray!")

    def _dpcca_worker(self, s: Union[int, Iterable], force_gc: Union[bool, tuple]):

        s_current = [s] if not isinstance(s, Iterable) else s

        cumsum_arr = SharedBuffer.get("ARR") if isinstance(self.arr, SharedBuffer) else cumsum(self.arr, axis=1)

        shape = self.arr.shape

        F = zeros((len(s_current), shape[0], shape[0]), dtype=float)
        R = zeros((len(s_current), shape[0], shape[0]), dtype=float)
        P = zeros((len(s_current), shape[0], shape[0]), dtype=float)

        for s_i, s_val in enumerate(s_current):

            V = arange(0, shape[1] - s_val, int(self.step * s_val))
            Xw = arange(s_val, dtype=int)
            Y = zeros((shape[0], len(V)), dtype=object)

            for n in range(cumsum_arr.shape[0]):
                for v_i, v in enumerate(V):
                    W = cumsum_arr[n][v:v + s_val]
                    if len(W) == 0:
                        print(f"\tFor s = {s_val} W is an empty slice!")
                        return P, R, F

                    p = polyfit(Xw, W, deg=self.pd)
                    Z = polyval(p, Xw)
                    Y[n][v_i] = Z - W

                    if isinstance(force_gc, tuple):
                        if n % force_gc[0] == 0:
                            gc.collect(force_gc[1])

            Y = array([concatenate(Y[i]) for i in range(Y.shape[0])])

            for n in range(shape[0]):
                for m in range(n + 1):
                    F[s_i][n][m] = mean(Y[n] * Y[m]) / (s_val - 1)
                    F[s_i][m][n] = F[s_i][n][m]

            for n in range(shape[0]):
                for m in range(n + 1):
                    R[s_i][n][m] = F[s_i][n][m] / sqrt(F[s_i][n][n] * F[s_i][m][m])
                    R[s_i][m][n] = R[s_i][n][m]

            Cinv = inv(R[s_i])

            for n in range(shape[0]):
                for m in range(n + 1):
                    if Cinv[n][n] * Cinv[m][m] < 0:
                        print(f"S = {s_val} | Error: Sqrt(-1)! No P array values for this S!")
                        break

                    P[s_i][n][m] = -Cinv[n][m] / sqrt(Cinv[n][n] * Cinv[m][m])
                    P[s_i][m][n] = P[s_i][n][m]
                else:
                    continue
                break

        return P, R, F

