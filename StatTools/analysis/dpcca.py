from collections.abc import Iterable
from ctypes import c_double
from functools import partial
from multiprocessing import Pool
from numpy import array, ndarray, array_split, arange, polyfit, polyval, zeros, mean, sqrt, \
    vstack, cumsum, concatenate
from numpy.linalg import inv
from typing import Union
from contextlib import closing
from numpy.random import normal
from StatTools.auxiliary import SharedBuffer
import gc


# @profile()
def dpcca_worker(s: Union[int, Iterable], arr: Union[ndarray, None], step: float, pd: int, buffer_in_use: bool,
                 gc_params: tuple = None, short_vectors=False, n_integral=1) -> Union[tuple, None]:
    """
    Core of DPCAA algorithm. Takes bunch of S-values and returns 3 3d-matrices: first index
    represents S value.
    """
    gc.set_threshold(10, 2, 2)
    s_current = [s] if not isinstance(s, Iterable) else s

    
    if buffer_in_use:
        cumsum_arr = SharedBuffer.get("ARR") 
    else:
        cumsum_arr = arr
        for _ in range(n_integral):
            cumsum_arr = cumsum(cumsum_arr, axis=1)

    shape = cumsum_arr.shape if buffer_in_use else arr.shape

    F = zeros((len(s_current), shape[0], shape[0]), dtype=float)
    R = zeros((len(s_current), shape[0], shape[0]), dtype=float)
    P = zeros((len(s_current), shape[0], shape[0]), dtype=float)

    for s_i, s_val in enumerate(s_current):

        V = arange(0, shape[1] - s_val, int(step * s_val))
        Xw = arange(s_val, dtype=int)
        Y = zeros((shape[0], len(V)), dtype=object)

        for n in range(cumsum_arr.shape[0]):
            for v_i, v in enumerate(V):
                W = cumsum_arr[n][v:v + s_val]
                if len(W) == 0:
                    print(f"\tFor s = {s_val} W is an empty slice!")
                    return P, R, F

                p = polyfit(Xw, W, deg=pd)
                Z = polyval(p, Xw)
                Y[n][v_i] = Z - W

                if gc_params is not None:
                    if n % gc_params[0] == 0:
                        gc.collect(gc_params[1])

                # loop_func(cumsum_arr, n, v, v_i, s_val, Xw, pd, Y)

        Y = array([concatenate(Y[i]) for i in range(Y.shape[0])])

        for n in range(shape[0]):
            for m in range(n + 1):
                F[s_i][n][m] = mean(Y[n] * Y[m]) #/ (s_val - 1)
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


def start_pool_with_buffer(buffer: SharedBuffer, processes: int, s_by_workers: ndarray, pd: int, step: float,
                           gc_params: tuple = None, n_integral=1):

    for _ in range(n_integral):
        buffer.apply_in_place(cumsum, by_1st_dim=True)

    with closing(Pool(processes=processes, initializer=buffer.buffer_init, initargs=({"ARR": buffer},))) as pool:
        pool_result = pool.map(
            partial(dpcca_worker, arr=None, pd=pd, step=step, buffer_in_use=True, gc_params=gc_params),
            s_by_workers)

    return pool_result


def concatenate_3d_matrices(p: ndarray, r: ndarray, f: ndarray):
    P = concatenate(p, axis=1)[0]
    R = concatenate(r, axis=1)[0]
    F = concatenate(f, axis=1)[0]
    return P, R, F


def dpcca(arr: ndarray, pd: int, step: float, s: Union[int, Iterable], processes: int = 1,
          buffer: Union[bool, SharedBuffer] = False, gc_params: tuple = None, short_vectors=False, n_integral=1) -> tuple:
    """
    Detrended Partial-Cross-Correlation Analysis : https://www.nature.com/articles/srep08143

    arr: dataset array
    pd: polynomial degree
    step: share of S - value. It's set usually as 0.5. The integer part of the number will be taken
    s : points where  fluctuation function F(s) is calculated. More on that in the article.
    process: num of workers to spawn
    buffer: allows to share input array between processes. NOTE: if you

    Returns 3 3-d matrices where first dimension represents given S-value.

    P, 
    R, 
    F — F^2
    s — Used scales

    Basic usage:
        You can get whole F(s) function for first vector as:

            s_vals = [i**2 for i in range(1, 5)]
            P, R, F = dpcaa(input_array, 2, 0.5, s_vals, len(s_vals))
            fluct_func = [F[s][0][0] for s in s_vals]

    """
    if short_vectors:
        return dpcca_worker(s, arr, step, pd, buffer_in_use=False, gc_params=gc_params, short_vectors=True, n_integral=n_integral)

    concatenate_all = False  # concatenate if 1d array , no need to use 3d P, R, F
    if arr.ndim == 1:
        arr = array([arr])
        concatenate_all = True

    if isinstance(s, Iterable):
        init_s_len = len(s)

        s = list(filter(lambda x: x <= arr.shape[1] / 4, s))
        if len(s) < 1:
            raise ValueError("All input S values are larger than vector shape / 4 !")

        if len(s) != init_s_len:
            print(f"\tDPCAA warning: only following S values are in use: {s}")

    elif isinstance(s, (float, int)):
        if s > arr.shape[1] / 4:
            raise ValueError("Cannot use S > L / 4")
        s = (s,)

    if processes == 1 or len(s) == 1:
        p, r, f = dpcca_worker(s, arr, step, pd, buffer_in_use=False, gc_params=gc_params, n_integral=n_integral)
        if concatenate_all:
            return concatenate_3d_matrices(p, r, f) + (s,)

        return p, r, f, s

    processes = len(s) if processes > len(s) else processes

    S = array(s, dtype=int) if not isinstance(s, ndarray) else s
    S_by_workers = array_split(S, processes)

    if isinstance(buffer, bool):
        if buffer:
            shared_input = SharedBuffer(arr.shape, c_double)
            shared_input.write(arr)

            pool_result = start_pool_with_buffer(shared_input, processes, S_by_workers, pd, step, gc_params, n_integral=n_integral)

        else:
            with closing(Pool(processes=processes)) as pool:
                pool_result = pool.map(partial(dpcca_worker, arr=arr, pd=pd, step=step, buffer_in_use=False,
                                               gc_params=gc_params, n_integral=n_integral), S_by_workers)

    elif isinstance(buffer, SharedBuffer):
        pool_result = start_pool_with_buffer(buffer, processes, S_by_workers, pd, step, gc_params, n_integral=n_integral)
    else:
        raise ValueError("Wrong type of input buffer!")

    P, R, F = array([]), array([]), array([])

    for res in pool_result:
        P = res[0] if P.size < 1 else vstack((P, res[0]))
        R = res[1] if R.size < 1 else vstack((R, res[1]))
        F = res[2] if F.size < 1 else vstack((F, res[2]))

    if concatenate_all:
        return concatenate_3d_matrices(P, R, F) + (s,)

    return P, R, F, s


if __name__ == '__main__':
    """
    Simple test. Having some S values , for 3 different H get fluctuation 
    function for second vector, calculate the slope and create a chart.
    """
    # vectors_length = 10000
    # n_vectors = 100  # (100, 10_000) dataset
    # s = [pow(2, i) for i in range(3, 14)]
    # step = 0.5
    # poly_deg = 2
    #
    # vector_index = 1
    # threads = 4
    #
    # for h in (0.5, 0.9, 1.5):
    #     # We can generate new dataset using statement below
    #     # x = FilteredArray(h, vectors_length).generate(n_vectors=n_vectors, progress_bar=False, threads=threads)
    #     # savetxt("C:\\Users\\ak698\\Desktop\\work\\vectors.txt", x)
    #
    #     x = loadtxt("C:\\Users\\ak698\\Desktop\\work\\vectors.txt")
    #
    #     # x = normal(0, 1, (10 ** 3, 10 ** 3))
    #
    #     P, R, F = dpcca(x, poly_deg, 0.5, s, 4, buffer=True)
    #
    #     s_vals = [s_ for s_ in range(F.shape[0])]
    #
    #     fluct_func = [F[s_][vector_index][vector_index] for s_ in s_vals]
    #     f = log10(fluct_func)
    #
    #     s_vals = log10([s[s_i] for s_i in s_vals])
    #
    #     coefs = polyfit(s_vals, f, deg=1)
    #     regres = polyval(coefs, s_vals)
    #     plot(s_vals, f, label="Fluct")
    #     plot(s_vals, regres, label=f"Approx. slope={round(coefs[0], 2)}")
    #     legend()
    #     xlabel("Log(S)")
    #     ylabel("Log( F(s) )")
    #     title(f"H = {h}")
    #     show()
    #
    #     print(h, coefs)

    x = normal(0, 1, 2 ** 10)
    p, r, f, s = dpcca(x, 2, 0.5, [2 ** i for i in range(3, 10)], processes=12, buffer=True)
    print(f, s)
