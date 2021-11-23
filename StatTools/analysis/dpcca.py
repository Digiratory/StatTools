from collections.abc import Iterable
from ctypes import c_double
from functools import partial
from multiprocessing import Pool
from matplotlib.pyplot import plot, semilogy, show, ylabel, xlabel, legend, title
from numpy import array, ndarray, log, array_split, arange, loadtxt, polyfit, polyval, zeros, mean, sqrt, dstack, stack, \
    vstack, cumsum, concatenate, any, log10, round, savetxt, int64
from numpy.linalg import inv
from typing import Union
from contextlib import closing
from StatTools.generators.base_filter import FilteredArray
from StatTools.auxiliary import SharedBuffer


# @profile()
def dpcca_worker(s: Union[int, Iterable], arr: Union[ndarray, None], step: float, pd: int, buffer_in_use: bool) -> \
        Union[tuple, None]:
    """
    Core of DPCAA algorithm. Takes bunch of S-values and returns 3 3d-matrices: first index
    represents S value.
    """

    s_current = [s] if not isinstance(s, Iterable) else s

    cumsum_arr = SharedBuffer.get("ARR") if buffer_in_use else cumsum(arr, axis=1)

    shape = cumsum_arr.shape if buffer_in_use else arr.shape

    F = zeros((len(s_current), shape[0], shape[0]), dtype=float)
    R = zeros((len(s_current), shape[0], shape[0]), dtype=float)
    P = zeros((len(s_current), shape[0], shape[0]), dtype=float)

    for s_i, s_val in enumerate(s_current):

        V = arange(0, shape[1] - s_val, int(step * s_val))
        Xw = arange(s_val, dtype=int)
        Y = zeros((shape[0], len(V)), dtype=object)

        for n, vector in enumerate(cumsum_arr):
            for v_i, v in enumerate(V):
                W = vector[v:v + s_val]
                if len(W) == 0:
                    print(f"\tFor s = {s_val} W is an empty slice!")
                    return P, R, F

                p = polyfit(Xw, W, deg=pd)
                Z = polyval(p, Xw)
                Y[n][v_i] = Z - W

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
                    raise ValueError("Inverted matrix has negative values!")

                P[s_i][n][m] = -Cinv[n][m] / sqrt(Cinv[n][n] * Cinv[m][m])
                P[s_i][m][n] = P[s_i][n][m]

    return P, R, F


def start_pool_with_buffer(buffer: SharedBuffer, processes: int, s_by_workers: ndarray, pd: int, step: float):
    buffer.apply_in_place(cumsum, by_1st_dim=True)

    with closing(Pool(processes=processes, initializer=buffer.buffer_init, initargs=({"ARR": buffer},))) as pool:
        pool_result = pool.map(partial(dpcca_worker, arr=None, pd=pd, step=step, buffer_in_use=True), s_by_workers)

    return pool_result


def dpcca(arr: ndarray, pd: int, step: float, s: Union[int, Iterable], processes: int,
          buffer: Union[bool, SharedBuffer] = False) -> tuple:
    """
    Detrended Partial-Cross-Correlation Analysis : https://www.nature.com/articles/srep08143

    arr: dataset array
    pd: polynomial degree
    step: share of S - value. It's set usually as 0.5. The integer part of the number will be taken
    s : points where  fluctuation function F(s) is calculated. More on that in the article.
    process: num of workers to spawn
    buffer: allows to share input array between processes. NOTE: if you

    Returns 3 3-d matrices where first dimension represents given S-value.

    Basic usage:
        You can get whole F(s) function for first vector as:

            s_vals = [i**2 for i in range(1, 5)]
            P, R, F = dpcaa(input_array, 2, 0.5, s_vals, len(s_vals))
            fluct_func = [F[s][0][0] for s in s_vals]

    """

    if isinstance(s, Iterable):
        init_s_len = len(s)
        s = list(filter(lambda x: x <= arr.shape[1] / 4, s))
        if len(s) < 1:
            raise ValueError("All input S values are larger than vector shape / 4 !")

        if len(s) != init_s_len:
            print(f"\tDPCAA warning: only following S values are in use: {s}")

    elif s > arr.shape[1] / 4:
        raise ValueError("Cannot use S > L / 4")

    if processes == 1 or len(s) == 1:
        return dpcca_worker(s, arr, step, pd, buffer_in_use=False)
    else:
        if processes > len(s):
            processes = len(s)

    S = array(s, dtype=int) if not isinstance(s, ndarray) else s
    S_by_workers = array_split(S, processes)

    P, R, F = array([]), array([]), array([])

    if isinstance(buffer, bool):
        if buffer:
            shared_input = SharedBuffer(arr.shape, c_double)
            shared_input.write(arr)

            pool_result = start_pool_with_buffer(shared_input, processes, S_by_workers, pd, step)

        else:
            with closing(Pool(processes=processes)) as pool:
                pool_result = pool.map(partial(dpcca_worker, arr=arr, pd=pd, step=step, buffer_in_use=False),
                                       S_by_workers)

    elif isinstance(buffer, SharedBuffer):
        pool_result = start_pool_with_buffer(buffer, processes, S_by_workers, pd, step)
    else:
        raise ValueError("Wrong type of input buffer!")

    for res in pool_result:
        P = res[0] if P.size < 1 else vstack((P, res[0]))
        R = res[1] if R.size < 1 else vstack((R, res[1]))
        F = res[2] if F.size < 1 else vstack((F, res[2]))

    return P, R, F


if __name__ == '__main__':
    """
    Simple test. Having some S values , for 3 different H get fluctuation 
    function for second vector, calculate the slope and create a chart.
    """
    vectors_length = 10000
    n_vectors = 100  # (100, 10_000) dataset
    s = [pow(2, i) for i in range(3, 14)]
    step = 0.5
    poly_deg = 2

    vector_index = 1
    threads = 4

    for h in (0.5, 0.9, 1.5):
        # We can generate new dataset using statement below
        # x = FilteredArray(h, vectors_length).generate(n_vectors=n_vectors, progress_bar=False, threads=threads)
        # savetxt("C:\\Users\\ak698\\Desktop\\work\\vectors.txt", x)

        x = loadtxt("C:\\Users\\ak698\\Desktop\\work\\vectors.txt")

        # x = normal(0, 1, (10 ** 3, 10 ** 3))

        P, R, F = dpcca(x, poly_deg, 0.5, s, 4, buffer=True)

        s_vals = [s_ for s_ in range(F.shape[0])]

        fluct_func = [F[s_][vector_index][vector_index] for s_ in s_vals]
        f = log10(fluct_func)

        s_vals = log10([s[s_i] for s_i in s_vals])

        coefs = polyfit(s_vals, f, deg=1)
        regres = polyval(coefs, s_vals)
        plot(s_vals, f, label="Fluct")
        plot(s_vals, regres, label=f"Approx. slope={round(coefs[0], 2)}")
        legend()
        xlabel("Log(S)")
        ylabel("Log( F(s) )")
        title(f"H = {h}")
        show()

        print(h, coefs)

