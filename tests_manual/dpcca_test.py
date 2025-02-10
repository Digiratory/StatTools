import ctypes
import time

from numpy.random import normal
from StatTools.analysis.dpcca import dpcca, movmean
from StatTools.generators.base_filter import Filter
from StatTools.auxiliary import SharedBuffer



def run_to_compare_with_matlab():
    h = 1.5
    length = 2 ** 20
    s = [2 ** i for i in range(3, 20)]
    step = 0.5
    threads = 12

    x = Filter(h, length).generate()
    p1, r1, f1, s1 = dpcca(x, 2, step, s, processes=threads)

    for k in [2 ** i for i in [8, 10, 12]]:
        x2 = movmean(x, k)
        p2, r2, f2, s2 = dpcca(x2, 2, step, s, processes=threads)

# @profile
def run_to_test_buffer():

    shape = 2**12, 2**11

    buffer = SharedBuffer(shape, ctypes.c_double)

    ref = buffer.to_array()
    print(f"Initial size: {8 * shape[0] * shape[1] // 1024 // 1024}")

    for v in ref:
        # v[:] = Filter(1.5, shape[1]).generate()
        v[:] = normal(0, 1, shape[1])

    print("Assignment is done!")
    s = [2**i for i in range(5, 15)]

    t1 = time.perf_counter()
    # fastest:
    # p, r, f, s_out = dpcca(normal(0, 1, shape), 2, 0.5, s, processes=12, gc_params=None, buffer=False)
    # cheapest:
    p, r, f, s_out = dpcca(ref, 2, 0.5, s, processes=12, gc_params=(2, 2), buffer=False)

    print(f"Took : {time.perf_counter() - t1}")

    vector_to_check = shape[0] // 2

    f_s = [f[s_i][vector_to_check][vector_to_check] for s_i in range(len(s_out))]



if __name__ == '__main__':

    # run_to_compare_with_matlab()
    run_to_test_buffer()
    # print(gc.get_threshold())