from contextlib import closing
from functools import partial
from multiprocessing import Pool, Array, cpu_count, Process, freeze_support
from numpy import array, frombuffer, copyto, sort, mean
from numpy.random import normal
from ctypes import c_double
from numpy.random import normal

def worker(i, shape):
    print(f"Process {i + 1} started . . .")

    s = 0
    for v in range(shape[0]):
        # v = normal(0, 1, 10**4)
        # s += mean(v)
        s += mean(sort(frombuffer(SHARED_ARR.get_obj(), dtype=c_double, offset=v * shape[0], count=shape[1])))

    print(f"S = {s}")
    return s


def init(arr):
    global SHARED_ARR
    SHARED_ARR = arr


def run_test():
    """
        Shared memory test. Generates some large array and share it between
        processes. Uses pymbler to track object in the main scope. Checks
        size of shared wrapper in each process.

        NOTE : I reallocate another chunk of memory which is the main
        disadvantage. It's better to start writing to shared memory at the
        beginning row-by-row. When using already constructed arrays in python
        you have to reallocate another chunk as far as I'm concerned.

        """

    shape = (3 * 10 ** 4, 10 ** 4)  # array shape
    processes = cpu_count()  # size of the pool

    x = normal(0, 1, shape)
    print(f"X[0][0] = {x[0][0]}, X[0][-1] = {x[0][-1]}")
    print(f"Input array to share has size : {asizeof.asizeof(x) / 1024 / 1024 // 1} Mb")

    shared_wrapper = Array(c_double, shape[0] * shape[1], lock=True)
    copyto(frombuffer(shared_wrapper.get_obj(), dtype=c_double).reshape(shape), x)

    with closing(Pool(processes=processes, initializer=init, initargs=(shared_wrapper,))) as pool:
        result = pool.map(partial(worker, shape=shape), range(processes))

    print("\nMain Python process summary:")
    print(tr.print_diff())


if __name__ == '__main__':
    """
    Run from the console :  mprof run -M python memory_test.py
                            mprof plot
    """
    run_test()
