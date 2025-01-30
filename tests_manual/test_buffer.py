import time
from contextlib import closing
from functools import partial
from multiprocessing import Pool

from memory_profiler import profile
from numpy import mean
from numpy.random import normal
from ctypes import c_double
from numpy.random import normal
from pympler import asizeof
from StatTools.auxiliary import SharedBuffer


def worker(i, shape, arr=None):
    print(f"\tProcess {i + 1} started . . .")

    arr = SharedBuffer.get('ARR') if arr is None else arr
    s = 0
    for v in range(shape[0]):
        # v = normal(0, 1, 10**4)
        # s += mean(v)
        s += mean(arr[v])

    return s


def init(arr):
    global SHARED_ARR
    SHARED_ARR = arr


def run_test_with_buffer(shape: tuple, processes:int):
    """
        Shared memory test. Generates some large array and share it between
        processes. Uses pymbler to track object in the main scope. Checks
        size of shared wrapper in each process.

        NOTE : I reallocate another chunk of memory which is the main
        disadvantage. It's better to start writing to shared memory at the
        beginning row-by-row. When using already constructed arrays in python
        you have to reallocate another chunk as far as I'm concerned.

        """

    x = normal(0, 1, shape)

    print("Started buffer test . . .")
    shared_array = SharedBuffer(shape, c_double)
    shared_array.write(x)

    with closing(Pool(processes=processes, initializer=shared_array.buffer_init, initargs=({'ARR': shared_array}, ))) \
            as pool:
        result = pool.map(partial(worker, shape=shape), range(processes))

    print("Buffer test is done!")


def run_test_raw_passing(shape:tuple, processes:int):

    x = normal(0, 1, shape)
    print("Starting raw passing test  . . .")

    with closing(Pool(processes=processes)) as pool:
        result = pool.map(partial(worker, shape=shape, arr=x), range(processes))

    print("Raw test is done!")

@profile
def test_reallocating(shape:tuple):
    x = normal(0, 1, shape)

    time.sleep(5)
    shared_arr = SharedBuffer(shape, c_double)
    shared_arr.write(x)
    time.sleep(3)

    print(f"Initial arr size: {asizeof.asizeof(x) //1024 //1024} Mb")


@profile
def test_reference(shape:tuple):

    buffer = SharedBuffer(shape, c_double)
    print(f"Buffer last elements: {buffer[-1, -6:-1]}")
    time.sleep(2)
    ref = buffer.to_array()

    for v in ref:
        v[:] = normal(0, 1, shape[1])

    print(f"Buffer last elements after populating with values: {buffer[-1, -6:-1]}")
    time.sleep(5)

    print("Test is done")



if __name__ == '__main__':
    """
    Run from the console :  mprof run -M python test_buffer.py
                            mprof plot
    """

    shape = (3 * 10 ** 4, 10 ** 4)  # array shape
    processes = 12

    # run_test_with_buffer(shape, processes)
    # time.sleep(5)
    # run_test_raw_passing(shape, processes)
    # time.sleep(5)
    # test_reallocating(shape)

    test_reference(shape)