from contextlib import closing
from functools import reduce, partial
from operator import mul

from memory_profiler import profile
from numpy import array, frombuffer, ndarray, s_, copyto, int64, cumsum, transpose, sum, sqrt
from typing import Union
from ctypes import c_double, c_int64
from multiprocessing import Array, Pool
from numpy.random import normal
from pympler import asizeof


class FBMotion:

    def __init__(self, h, field_size):
        self.n = 2 ** field_size + 1

