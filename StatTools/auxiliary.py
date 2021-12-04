import operator
import time
from contextlib import closing
from multiprocessing import Value, Lock, Array, cpu_count, Pool
from threading import Thread
from numpy import ndarray, array, frombuffer, copyto, s_
from typing import Union
from ctypes import c_double, c_int64
from functools import reduce, partial
from operator import mul
import numpy


class SharedBuffer:
    """
    The instance of this class allows me manage shared memory. At this
    moment it supports only 1d, 2d arrays. Only numerical data. No need
    to declare global initializers for a pool.

    Basic usage:

        1. Suppose you have some array you want to share within your pool:

            shape = (10 ** 3, 10 ** 3)                  # just input shape
            some_arr = int64(normal(100, 30, shape))    # creating array

            s = SharedBuffer(shape, c_double)           # initialize buffer
            s.write(some_arr)                           # copy input data

        NOTE: IF YOU USE THIS METHOD 4th ROW DOUBLES MEMORY USAGE!

        2. You can copy data sequentially. It terms of memory it's way more
        efficient path. Wherever you want to extract data from the buffer
        you have to use this type of slicing:

            s[1:10, :] - get a portion
            s[7, 3]    - get a value

            I didn't really override __set_item__ and __get_item__ in the
            proper way.

        3. Having initialized the shared buffer ('s' in the example above)
        you can use this pattern to call from workers:

        def worker():
            handler = SharedBuffer.get("ARR")
            ...

        if __name__ = '__main__':

            shape = (10 ** 3, 10 ** 3)
            some_arr = int64(normal(100, 30, shape))
            s = SharedBuffer(shape, c_double)
            s.write(some_arr)

            with closing(Pool(processes=4, initializer=s.buffer_init,
                                    initargs=({"ARR":s}, ))) as pool:
                ...


    Methods available:

        s.apply(func, by_1st_dim=False) - allows to apply your function to
            entire buffer or for elements at the first dimension (by_1st_dim=True).
            This way you don't change the buffer itself, but you can get a result.

            s.apply(sum) - get the total sum of buffer

        s.apply_in_place(func, by_1st_dim=False) - same as 'apply', but changes the
            buffer.

        s.to_numpy() - gives you a simple numpy array.


    """

    def __init__(self, shape: tuple, dtype=Union[c_double, c_int64]):
        if len(shape) > 3:
            raise NotImplementedError("Only 1d, 2d- matrices are supported for now!")

        self.dtype, self.shape = dtype, shape
        self.offset = shape[1] if len(shape) == 2 else 1
        self.buffer = Array(dtype, reduce(mul, self.shape), lock=True)
        self.iter_counter = 0

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.__get_handle()[item]
        else:
            return self.__get_handle()[s_[item]]

    def __setitem__(self, key, value):

        if isinstance(key, int):
            self.__get_handle()[key] = value
        else:
            self.__get_handle()[s_[key]] = value

    def __repr__(self):
        return str(self.__get_handle())

    def __iter__(self):
        return self

    def __next__(self):
        while self.iter_counter < self.shape[0]:
            self.iter_counter += 1
            return self[self.iter_counter - 1]
        self.iter_counter = 0
        raise StopIteration

    def __del__(self):
        del self.buffer

    def __get_handle(self):
        return frombuffer(self.buffer.get_obj(), dtype=self.dtype).reshape(self.shape)

    def write(self, arr: ndarray, by_1_st_dim:bool=False) -> None:
        if arr.shape != self.shape:
            raise ValueError(f"Input array must have the same shape! arr: {arr.shape}")

        if by_1_st_dim:
            for i, v in enumerate(self.__get_handle()):
                v[:] = arr[i]
        else:
            copyto(self.__get_handle(), arr)

    def apply(self, func, by_1st_dim=False):
        result = []
        if by_1st_dim:
            for i, v in enumerate(self):
                result.append(func(v))
            return result
        else:
            return func(self.__get_handle().reshape(self.shape))

    # @profile
    def apply_in_place(self, func, by_1st_dim=False):
        if by_1st_dim:
            for i, v in enumerate(self):
                self[i] = func(v)
        else:
            self.to_array()[:] = func(self.to_array())

    def to_array(self):
        return self.__get_handle().reshape(self.shape)


    @staticmethod
    def buffer_init(vars_to_update):
        globals().update(vars_to_update)

    @classmethod
    def get(cls, name):
        return globals()[name]



class PearsonParallel:

    """

        КОД СТАРЫЙ, НЕ РЕФАКТОРИЛ!


    """




    def __init__(self, input_array):

        if isinstance(input_array, str):
            try:
                input_array = numpy.loadtxt(input_array)
            except OSError:
                error_str = "\n    The file either doesn't exit or you use a wrong path!"
                raise NameError(error_str)

        if isinstance(input_array, list):
            try:
                input_array = numpy.array(input_array)
            except numpy.VisibleDeprecationWarning:
                error_str = "\n    Error occurred when converting list to numpy array! " \
                            "\n    List probably has different dimensions!"
                raise NameError(error_str)

        if input_array.ndim == 1:
            error_str = "\n    PearsonParallel got 1-dimensional array !"
            raise NameError(error_str)

        if numpy.size(input_array) < pow(10, 5):
            print("\n    PearsonParallel Warning: Working in parallel mode with such small arrays is not effective !")

        self.arr = input_array

        if len(input_array) > 2:
            self.triangle_result = self.triangle_divider(len(input_array))
            self.working_ranges = self.triangle_result[0]
            self.cells_to_count = self.triangle_result[1]
            self.quantity = self.arr.shape[0]
            self.length = self.arr.shape[1]

    def create_matrix(self, threads=cpu_count(), progress_bar=False):

        if threads < 1:
            error_str = "\n    PearsonParallel Error: There is no point of calling this method using less than 2 " \
                        "threads since for loop is going to be faster!"
            raise NameError(error_str)

        if len(self.arr) == 2:
            return numpy.corrcoef(self.arr[0], self.arr[1])[0][1]

        shared_array = Array(c_double, self.quantity * self.length, lock=True)
        numpy.copyto(numpy.frombuffer(shared_array.get_obj()).reshape(self.arr.shape), self.arr)
        del self.arr

        result_matrix = Array(c_double, self.quantity * self.quantity, lock=True)

        bar_counter = Value('i', 0)
        bar_lock = Lock()



        with closing(Pool(processes=threads, initializer=self.global_initializer, initargs=(shared_array,
                                                                                            bar_counter,
                                                                                            bar_lock,
                                                                                            result_matrix))) as pool:
            pool.map(partial(self.corr_matrix, quantity=self.quantity, length=self.length), self.working_ranges)

        ans = numpy.frombuffer((result_matrix.get_obj())).reshape((self.quantity, self.quantity))
        for i, j in zip(range(len(ans)), range(len(ans))):
            ans[i][j] = 1.0
        return ans

    @staticmethod
    def triangle_divider(field_size):
        cpu_available = cpu_count()
        cells_to_count = (field_size * field_size - field_size) / 2

        cycle = int(cells_to_count / cpu_available)

        cells = []
        start = [0, 0]
        on_interval = 0

        for r1 in range(field_size):
            for r2 in range(field_size):

                if r1 >= r2:
                    continue
                else:

                    if on_interval <= cycle:
                        on_interval += 1
                    else:
                        cells.append([start, [r1, r2]])
                        start = [r1, r2]
                        on_interval = 0

        cells.append([cells[-1][1], [field_size - 1, field_size - 1]])

        return [cells, cells_to_count]

    @staticmethod
    def global_initializer(arr, bar_val, bar_lock, result_arr):
        global shared_array
        global counter
        global lock
        global matrix

        shared_array = arr
        counter = bar_val
        lock = bar_lock
        matrix = result_arr

    @staticmethod
    def corr_matrix(working_range, quantity, length):
        def get_row(index):
            return numpy.frombuffer(shared_array.get_obj()).reshape((quantity, length))[index]

        def write_to_matrix(value, r1, r2):
            numpy.frombuffer(matrix.get_obj()).reshape((quantity, quantity))[r1][r2] = value

        start = working_range[0]
        stop = working_range[1]

        iterations_buffer = 0

        for r1 in range(start[0], stop[0] + 1):

            if r1 == stop[0]:
                r2_stop = stop[1]
            else:
                r2_stop = quantity

            if r1 == start[0]:
                r2_start = start[1]
            else:
                r2_start = r1 + 1

            for r2 in range(r2_start, r2_stop):

                if r1 > r2:
                    continue
                else:
                    corr_value = numpy.corrcoef(get_row(r1), get_row(r2))[0][1]
                    write_to_matrix(corr_value, r1, r2)
                    write_to_matrix(corr_value, r2, r1)

                    iterations_buffer += 1

                    if iterations_buffer >= 250:
                        with lock:
                            counter.value += iterations_buffer
                            iterations_buffer = 0

        with lock:
            counter.value += iterations_buffer


class CheckNumpy:

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if isinstance(value, ndarray):
            instance.__dict__[self.name] = value
        elif isinstance(value, list):
            try:
                instance.__dict__[self.name] = array(value)
            except Exception:
                raise ValueError("Cannot cast input list to numpy array!")
        else:
            raise ValueError("Only list or numpy.ndarray can be used as input data!")

