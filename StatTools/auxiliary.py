import operator
import time
from multiprocessing import Value, Lock, Array
from threading import Thread
from numpy import ndarray, array, frombuffer, copyto, s_
from typing import Union
from ctypes import c_double, c_int64
from functools import reduce
from operator import mul


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

