from functools import partial
from typing import Optional

from numpy import uint8, array, max, min, ndarray, zeros
from numpy.random import randn
from math import floor
from ctypes import c_double
from StatTools.auxiliary import SharedBuffer
from random import gauss
from C_StatTools import fbm_core


def add_h_values(vector: ndarray, k: int, h: float):
    return array([v + (pow(0.5, k * (h - 1)) * gauss(0, 1)) if v != 0 else 0 for v in vector])


def quant_array(vector: ndarray, min_val: float, max_val: float):
    return ((vector - min_val) / (max_val - min_val) * 255).astype(uint8)


# @profile()
def fb_motion_python(h: float, field_size: int):
    """
    This is the algorithm. It need C version for sure.
    """

    n = 2 ** field_size + 1
    shape = n, n

    F = SharedBuffer(shape, c_double)

    F[0, 0], F[0, -1], F[-1, 0], F[-1, -1] = randn(4)
    for k in range(1, field_size + 1):
        m = 2 ** k

        fl = floor(n / m)

        l1 = fl
        s = fl * 2
        l2 = floor((m - 1) * n / m) + 1

        for i in range(l1, l2, s):
            for j in range(l1, l2, s):
                v1 = F[i - fl, j - fl]
                v2 = F[i - fl, j + fl]
                v3 = F[i + fl, j - fl]
                v4 = F[i + fl, j + fl]

                F[i, j] = (v1 + v2 + v3 + v4) / 4

        for i in range(0, n + 1, s):
            for j in range(fl, l2, s):
                F[i, j] = (F[i, j - fl] + F[i, j + fl]) / 2

        for j in range(0, n + 1, s):
            for i in range(fl, l2, s):
                F[i, j] = (F[i - fl, j] + F[i + fl, j]) / 2

        F.apply_in_place(func=partial(add_h_values, k=k, h=h), by_1st_dim=True)

    max_val = max(F.to_array())
    min_val = min(F.to_array())
    F.apply_in_place(func=partial(quant_array, min_val=min_val, max_val=max_val), by_1st_dim=True)

    z = array(F.to_array(), dtype=uint8)
    return z


# @profile
def fb_motion(h: float, field_size: int, filter_mine: Optional[ndarray] = None) -> ndarray:
    """
    This is the same algorithm as fb_motion_python but with C compiled core.
    In average you can get up to 10x performance boost using this version
    over python one.

    Basic usage:

        result = fb_motion(1.5, 10)      # where H = 1.5 and field is 2^10+1

        im = Image.fromarray(result)    # now you can save the image
        im.save("filename.jpeg")

    The result is quantized array (numpy.ndarray) that can be represented as image.

    You can filter you own array:

        my_arr = zeros((2**12 + 1, 2**12 + 1))          # size is supposed to be 2^N + 1
        result = fb_motion(1.5, 12, filter_mine = my_arr)

    """

    if filter_mine is None:
        n = 2 ** field_size + 1
        zeros_arr = zeros((n, n), dtype=float)
        fbm_core(zeros_arr, h, field_size)
        return zeros_arr.astype(uint8)
    else:
        print("HERE")
        shape = filter_mine.shape
        if filter_mine.ndim == 1 or shape[0] != shape[1]:
            raise ValueError("Cannot process such input array!")
        if 2**field_size > shape[0]:
            raise ValueError("2^degree > input array shape. You either use less or equal.")
        fbm_core(filter_mine, h, field_size)
        return filter_mine.astype(uint8)


if __name__ == '__main__':
    # t1 = time.perf_counter()
    # r = FBMotion_python(1.6, 10)
    # print(f"Took: {time.perf_counter() - t1}")
    # im = Image.fromarray(r)
    # im.save("filename.jpeg")

    arr = fb_motion(0.5, 12)

    # some_func()
