import numpy as np
from typing import Union, Iterable, Tuple

def _fa_worker(array: np.ndarray, step: float, S:float) -> float:
    """Core of FA algorithm. 

    Args:
        arrays (np.ndarray): preprocessed (integrated) arrays
        step (float): share of S - value. It's set usually as 0.5. The integer part of the number will be taken
        S (float): Scale to compute

    Returns:
        float: F(S)
    """
    V = np.arange(0, array.shape[1] - S, max(int(step * S), 1))
    Fv = np.zeros((array.shape[0], len(V)), dtype=float)
    for v_i, v in enumerate(V):
        Fv[:, v_i] = array[:, v] - array[:, v + S]
    F2 = np.mean(Fv**2,axis=1)
    return F2


def fa(arr: np.ndarray, 
       step: float, 
       s: Union[int, Iterable], 
       n_integral=1) -> Tuple[np.ndarray, np.ndarray]:
    """Execute Fluctuational Analysis for time series

    Basic usage:
        You can get whole F(s) function for first vector as:
        ```python
            from StatTools.analysis import fa
            s_vals = [i**2 for i in range(1, 5)]
            F, Scales = fa(input_array, 0.5, s_vals)
        ```
    Args:
        arr (np.ndarray): dataset array(s), If multiple signals, the first dimendtion is signal index (N, Samples).
        step (float): share of S - value. It's set usually as 0.5. The integer part of the number will be taken
        s (Union[int, Iterable]): scales where fluctuation function F^2(s) should be calculated.
        processes (int, optional): num of workers to spawn. Defaults to 1.
        gc_params (tuple, optional): _description_. Defaults to None.
        short_vectors (bool, optional): _description_. Defaults to False.
        n_integral (int, optional): _description_. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: F^2(S), Scales
    """
    if len(arr.shape) > 2:
        raise ValueError(f"Unsupported dimention of input signals array: expected 1 or 2, got {arr.shape}")
    
    if len(arr.shape) == 1:
        cumsum_arr = arr[np.newaxis, :]    
    else:
        cumsum_arr = arr

    if isinstance(s, Iterable):
        init_s_len = len(s)

        s = list(filter(lambda x: x <= cumsum_arr.shape[1] / 4, s))
        if len(s) < 1:
            raise ValueError("All input S values are larger than vector shape / 4 !")

        if len(s) != init_s_len:
            print(f"\tFA warning: only following S values are in use: {s}")

    elif isinstance(s, (float, int)):
        if s > cumsum_arr.shape[1] / 4:
            raise ValueError("Cannot use S > L / 4")
        s = (s,)

    s_current = [s] if not isinstance(s, Iterable) else s

    
    for _ in range(n_integral):
        cumsum_arr = np.cumsum(cumsum_arr, axis=1)

    F = np.zeros((cumsum_arr.shape[0], len(s_current)), dtype=float)

    for s_idx, s in enumerate(s_current):
        F[:, s_idx] = _fa_worker(cumsum_arr, step, s)

    return (
        F[0] if len(arr.shape) == 1 else F,
        s_current
    )


