import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd


def __dfa_polyfit_core(windowed: pd.Series, order: int) -> float:
    # Drop windows that contains less than needed for approximation
    # TODO: probably better to drop windows that contains less than some amount of non perfectly fitted points?
    if len(windowed.index) < order + 1:
        return []
    times = windowed.index.map(lambda x: x.value).to_numpy()
    z = np.polyfit(times, windowed.to_numpy(), order)
    current_trend = np.polyval(z, times)
    detrended_window = windowed - current_trend
    return list(detrended_window.values)


def __dfa_scale_core(
    dataset: pd.Series,
    scale: pd.Timedelta,
    step: pd.Timedelta,
    order: int,
    num_process=None,
) -> float:
    residuals = []
    start_time = dataset.index[0]
    start_times = []

    if step is None:
        start_times = list(dataset[: dataset.index[-1] - scale].index)
    else:
        while start_time < dataset.index[-1] - scale:
            start_time += step
            start_times.append(start_time)

    if len(start_times) == 0:
        print(f"No valid windows with scale {scale} found.")
        return np.nan

    if num_process is None:
        for start_time in start_times:
            residuals.extend(
                __dfa_polyfit_core(dataset[start_time : start_time + scale], order)
            )
    else:
        assert num_process > 0, "Invalid value for 'process'"
        num_process = min(num_process, len(start_times))
        with Pool(num_process) as p:
            residuals = list(
                itertools.chain(
                    *list(
                        p.starmap(
                            __dfa_polyfit_core,
                            [
                                (dataset[start_time : start_time + scale].copy(), order)
                                for start_time in start_times
                            ],
                        )
                    )
                )
            )
    residuals = np.array(residuals)
    F_2 = np.mean(residuals**2)
    return F_2


def dfa(
    data: pd.Series,
    scales: list[pd.Timedelta],
    step: float = None,
    order: int = 2,
    num_process=None,
) -> pd.Series:
    """Calculate the Detrended Fluctuation Analysis (DFA)
    for a given nonequidistant dataset.

    The timeseries for the analysis should coutains a measurement of the value, but not increments, as usualy
    used for fluctuation analysis.

    WARNING: Current implementation can be unstable for h<1,
    if the data contains a lot of gaps in the data (~25% of the equivalent equidistant data).

    Args:
        data (pd.Series): The input data to analyze.
            The index of the Series must be timeseries.
        scales (list[pd.Timedelta]): A list of scales to use for the DFA analysis.
            Each scale should be a timedelta object representing the time interval.
        step (int): The step size to use for the DFA analysis.
            This is used to determine the number of windows to use for each scale.
        order (int, optional): The polynomial order used for detrending. Defaults to 2.
        num_process (int): The number of processes to use for parallel computation. Defaults to None.
            If None, the computation will be done sequentially.
    Returns:
        pd.Series: A series containing the DFA values, i.e., the squared Fluctuation Function.
    """
    # Ensure the index is a valid timeseries
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The index of the Series must be a valid timeseries")

    FF_2 = []

    for scale in scales:
        if step is not None:
            step_scale = step * scale
        else:
            step_scale = None
        FF_2.append(__dfa_scale_core(data, scale, step_scale, order, num_process))

    return pd.Series(FF_2, index=scales)
