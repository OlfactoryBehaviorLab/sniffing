import numpy as np
import pandas as pd
from scipy import signal, fft
import matplotlib.pyplot as plt

def _instantaneous_frequency(peaks):
    pairs = zip(peaks.iloc[:-1].index, peaks.iloc[1:].index)
    frequencies = []
    times = []

    for inhale1, inhale2 in pairs:
        delta_t = abs(inhale2 - inhale1)
        frequency = round(1 / (delta_t / 1000), 2)
        frequencies.append(frequency)
        times.append(inhale1 + (delta_t / 2))

    return frequencies, times


def calc_frequencies(true_inhales, true_exhales):
    inhale_frequencies, inhale_times = _instantaneous_frequency(true_inhales)
    exhale_frequencies, exhale_times = _instantaneous_frequency(true_exhales)

    return inhale_frequencies, exhale_frequencies, inhale_times, exhale_times


def moving_window_frequency(inhale_ts: pd.Series, trial_timestamps: pd.Series, window_size: int=100, window_hop: int=50):
    """

    :param inhale_ts:
    :param trial_timestamps:
    :param window_size:
    :param window_hop:
    :return:
    """
    ts_bins = pd.DataFrame()

    pre_fv_timestamps = trial_timestamps.loc[:0]
    post_fv_timestamps = trial_timestamps.loc[0:]

    num_pre_fv_bins = (len(pre_fv_timestamps) - window_size) // window_hop
    pre_fv_bin_starts = np.linspace(pre_fv_timestamps[0], (pre_fv_timestamps[-1] - window_size), num_pre_fv_bins)

    num_post_fv_bins = (len(post_fv_timestamps) - window_size) // window_hop
    post_fv_bin_starts = np.linspace(post_fv_timestamps[0], (post_fv_timestamps[-1] - window_size), num_post_fv_bins)

