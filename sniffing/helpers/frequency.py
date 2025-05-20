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


def _calc_freq(counts, bin_dur_ms):
    if counts == 0:
        return 0
    else:
        return 1 / (counts / (bin_dur_ms / 1000))


def moving_window_frequency(inhale_ts: pd.Series, trial_timestamps: pd.Series, window_size_ms: int=100, window_hop_ms: int=50):
    """

    :param inhale_ts:
    :param trial_timestamps:
    :param window_size_ms:
    :param window_hop_ms:
    :return:
    """
    pre_fv_ts_bins = pd.DataFrame(columns=['count', 'frequency'])
    post_fv_ts_bins = pd.DataFrame(columns=['count', 'frequency'])

    pre_fv_timestamps = trial_timestamps[trial_timestamps <= 0]
    post_fv_timestamps = trial_timestamps[trial_timestamps >= 0]

    win_end = pre_fv_timestamps[-1]
    win_start = win_end - window_size_ms

    while win_start >= pre_fv_timestamps[0]:
        num_current_bin = inhale_ts.between(win_start, win_end).sum()
        frequency = _calc_freq(num_current_bin, window_size_ms)
        pre_fv_ts_bins.loc[win_end, 'count'] = num_current_bin
        pre_fv_ts_bins.loc[win_end, 'frequency'] = frequency
        win_end -= window_hop_ms
        win_start = win_end - window_size_ms

    win_start = post_fv_timestamps[0]
    win_end = win_start + window_size_ms

    while win_end <= post_fv_timestamps[-1]:
        num_current_bin = inhale_ts.between(win_start, win_end).sum()
        frequency = _calc_freq(num_current_bin, window_size_ms)
        post_fv_ts_bins.loc[win_end, 'count'] = num_current_bin
        post_fv_ts_bins.loc[win_end, 'frequency'] = frequency
        win_start += window_hop_ms
        win_end = win_start + window_size_ms

    print(pre_fv_ts_bins, post_fv_ts_bins)