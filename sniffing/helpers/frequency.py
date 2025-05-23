import numpy as np
import pandas as pd
from scipy import signal, fft
import matplotlib.pyplot as plt

from numba import njit, prange
from numba import types, typed

def _instantaneous_frequency(peaks):
    pairs = zip(peaks.iloc[:-1].index, peaks.iloc[1:].index)
    frequencies = []
    times = []

    for inhale1, inhale2 in pairs:
        delta_t = abs(inhale2 - inhale1)
        # frequency = round(1 / (delta_t / 1000), 2)
        frequency = round(_calc_freq(1, delta_t), 2)
        frequencies.append(frequency)
        times.append(inhale1 + (delta_t / 2))

    return frequencies, times


def calc_frequencies(true_inhales, true_exhales):
    inhale_frequencies, inhale_times = _instantaneous_frequency(true_inhales)
    exhale_frequencies, exhale_times = _instantaneous_frequency(true_exhales)

    return inhale_frequencies, exhale_frequencies, inhale_times, exhale_times


def twoside_moving_window_frequency(inhale_ts: pd.Series, trial_timestamps: pd.Series, window_size_ms: int=100, window_hop_ms: int=10):
    """

    :param inhale_ts:
    :param trial_timestamps:
    :param window_size_ms:
    :param window_hop_ms:
    :return:
    """
    ts_bins = pd.DataFrame(columns=['count', 'frequency'])

    pre_fv_timestamps = trial_timestamps[trial_timestamps <= 0]
    post_fv_timestamps = trial_timestamps[trial_timestamps >= 0]

    win_end = pre_fv_timestamps[-1]
    win_start = win_end - window_size_ms
    while win_start >= pre_fv_timestamps[0]:
        num_current_bin = inhale_ts.between(win_start, win_end).sum()
        frequency = _calc_freq(num_current_bin, window_size_ms)
        ts_bins.loc[win_end, 'count'] = num_current_bin
        ts_bins.loc[win_end, 'frequency'] = frequency
        win_end -= window_hop_ms
        win_start = win_end - window_size_ms

    win_start = post_fv_timestamps[0]
    win_end = win_start + window_size_ms
    while win_end <= (post_fv_timestamps[-1]):
        num_current_bin = inhale_ts.between(win_start, win_end).sum()
        frequency = _calc_freq(num_current_bin, window_size_ms)
        ts_bins.loc[win_start, 'count'] = num_current_bin
        ts_bins.loc[win_start, 'frequency'] = frequency
        win_start += window_hop_ms
        win_end = win_start + window_size_ms

    return ts_bins.sort_index()


@njit
def _calc_freq(counts, bin_dur_ms) -> float:
    try:
        return counts / (bin_dur_ms / 1000)
    except Exception:
        return 0.0


@njit(parallel=True)
def oneside_moving_window_frequency(inhale_ts: np.array, trial_timestamps: np.array, window_size_ms: int=100, window_step_ms: int=10):
    """

    :param inhale_ts:
    :param trial_timestamps:
    :param window_size_ms:
    :param window_step_ms:
    :return:
    """

    start_bins = np.arange(trial_timestamps[0], trial_timestamps[-1], window_step_ms)
    end_bins = start_bins + window_size_ms
    good_bin_indices = np.where(end_bins <= trial_timestamps[-1])[0]
    window_bins = np.array(list(zip(start_bins, end_bins)))[good_bin_indices]
    bin_centers = np.zeros(window_bins.shape[0])
    counts = np.zeros(window_bins.shape[0])
    frequencies = np.zeros(window_bins.shape[0])

    for i in prange(len(window_bins)):
        win_start, win_end = window_bins[i]

        num_sniffs = np.logical_and(win_end > inhale_ts, inhale_ts >= win_start).sum()
        frequency = _calc_freq(num_sniffs, window_size_ms)
        bin_centers[i] = (win_start + win_end) / 2
        counts[i] = num_sniffs
        frequencies[i] = frequency

    return bin_centers, counts, frequencies


@njit(parallel=True)
def static_window_frequency(inhale_ts: np.array, trial_timestamps: np.array, window_size_ms: int=100) -> tuple[np.array, np.array, np.array]:
    bin_starts = np.arange(trial_timestamps[0], trial_timestamps[-1], window_size_ms)
    bins = list(zip(bin_starts[:-1], bin_starts[1:]))

    bin_counts = np.zeros(len(bins))
    bin_freq = np.copy(bin_counts)
    bin_centers = np.copy(bin_counts)

    for i in prange(len(bins)):
        bin_start, bin_end = bins[i]
        bin_center = bin_start + window_size_ms / 2

        num_sniffs = np.logical_and(inhale_ts >= bin_start, inhale_ts < bin_end).sum()
        bin_counts[i] = num_sniffs
        bin_freq[i] = _calc_freq(num_sniffs, window_size_ms)
        bin_centers[i] = bin_center

    return bin_counts, bin_freq, bin_centers
