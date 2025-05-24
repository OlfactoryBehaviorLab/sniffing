"""
Dewan Sniffing Analysis Preprocessing Module
Austin Pauley, Dewan Lab, Florida State University, 2025
"""
import numpy as np
import pandas as pd
from scipy import signal, stats


def filter_sniff_traces(sniff_traces: dict, filter, baseline:bool=False, z_score:bool=False, shift:bool=False) -> dict:
    """
    Function takes a set of sniff traces and a Scipy signal filter and applies it to each trace.
    :param sniff_traces: (dict) Dictionary containing sniff traces
    :param filter: Scipy filter with 'sos' output type
    :param baseline: (bool) If true, apply a 1Hz highpass butterworth filter to remove any baseline drift from the trace 
    :param z_score: (bool) If true, zscore each sniff trace to itself
    :param shift: (bool) If true, the first index is subtracted from the data to make index [0] zero
    :return: (dict) filtered sniff traces
    """

    filtered_traces = {}
    baseline_filter = signal.butter(2, 1, 'highpass', output='sos', fs=1000)
    for name, trace in sniff_traces.items():
        index = trace.index
        # smoothed_data = signal.savgol_filter(trace, 3, 2)
        if trace.shape[0] == 0:
            print(f'{name} has no sniff data, skipping!')
            continue

        filtered_trace = signal.sosfiltfilt(filter, trace)
        if baseline:
            filtered_trace = signal.sosfiltfilt(baseline_filter, filtered_trace)
        if z_score:
            filtered_trace = stats.zscore(filtered_trace)
        if shift:
            filtered_trace = filtered_trace - filtered_trace[0]

        filtered_trace = pd.Series(filtered_trace, index=index)
        filtered_traces[name] = filtered_trace
    return filtered_traces


def get_trace_features(trace: pd.Series) -> tuple[pd.Series, pd.Series, np.array]:
    """

    :param trace:
    :return:
    """
    crossings = np.where(np.diff(np.signbit(trace)))[0]
    crossings = trace.index[crossings]

    # inhale_peak, props = signal.find_peaks(trace, distance=50, height=0.1)
    # # We can get the exhales by inverting the trace and running the same function
    # exhalation_peak, props_2 = signal.find_peaks(trace * -1, distance=50, height=0.1)

    exhalation_peak, _ = signal.find_peaks(trace * -1, distance=30, width=10, prominence=.1)
    inhale_peak, _ = signal.find_peaks(trace, distance=30, width=10, prominence=.1)


    inhale_x = trace.index[inhale_peak]
    inhale_y = trace.loc[inhale_x]
    inhales = pd.Series(inhale_x, index=inhale_y)

    exhale_x = trace.index[exhalation_peak]
    exhale_y = trace.loc[exhale_x]
    exhales = pd.Series(exhale_x, index=exhale_y)

    return inhales, exhales, crossings


def _pick_true_peaks(inhales, run_start, current_index):
    ambiguous_sniffs = inhales.iloc[run_start:current_index + 1]
    ambiguous_sniffs_magnitude = inhales.index[run_start:current_index + 1]
    max_sniff_magnitude = np.argmax(ambiguous_sniffs_magnitude)
    true_sniff_timestamp = ambiguous_sniffs.iloc[max_sniff_magnitude]
    true_sniff_magnitude = ambiguous_sniffs_magnitude[max_sniff_magnitude]
    return true_sniff_timestamp, true_sniff_magnitude


def filter_sniff_peaks(inhales: pd.Series, exhales: pd.Series):

    good_inhales = pd.DataFrame(columns=['magnitude'])

    inhale_index = 0
    run_start = 0

    while True:
        current_inhale = inhales.iloc[inhale_index]

        if inhale_index >= inhales.shape[0] - 1:  # Do while loop proxy
        # If we're at the last sniff, make sure it is also recorded
            if run_start == 0:
                good_inhales.loc[current_inhale, 'magnitude'] = inhales.index[inhale_index]
            else:
                true_sniff_timestamp, true_sniff_magnitude = _pick_true_peaks(inhales, run_start, inhale_index)
                good_inhales.loc[true_sniff_timestamp, 'magnitude'] = true_sniff_magnitude

            break

        next_inhale = inhales.iloc[inhale_index + 1]

        # Hi future Austin, this works by seeing if the first exhale after the current inhale, and the
        # first exhale before the next inhale are the same (using bitwise AND). If they are, the two inhales
        # must be separated by an exhale.

        if np.any(np.logical_and(current_inhale < exhales, exhales < next_inhale)):
            if run_start == 0:
                good_inhales.loc[current_inhale, 'magnitude'] = inhales.index[inhale_index]
            else:
                true_sniff_timestamp, true_sniff_magnitude = _pick_true_peaks(inhales, run_start, inhale_index)
                good_inhales.loc[true_sniff_timestamp, 'magnitude'] = true_sniff_magnitude
                run_start = 0
        else:
            if run_start == 0:
                run_start = inhale_index

        inhale_index += 1

    return good_inhales


def get_flanking_exhales(true_inhales: pd.Series, exhales: pd.Series):
    flanking_exhales = pd.DataFrame(index=true_inhales, columns=['pre', 'post', 'pre_mag', 'post_mag'])

    exhale_timestamps = exhales.values
    exhale_magnitudes = exhales.index

    for inhale in true_inhales:
        next_exhales = inhale < exhale_timestamps
        previous_exhales = exhale_timestamps < inhale

        if np.any(next_exhales):
            next_exhale = exhale_timestamps[next_exhales][0]
            next_exhale_mag = exhale_magnitudes[next_exhales][0]
        else:
            next_exhale = np.inf
            next_exhale_mag = np.inf

        if np.any(previous_exhales):
            previous_exhale = exhale_timestamps[previous_exhales][-1]
            previous_exhale_mag = exhale_magnitudes[previous_exhales][-1]
        else:
            previous_exhale = -np.inf
            previous_exhale_mag = -np.inf

        flanking_exhales.loc[inhale, 'pre'] = previous_exhale
        flanking_exhales.loc[inhale, 'post'] = next_exhale
        flanking_exhales.loc[inhale, 'pre_mag'] = previous_exhale_mag
        flanking_exhales.loc[inhale, 'post_mag'] = next_exhale_mag

    return flanking_exhales


def offset_timestamps(offset, trace, true_inhales, true_exhales, crossings):
    true_inhales.index = true_inhales.index - offset
    true_exhales.index = true_exhales.index - offset
    true_inhales.loc[:, 'inhale_start'] = true_inhales.loc[:, 'inhale_start'] - offset
    true_exhales.loc[:, 'exhale_start'] = true_exhales.loc[:, 'exhale_start'] - offset
    trace.index = trace.index - offset
    # Everything else is a dataframe that is directly edited, crossings has to be returned
    return crossings - offset


def get_inhalation_durations(true_inhales: pd.DataFrame, flanking_exhales: pd.DataFrame, sniff_trace):

    inhalation_durations = pd.DataFrame(index=true_inhales.index, columns=['duration', 'left_ts', 'right_ts'])

    for inhale, inhale_mag in true_inhales.iterrows():
        inhale_mag = inhale_mag.values
        pre_exhale_ts, post_exhale_ts, pre_exhale_mag, post_exhale_mag = flanking_exhales.loc[inhale].values

        # Sometimes we can't calculate the duration of the first/last sniff
        if np.isinf(pre_exhale_ts) or np.isinf(post_exhale_ts):
            inhalation_durations.loc[inhale] = np.nan
            continue

        # Instead of taking the midpoint time, lets get the middle of the magnitude incase it isn't linear
        left_sniff_trace_subset = sniff_trace.loc[pre_exhale_ts:inhale]
        right_sniff_trace_subset = sniff_trace.loc[inhale:post_exhale_ts]
        left_midpoint = (inhale_mag + pre_exhale_mag) / 2
        right_midpoint = (inhale_mag + post_exhale_mag) / 2

        left_timestamp = left_sniff_trace_subset.index[left_midpoint <= left_sniff_trace_subset.values][-1]
        right_timestamp = right_sniff_trace_subset.index[right_midpoint >= right_sniff_trace_subset.values][-1]

        duration = right_timestamp - left_timestamp

        inhalation_durations.loc[inhale, 'duration'] = duration
        inhalation_durations.loc[inhale, 'left_ts'] = left_timestamp
        inhalation_durations.loc[inhale, 'right_ts'] = right_timestamp

    return inhalation_durations



