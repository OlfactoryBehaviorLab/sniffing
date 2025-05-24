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
    flanking_exhales = pd.DataFrame(index=true_inhales, columns=['pre', 'post'])

    for inhale in true_inhales:
        print(inhale)
        next_exhales = inhale < exhales
        previous_exhales = exhales < inhale

        if np.any(next_exhales):
            next_exhale = exhales[next_exhales][0]
        else:
            next_exhale = np.inf

        if np.any(previous_exhales):
            previous_exhale = exhales[previous_exhales][-1]
        else:
            previous_exhale = -np.inf

        flanking_exhales.loc[inhale, 'pre'] = previous_exhale
        flanking_exhales.loc[inhale, 'post'] = next_exhale

    return flanking_exhales


def get_true_peaks(inhales: pd.Series, exhales: pd.Series, crossing_pairs: np.array) -> tuple[pd.DataFrame, pd.DataFrame]:
    """

    :param inhales:
    :param exhales:
    :param crossing_pairs:
    :return:
    """
    true_inhales_x = []
    true_inhales_y = []
    true_exhales_x = []
    true_exhales_y = []

    inhales_prev_cross = []
    inhales_next_cross = []
    exhales_prev_cross = []
    exhales_next_cross = []
    for first_cross, second_cross in crossing_pairs:

        inhale_peaks_mask = inhales.between(first_cross, second_cross)
        exhale_peaks_mask = exhales.between(first_cross, second_cross)

        inhale_peaks = inhales.loc[inhale_peaks_mask]
        exhale_peaks = exhales.loc[exhale_peaks_mask]

        possible_inhale = None
        possible_exhale = None

        if inhale_peaks.shape[0] == 1:
            possible_inhale = inhale_peaks
        elif inhale_peaks.shape[0] > 1:
            max_val = inhale_peaks.index.max()
            max_inhale_peak = inhale_peaks.loc[max_val]
            possible_inhale = pd.Series(max_inhale_peak, index=[max_val])

        if exhale_peaks.shape[0] == 1:
            possible_exhale = exhale_peaks
        elif exhale_peaks.shape[0] > 1:
            max_val = exhale_peaks.index.min()
            max_exhale_peak = exhale_peaks.loc[max_val]
            possible_exhale = pd.Series(max_exhale_peak, index=[max_val])

        if possible_inhale is None and possible_exhale is None:
            continue
        elif possible_inhale is None:
            inhale = False
        elif possible_exhale is None:
            inhale = True
        elif abs(possible_exhale.index) > possible_inhale.index:
            inhale = False
        else:
            inhale = True

        if inhale:
            true_inhales_x.extend(possible_inhale.values)
            true_inhales_y.extend(possible_inhale.index.values)
            inhales_prev_cross.extend([first_cross])
            inhales_next_cross.extend([second_cross])
        else:
            true_exhales_x.extend(possible_exhale.values)
            true_exhales_y.extend(possible_exhale.index.values)
            exhales_prev_cross.extend([first_cross])
            exhales_next_cross.extend([second_cross])

    true_inhales = pd.DataFrame(
        {'magnitude': true_inhales_y, 'inhale_start': inhales_prev_cross, 'inhale_end': inhales_next_cross},
        index=true_inhales_x
    )
    true_exhales = pd.DataFrame(
        {'magnitude': true_exhales_y, 'exhale_start': exhales_prev_cross, 'exhale_end' :exhales_next_cross},
        index=true_exhales_x)

    return true_inhales, true_exhales


def offset_timestamps(offset, trace, true_inhales, true_exhales, crossings):
    true_inhales.index = true_inhales.index - offset
    true_exhales.index = true_exhales.index - offset
    true_inhales.loc[:, 'inhale_start'] = true_inhales.loc[:, 'inhale_start'] - offset
    true_exhales.loc[:, 'exhale_start'] = true_exhales.loc[:, 'exhale_start'] - offset
    trace.index = trace.index - offset
    # Everything else is a dataframe that is directly edited, crossings has to be returned
    return crossings - offset


def get_bin_counts(trial_number, true_inhales, inhale_bins_df):
    for inhale in true_inhales.index:
        inhale_bin_index = np.where(inhale >= inhale_bins_df.columns)[0][-1]
        inhale_bin = inhale_bins_df.columns[inhale_bin_index]
        inhale_bins_df.loc[trial_number, inhale_bin] += 1


def inhalation_durations(true_inhales: pd.DataFrame):
    durations = []
    for name, inhale in true_inhales.iterrows():
        _duration = inhale['inhale_end'] - inhale['inhale_start']
        durations.append(_duration)

    return pd.DataFrame(zip(true_inhales.index, durations), columns=['timestamp', 'duration'])



