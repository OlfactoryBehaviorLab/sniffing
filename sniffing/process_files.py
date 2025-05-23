from dewan_h5 import DewanH5
from .helpers import preprocessing, frequency, plotting

import pandas as pd
import numpy as np
from scipy import signal
from tqdm.auto import tqdm


LOWER_FILTER_BAND = 0.01  # Lower Frequency (Hz)
UPPER_FILTER_BAND = 100  # Upper Frequency (Hz)
PRE_FV_TIME = -1000  # ms before FV
MAX_POST_FV_TIME = 2000 # ms after FV
BIN_SIZE = 50 # ms

NAN_THRESHOLD = 20

def process_files(h5_files, output_dir, plot_raw_traces=False, plot_figs=True, display_plots=False):
    bins = np.arange(2*PRE_FV_TIME, MAX_POST_FV_TIME, BIN_SIZE)

    for h5_file_path in tqdm(h5_files, total=len(h5_files), desc='Processing H5 Files:'):
        try:
            print(f'Processing {h5_file_path.name}')

            with DewanH5(h5_file_path, drop_early_lick_trials=True) as h5:
                go_trial_counts = pd.DataFrame()
                false_alarm_counts = pd.DataFrame()
                correct_rejection_counts = pd.DataFrame()
                missed_counts = pd.DataFrame()
                go_trial_freq = pd.DataFrame()
                false_alarm_freq = pd.DataFrame()
                correct_rejection_freq = pd.DataFrame()
                missed_freq = pd.DataFrame()

                file_output_dir = output_dir.joinpath(f'mouse-{h5.mouse}', h5.concentration)
                file_output_dir.mkdir(exist_ok=True, parents=True)

                bp_filter = signal.cheby2(2, 40, [LOWER_FILTER_BAND, UPPER_FILTER_BAND], 'bandpass',
                                          output='sos', fs=1000)
                filtered_traces = preprocessing.filter_sniff_traces(h5.sniff, bp_filter, baseline=True, z_score=True)

                filtered_trace_keys = list(filtered_traces.keys())

                for trial_number in tqdm(filtered_trace_keys, total=len(filtered_trace_keys)):
                    filtered_trimmed_trace = filtered_traces[trial_number].loc[PRE_FV_TIME:]

                    if plot_raw_traces:
                        raw_data = h5.sniff[trial_number].loc[PRE_FV_TIME:]
                        plotting.plot_multi_traces([raw_data, filtered_trimmed_trace], trial_number)

                    trial_result = h5.trial_parameters.loc[trial_number, 'result']

                    inhales, exhales, crossings = preprocessing.get_trace_features(filtered_trimmed_trace)
                    crossing_pairs = np.fromiter(zip(crossings[:-1], crossings[1:]), dtype=object)

                    true_inhales = preprocessing.filter_sniff_peaks(inhales, exhales)
                    true_inhales_post_fv = true_inhales.loc[0:]
                    if len(true_inhales_post_fv) == 0:
                        print(f'{trial_number} has no inhales after the FV!')
                        continue

                    _counts, _frequencies, _centers = frequency.static_window_frequency(
                        true_inhales.values,
                        filtered_trimmed_trace.index.values,
                        BIN_SIZE
                    )

                    bin_counts = pd.DataFrame(_counts, index=_centers, columns=['counts'])
                    bin_frequencies = pd.DataFrame(_frequencies, index=_centers, columns=['freq'])

                    if trial_result == 1:
                        go_trial_counts = pd.concat((go_trial_counts, bin_counts), axis=1)
                        go_trial_freq = pd.concat((go_trial_freq, bin_frequencies), axis=1)
                    elif trial_result == 2:
                        correct_rejection_counts = pd.concat((correct_rejection_counts, bin_counts), axis=1)
                        correct_rejection_freq = pd.concat((correct_rejection_freq, bin_frequencies), axis=1)
                    elif trial_result == 3:
                        false_alarm_counts = pd.concat((false_alarm_counts, bin_counts), axis=1)
                        false_alarm_freq = pd.concat((false_alarm_freq, bin_frequencies), axis=1)
                    elif trial_result == 5:
                        missed_counts = pd.concat((missed_counts, bin_counts), axis=1)
                        missed_freq = pd.concat((missed_freq, bin_frequencies), axis=1)

                    plot_output_dir = file_output_dir.joinpath('figures')
                    plot_output_dir.mkdir(exist_ok=True, parents=True)
                    if plot_figs:
                        # plotting.plot_crossing_frequencies(filtered_trimmed_trace, inhales, exhales,
                        #                                    inhale_frequencies, exhale_frequencies, inhale_times,
                        #                                    exhale_times, crossings, trial_number, plot_output_dir, display_plots)
                        plotting.plot_true_sniffs(filtered_trimmed_trace, true_inhales, inhales, exhales, crossings, trial_number, plot_output_dir, display_plots)

                go_trial_counts = go_trial_counts.fillna(0)
                false_alarm_counts = false_alarm_counts.fillna(0)
                correct_rejection_counts = correct_rejection_counts.fillna(0)
                missed_counts = missed_counts.fillna(0)

                mean_go_trial_counts = go_trial_counts.mean(axis=1)
                mean_false_alarm_counts = false_alarm_counts.mean(axis=1)
                mean_correct_rejection_counts = correct_rejection_counts.mean(axis=1)
                mean_missed_counts = missed_counts.mean(axis=1)

                go_trial_freq = go_trial_freq.fillna(0)
                false_alarm_freq = false_alarm_freq.fillna(0)
                correct_rejection_freq = correct_rejection_freq.fillna(0)
                missed_freq = missed_freq.fillna(0)

                mean_go_trial_freq = go_trial_freq.mean(axis=1)
                mean_false_alarm_freq = false_alarm_freq.mean(axis=1)
                mean_correct_rejection_freq = correct_rejection_freq.mean(axis=1)
                mean_missed_freq = missed_freq.mean(axis=1)

                if False:
                    fig, axs= plotting.plot_binned_frequencies(
                            [
                                    mean_go_trial_counts,
                                    mean_false_alarm_counts,
                                    mean_correct_rejection_counts,
                                    mean_missed_counts
                                    ],
                            [
                                    'Mean Go Trial',
                                    'Mean False Alarm',
                                    'Mean Correct Rejection',
                                    'Mean Missed'
                                    ],
                       [
                                    go_trial_counts.shape[1],
                                    false_alarm_counts.shape[1],
                                    correct_rejection_counts.shape[1],
                                    missed_counts.shape[1]
                                    ],
                        h5.mouse, h5.concentration, display_plots
                    )

                go_trial_counts.to_excel(file_output_dir.joinpath('go_trial_counts.xlsx'))
                false_alarm_counts.to_excel(file_output_dir.joinpath('false_alarm_counts.xlsx'))
                correct_rejection_counts.to_excel(file_output_dir.joinpath('correct_rejection_counts.xlsx'))
                missed_counts.to_excel(file_output_dir.joinpath('missed_counts.xlsx'))
                #fig.savefig(file_output_dir.joinpath(f'binned_frequency_hist.pdf'))

                h5.export(file_output_dir)


        except Exception as e:
            import traceback
            print(f'Error processing H5 file {h5_file_path}')
            print(traceback.format_exc())