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

def process_files(h5_files, output_dir, plot_figs=True):
    bins = np.arange(2*PRE_FV_TIME, MAX_POST_FV_TIME, BIN_SIZE)
    for h5_file_path in tqdm(h5_files, total=len(h5_files), desc='Processing H5 Files:'):
        try:
            print(f'Processing {h5_file_path.name}')

            with DewanH5(h5_file_path, drop_early_lick_trials=True) as h5:
                results = pd.DataFrame()
                all_inhalation_durations = pd.DataFrame()

                go_trial_ts_bins = pd.DataFrame()
                false_alarm_ts_bins = pd.DataFrame()
                correct_rejection_ts_bins = pd.DataFrame()
                missed_ts_bins = pd.DataFrame()

                file_output_dir = output_dir.joinpath(f'mouse-{h5.mouse}', h5.concentration)
                file_output_dir.mkdir(exist_ok=True, parents=True)

                bp_filter = signal.cheby2(2, 40, [LOWER_FILTER_BAND, UPPER_FILTER_BAND], 'bandpass',
                                          output='sos', fs=1000)
                filtered_traces = preprocessing.filter_sniff_traces(h5.sniff, bp_filter, baseline=True, z_score=True)
                filtered_trial_names = list(filtered_traces.keys())
                inhale_bins = pd.DataFrame(0, index=filtered_trial_names, columns=bins)

                filtered_trace_keys = list(filtered_traces.keys())

                for trial_number in tqdm(filtered_trace_keys, total=len(filtered_trace_keys)):
                    filtered_trimmed_trace = filtered_traces[trial_number].loc[PRE_FV_TIME:]
                    # raw_data = h5.sniff[trial_number].loc[PRE_FV_TIME:]
                    # plotting.plot_multi_traces([raw_data, filtered_trimmed_trace])

                    trial_result = h5.trial_parameters.loc[trial_number, 'result']

                    inhales, exhales, crossings = preprocessing.get_trace_features(filtered_trimmed_trace)
                    crossing_pairs = np.fromiter(zip(crossings[:-1], crossings[1:]), dtype=object)

                    true_inhales, true_exhales = preprocessing.get_true_peaks(inhales, exhales, crossing_pairs)

                    inhalation_durations = preprocessing.inhalation_durations(true_inhales)

                    true_inhales_post_fv = true_inhales.loc[0:]

                    if len(true_inhales_post_fv) == 0:
                        print(f'{trial_number} has no inhales after the FV!')
                        continue

                    preprocessing.get_bin_counts(trial_number, true_inhales, inhale_bins)

                    true_inhales_ts = pd.Series(true_inhales.index)
                    bin_centers, counts, frequencies = frequency.oneside_moving_window_frequency(true_inhales_ts.values, filtered_trimmed_trace.index.values)
                    ts_bins = pd.DataFrame(zip(counts, frequencies), index=bin_centers, columns=['counts', 'frequency'])

                    if trial_result == 1:
                        go_trial_ts_bins = pd.concat((go_trial_ts_bins, ts_bins['frequency']), axis=1)
                    elif trial_result == 2:
                        correct_rejection_ts_bins = pd.concat((correct_rejection_ts_bins, ts_bins['frequency']), axis=1)
                    elif trial_result == 3:
                        false_alarm_ts_bins = pd.concat((false_alarm_ts_bins, ts_bins['frequency']), axis=1)
                    elif trial_result == 5:
                        missed_ts_bins = pd.concat((missed_ts_bins, ts_bins['frequency']), axis=1)

                    inhale_frequencies, exhale_frequencies, inhale_times, exhale_times = frequency.calc_frequencies(
                        true_inhales, true_exhales)


                    _columns = pd.MultiIndex.from_product([[trial_number], ['inhale_time', 'inhale_freq']],
                                                          names=['Trial', 'Data'])
                    all_trial_data = pd.DataFrame(zip(inhale_times, inhale_frequencies), columns=_columns)

                    results = pd.concat([results, all_trial_data], axis=1)

                    _columns = pd.MultiIndex.from_product([[trial_number], ['timestamp', 'duration']], names=['Trial', 'Inhales'])
                    inhalation_durations.columns = _columns
                    all_inhalation_durations = pd.concat([all_inhalation_durations, inhalation_durations], axis=1)

                    plot_output_dir = file_output_dir.joinpath('figures')
                    plot_output_dir.mkdir(exist_ok=True, parents=True)
                    if plot_figs:
                        plotting.plot_crossing_frequencies(filtered_trimmed_trace, inhales, exhales,
                                                           inhale_frequencies, exhale_frequencies, inhale_times,
                                                           exhale_times, crossings, trial_number, plot_output_dir)

                trial_types = h5.trial_parameters.loc[filtered_trial_names, 'trial_type']
                trial_results = h5.trial_parameters.loc[filtered_trial_names, 'result']

                inhale_bins.insert(0, 'trial_type', trial_types)
                inhale_bins.insert(1, 'results', trial_results)

                results_path = file_output_dir.joinpath(f'mouse-{h5.mouse}-{h5.concentration}.xlsx')
                results.to_excel(results_path)

                bins_path = file_output_dir.joinpath(f'mouse-{h5.mouse}-{h5.concentration}-bins.xlsx')
                inhale_bins.to_excel(bins_path)

                inhalation_path = file_output_dir.joinpath(f'mouse-{h5.mouse}-{h5.concentration}-inhalation_durations.xlsx')
                all_inhalation_durations.to_excel(inhalation_path)

                go_trial_ts_bins = go_trial_ts_bins.dropna(axis=0)
                false_alarm_ts_bins = false_alarm_ts_bins.dropna(axis=0)
                correct_rejection_ts_bins = correct_rejection_ts_bins.dropna(axis=0)
                missed_ts_bins = missed_ts_bins.dropna(axis=0)

                mean_go_trial_ts_bins = go_trial_ts_bins.mean(axis=1)
                mean_false_alarm_ts_bins = false_alarm_ts_bins.mean(axis=1)
                mean_correct_rejection_ts_bins = correct_rejection_ts_bins.mean(axis=1)
                mean_missed_ts_bins = missed_ts_bins.mean(axis=1)

                # mean_go_trial_ts_bins = go_trial_ts_bins.sum(axis=1) / go_trial_ts_bins.shape[1]
                # mean_false_alarm_ts_bins = false_alarm_ts_bins.sum(axis=1) / false_alarm_ts_bins.shape[1]
                # mean_correct_rejection_ts_bins = correct_rejection_ts_bins.sum(axis=1) / correct_rejection_ts_bins.shape[1]
                # mean_missed_ts_bins = missed_ts_bins.sum(axis=1) / missed_ts_bins.shape[1]



                fig, axs= plotting.plot_binned_frequencies(
                        [
                                mean_go_trial_ts_bins,
                                mean_false_alarm_ts_bins,
                                mean_correct_rejection_ts_bins,
                                mean_missed_ts_bins
                                ],
                        [
                                'Mean Go Trial',
                                'Mean False Alarm',
                                'Mean Correct Rejection',
                                'Mean Missed'
                                ],
                   [
                                go_trial_ts_bins.shape[1],
                                false_alarm_ts_bins.shape[1],
                                correct_rejection_ts_bins.shape[1],
                                missed_ts_bins.shape[1]
                                ],
                    h5.mouse, h5.concentration
                )

                mean_go_trial_ts_bins.to_excel(file_output_dir.joinpath('mean_go_trial_ts_bins.xlsx'))
                mean_false_alarm_ts_bins.to_excel(file_output_dir.joinpath('mean_false_alarm_ts_bins.xlsx'))
                mean_correct_rejection_ts_bins.to_excel(file_output_dir.joinpath('mean_correct_rejection_ts_bins.xlsx'))
                mean_missed_ts_bins.to_excel(file_output_dir.joinpath('mean_missed_ts_bins.xlsx'))
                fig.savefig(file_output_dir.joinpath(f'binned_frequency_hist.pdf'))

                h5.export(file_output_dir)


        except Exception as e:
            print(f'Error processing H5 file {h5_file_path}')
            raise e