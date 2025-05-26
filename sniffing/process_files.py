from dewan_h5 import DewanH5
from dewan_manual_curation import manual_curation
from dewan_manual_curation._components import analog_trace
from .helpers import preprocessing, frequency, plotting

import pandas as pd
import numpy as np
from scipy import signal
from tqdm.auto import tqdm

LOWER_FILTER_BAND = 0.01  # Lower Frequency (Hz)
UPPER_FILTER_BAND = 100  # Upper Frequency (Hz)
PRE_FV_TIME = -1000  # ms before FV
MAX_POST_FV_TIME = 2000 # ms after FV
BIN_SIZE = 100 # ms
BIN_STEPS = 50 # ms

NAN_THRESHOLD = 20

def process_files(h5_files, output_dir, run_manual_curation=False, filtered_manual_curation=True,
                  plot_figs=True, display_plots=False, ignore_errors=True):

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

                inhalation_durations = pd.DataFrame()

                file_output_dir = output_dir.joinpath(f'mouse-{h5.mouse}', h5.concentration)
                file_output_dir.mkdir(exist_ok=True, parents=True)

                bp_filter = signal.cheby2(2, 40, [LOWER_FILTER_BAND, UPPER_FILTER_BAND], 'bandpass',
                                          output='sos', fs=1000)
                filtered_traces = preprocessing.filter_sniff_traces(h5.sniff, bp_filter, baseline=True, z_score=True)

                filtered_trace_keys = list(filtered_traces.keys())

                if filtered_manual_curation:
                    gui_sniff_traces = analog_trace.AnalogTrace.generate_sniff_traces(filtered_trace_keys, h5, filtered_traces=filtered_traces)
                else:
                    gui_sniff_traces = analog_trace.AnalogTrace.generate_sniff_traces(filtered_trace_keys, h5)


                curated_traces = manual_curation.launch_sniff_gui(gui_sniff_traces, filtered_trace_keys)


                for trial_number in tqdm(filtered_trace_keys, total=len(filtered_trace_keys), leave=True, position=0):
                    filtered_trimmed_trace = filtered_traces[trial_number].loc[PRE_FV_TIME:]

                    if plot_raw_traces:
                        raw_data = h5.sniff[trial_number].loc[PRE_FV_TIME:]
                        plotting.plot_multi_traces([raw_data, filtered_trimmed_trace], trial_number)

                    trial_result = h5.trial_parameters.loc[trial_number, 'result']

                    inhales, exhales, crossings = preprocessing.get_trace_features(filtered_trimmed_trace)

                    true_inhales = preprocessing.filter_sniff_peaks(inhales, exhales)
                    flanking_exhales = preprocessing.get_flanking_exhales(true_inhales.index, exhales)

                    trial_inhalation_duration = preprocessing.get_inhalation_durations(true_inhales, flanking_exhales, filtered_trimmed_trace)
                    trial_inhalation_duration = trial_inhalation_duration.reset_index(names='timestamps')

                    new_columns = pd.MultiIndex.from_product([[str(trial_number)], trial_inhalation_duration.columns.values], names=['Trial', 'Data'])
                    trial_inhalation_duration.columns = new_columns
                    trial_inhalation_duration = trial_inhalation_duration.infer_objects().fillna('X')

                    inhalation_durations = pd.concat(
                        (inhalation_durations, trial_inhalation_duration), axis=1
                    )

                    true_inhales_post_fv = true_inhales.loc[0:]
                    if len(true_inhales_post_fv) == 0:
                        print(f'{trial_number} has no inhales after the FV!')
                        continue

                    _counts, _frequencies, _centers = frequency.static_window_frequency(
                        true_inhales.index.values,
                        filtered_trimmed_trace.index.values,
                        BIN_SIZE
                    )

                    bin_counts = pd.DataFrame(_counts, index=_centers, columns=[str(trial_number)])
                    bin_frequencies = pd.DataFrame(_frequencies, index=_centers, columns=[str(trial_number)])

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
                        plotting.plot_traces(raw_data, filtered_trimmed_trace, true_inhales, inhales, exhales, trial_number, plot_output_dir, display_plots )

                go_trial_counts = go_trial_counts.fillna(0)
                false_alarm_counts = false_alarm_counts.fillna(0)
                correct_rejection_counts = correct_rejection_counts.fillna(0)
                missed_counts = missed_counts.fillna(0)

                go_trial_counts['sum'] = go_trial_counts.sum(axis=1)
                false_alarm_counts['sum'] = false_alarm_counts.sum(axis=1)
                correct_rejection_counts['sum'] = correct_rejection_counts.sum(axis=1)
                missed_counts['sum'] = missed_counts.sum(axis=1)

                # Calculate mean excluding the sum column we just added
                go_trial_counts['mean'] = go_trial_counts[go_trial_counts.columns[:-1]].mean(axis=1)
                false_alarm_counts['mean'] = false_alarm_counts[false_alarm_counts.columns[:-1]].mean(axis=1)
                correct_rejection_counts['mean'] = correct_rejection_counts[correct_rejection_counts.columns[:-1]].mean(axis=1)
                missed_counts['mean'] = missed_counts[missed_counts.columns[:-1]].mean(axis=1)

                go_trial_freq = go_trial_freq.fillna(0)
                false_alarm_freq = false_alarm_freq.fillna(0)
                correct_rejection_freq = correct_rejection_freq.fillna(0)
                missed_freq = missed_freq.fillna(0)

                go_trial_freq['mean'] = go_trial_freq.mean(axis=1)
                false_alarm_freq['mean'] = false_alarm_freq.mean(axis=1)
                correct_rejection_freq['mean'] = correct_rejection_freq.mean(axis=1)
                missed_freq['mean'] = missed_freq.mean(axis=1)

            #     fig, axs= plotting.plot_binned_frequencies(
            #             [
            #                     mean_go_trial_counts,
            #                     mean_false_alarm_counts,
            #                     mean_correct_rejection_counts,
            #                     mean_missed_counts
            #                     ],
            #             [
            #                     'Mean Go Trial',
            #                     'Mean False Alarm',
            #                     'Mean Correct Rejection',
            #                     'Mean Missed'
            #                     ],
            #        [
            #                     go_trial_counts.shape[1],
            #                     false_alarm_counts.shape[1],
            #                     correct_rejection_counts.shape[1],
            #                     missed_counts.shape[1]
            #                     ],
            #         h5.mouse, h5.concentration, display_plots
            #     )

                go_trial_counts.to_excel(file_output_dir.joinpath('go_trial_counts.xlsx'))
                false_alarm_counts.to_excel(file_output_dir.joinpath('false_alarm_counts.xlsx'))
                correct_rejection_counts.to_excel(file_output_dir.joinpath('correct_rejection_counts.xlsx'))
                missed_counts.to_excel(file_output_dir.joinpath('missed_counts.xlsx'))
                inhalation_durations.to_excel(file_output_dir.joinpath('inhalation_durations.xlsx'))
                #fig.savefig(file_output_dir.joinpath(f'binned_frequency_hist.pdf'))

                h5.export(file_output_dir)


        except Exception as e:
            import traceback
            print(f'Error processing H5 file {h5_file_path}')
            print(traceback.format_exc())