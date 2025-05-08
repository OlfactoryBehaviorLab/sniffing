from dewan_h5 import DewanH5
from . import preprocessing, plotting, analysis

import pandas as pd
import numpy as np
from scipy import signal
from tqdm.auto import tqdm


LOWER_FILTER_BAND = 0.01  # Lower Frequency (Hz)
UPPER_FILTER_BAND = 100  # Upper Frequency (Hz)
PRE_FV_TIME = -1000  # ms before FV
MAX_POST_FV_TIME = 2000 # ms after FV
BIN_SIZE = 50 # ms

def process_files(h5_files, output_dir, plot_figs=False):
    bins = np.arange(2*PRE_FV_TIME, MAX_POST_FV_TIME, BIN_SIZE)
    for h5_file_path in tqdm(h5_files, total=len(h5_files), desc='Processing H5 Files:'):
        try:
            print(f'Processing {h5_file_path.name}')

            with DewanH5(h5_file_path) as h5:
                results = pd.DataFrame()

                # For the blank experiments, the only concentration is zero
                if len(h5.concentrations) == 1:
                    _concentration = h5.concentrations
                else:
                    _concentration = h5.concentrations[h5.concentrations > 0][0]

                experiment_concentration = np.format_float_scientific(_concentration, 1)
                file_output_dir = output_dir.joinpath(f'mouse-{h5.mouse}', experiment_concentration)

                bp_filter = signal.cheby2(2, 40, [LOWER_FILTER_BAND, UPPER_FILTER_BAND], 'bandpass',
                                          output='sos', fs=1000)
                filtered_traces = preprocessing.filter_sniff_traces(h5.sniff, bp_filter, baseline=True, z_score=True)
                filtered_trial_names = list(filtered_traces.keys())
                inhale_bins = pd.DataFrame(0, index=filtered_trial_names, columns=bins)

                for trial_number in filtered_traces.keys():
                    # raw_data = h5.sniff[trial_number].loc[PRE_FV_TIME:]
                    filtered_trimmed_trace = filtered_traces[trial_number].loc[PRE_FV_TIME:]
                    # plotting.plot_multi_traces([raw_data, filtered_trimmed_trace])
                    inhales, exhales, crossings = preprocessing.get_trace_features(filtered_trimmed_trace)
                    crossing_pairs = np.fromiter(zip(crossings[:-1], crossings[1:]), dtype=object)

                    true_inhales, true_exhales = preprocessing.get_true_peaks(inhales, exhales, crossing_pairs)

                    true_inhales_post_fv = true_inhales.loc[0:]

                    if len(true_inhales_post_fv) == 0:
                        print(f'{trial_number} has no inhales after the FV!')
                        continue

                    first_true_inhale = true_inhales_post_fv.iloc[0]
                    first_crossing = first_true_inhale['crossing']

                    if first_crossing > 0:
                        crossings = preprocessing.offset_timestamps(first_crossing, filtered_trimmed_trace,
                                                                    true_inhales, true_exhales, crossings)

                    preprocessing.get_bin_counts(trial_number, true_inhales, inhale_bins)

                    inhale_frequencies, exhale_frequencies, inhale_times, exhale_times = analysis.calc_frequencies(
                        true_inhales, true_exhales)

                    _columns = pd.MultiIndex.from_product([[trial_number], ['inhale_time', 'inhale_freq']],
                                                          names=['Trial', 'Data'])
                    all_trial_data = pd.DataFrame(zip(inhale_times, inhale_frequencies), columns=_columns)

                    results = pd.concat([results, all_trial_data], axis=1)

                    plot_output_dir = file_output_dir.joinpath('figures')
                    plot_output_dir.mkdir(exist_ok=True, parents=True)
                    if plot_figs:
                        plotting.plot_crossing_frequencies(filtered_trimmed_trace, true_inhales, true_exhales,
                                                           inhale_frequencies, exhale_frequencies, inhale_times,
                                                           exhale_times, crossings, trial_number, plot_output_dir)

                trial_types = h5.trial_parameters.loc[filtered_trial_names, 'trial_type']
                trial_results = h5.trial_parameters.loc[filtered_trial_names, 'result']

                inhale_bins.insert(0, 'trial_type', trial_types)
                inhale_bins.insert(1, 'results', trial_results)

                results_path = file_output_dir.joinpath(f'mouse-{h5.mouse}-{experiment_concentration}.xlsx')
                results.to_excel(results_path)

                bins_path = file_output_dir.joinpath(f'mouse-{h5.mouse}-{experiment_concentration}-bins.xlsx')
                inhale_bins.to_excel(bins_path)

                h5.export(file_output_dir)

        except Exception as e:
            print(f'Error processing H5 file {h5_file_path}')
            print(e)