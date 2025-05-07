from dewan_h5 import DewanH5
from sniffing_dynamics.sniffing import preprocessing, plotting, analysis

import pandas as pd
import numpy as np
from scipy import signal

import argparse
from pathlib import Path

LOWER_FILTER_BAND = 0.01  # Lower Frequency (Hz)
UPPER_FILTER_BAND = 100  # Upper Frequency (Hz)
PRE_FV_TIME = -1000  # ms before FV

def main():
    parser = argparse.ArgumentParser(description='Sniffing Analysis')
    parser.add_argument('-d', '--data_dir', help="Path to data directory")
    parser.add_argument('-o', '--output_dir', help="Path to output directory")
    args = parser.parse_args()

    if args.data_dir:
        _path = Path(args.data_dir)
        if not _path.exists():
            raise FileNotFoundError(f"Data directory {args.data_dir} does not exist")
        data_dir = _path
    else:
        data_dir = Path('/mnt/r2d2/11_Data/GoodSniffData')

    if args.output_dir:
        _path = Path(args.output_dir)
        if not _path.exists():
            raise FileNotFoundError(f"Output directory {args.output_dir} does not exist")
        output_dir = _path
    else:
        output_dir = data_dir.joinpath('output')

    h5_files = Path(data_dir).glob('*.h5')
    output_dir.mkdir(exist_ok=True)

    for h5_file_path in h5_files:
        try:
            print(f'Processing {h5_file_path.name}')

            with DewanH5(h5_file_path) as h5:
                results = pd.DataFrame()
                _concentration = h5.concentrations[h5.concentrations > 0][0]
                experiment_concentration = "".join(np.format_float_scientific(_concentration, 1).split('.'))
                file_output_dir = output_dir.joinpath(f'mouse-{h5.mouse}', experiment_concentration)

                bp_filter = signal.cheby2(2, 40, [lower_filter_band, upper_filter_band], 'bandpass', output='sos',
                                          fs=1000)
                filtered_traces = preprocessing.filter_sniff_traces(h5.sniff, bp_filter, baseline=True, z_score=True)

                for trial_number in h5.sniff.keys():
                    raw_data = h5.sniff[trial_number].loc[pre_fv_time:]
                    filtered_trimmed_trace = filtered_traces[trial_number].loc[pre_fv_time:]
                    # plotting.plot_multi_traces([raw_data, filtered_trimmed_trace])
                    inhales, exhales, crossings = preprocessing.get_trace_features(filtered_trimmed_trace)
                    crossing_pairs = np.fromiter(zip(crossings[:-1], crossings[1:]), dtype=object)

                    true_inhales, true_exhales = preprocessing.get_true_peaks(inhales, exhales, crossing_pairs)

                    true_inhales_post_fv = true_inhales.loc[0:]
                    first_true_inhale = true_inhales_post_fv.iloc[0]
                    first_crossing = first_true_inhale['crossing']

                    if first_crossing > 0:
                        crossings = preprocessing.offset_timestamps(first_crossing, filtered_trimmed_trace,
                                                                    true_inhales, true_exhales, crossings)
                    inhale_frequencies, exhale_frequencies, inhale_times, exhale_times = analysis.calc_frequencies(
                        true_inhales, true_exhales)

                    _columns = pd.MultiIndex.from_product([[trial_number], ['inhale_time', 'inhale_freq']],
                                                          names=['Trial', 'Data'])
                    all_trial_data = pd.DataFrame(zip(inhale_times, inhale_frequencies), columns=_columns)

                    results = pd.concat([results, all_trial_data], axis=1)

                    plot_output_dir = file_output_dir.joinpath('figures')
                    plot_output_dir.mkdir(exist_ok=True, parents=True)
                    plotting.plot_crossing_frequencies(filtered_trimmed_trace, true_inhales, true_exhales,
                                                       inhale_frequencies, exhale_frequencies, inhale_times,
                                                       exhale_times, crossings, trial_number, plot_output_dir)
                results_path = file_output_dir.joinpath(f'mouse-{h5.mouse}-{experiment_concentration}.xlsx')
                results.to_excel(results_path)
        except Exception:
            print(f'Error processing H5 file {h5_file_path}')


if __name__ == '__main__':
    main()