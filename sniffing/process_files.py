from dewan_h5 import DewanH5
from dewan_manual_curation import manual_curation
from dewan_manual_curation._components import analog_trace
from .helpers import preprocessing, output, plotting
import pandas as pd
import numpy as np
from scipy import signal
from tqdm.auto import tqdm

LOWER_FILTER_BAND = 0.01  # Lower Frequency (Hz)
UPPER_FILTER_BAND = 100  # Upper Frequency (Hz)
PRE_FV_TIME = -1000  # ms before FV
MAX_POST_FV_TIME = 2000  # ms after FV
BIN_SIZE = 100  # ms
BIN_STEPS = 50  # ms

PRE_ODOR_COUNT_TIME_MS = -350
POST_ODOR_COUNT_TIME_MS = 350


def process_files(h5_files, output_dir, run_manual_curation=False, filtered_manual_curation=True,
                  plot_figs=False, display_plots=False, ignore_errors=False):
    for h5_file_path in tqdm(h5_files, total=len(h5_files), desc='Processing H5 Files:'):
        try:
            print(f'Processing {h5_file_path.name}')

            with DewanH5(h5_file_path, drop_early_lick_trials=True) as h5:
                inhale_counts = pd.DataFrame(index=[PRE_ODOR_COUNT_TIME_MS, POST_ODOR_COUNT_TIME_MS])
                inhale_latencies = pd.DataFrame(index=[1, 2, 3])
                inhale_durations = pd.DataFrame()

                file_output_dir = output_dir.joinpath(f'mouse-{h5.mouse}', h5.concentration)
                file_output_dir.mkdir(exist_ok=True, parents=True)

                bp_filter = signal.cheby2(2, 40, [LOWER_FILTER_BAND, UPPER_FILTER_BAND], 'bandpass',
                                          output='sos', fs=1000)
                filtered_traces = preprocessing.filter_sniff_traces(h5.sniff, bp_filter, baseline=True, z_score=True)

                filtered_trace_keys = list(filtered_traces.keys())

                if run_manual_curation:
                    if filtered_manual_curation:
                        gui_sniff_traces = analog_trace.AnalogTrace.generate_sniff_traces(filtered_trace_keys, h5,
                                                                                          filtered_traces=filtered_traces)
                    else:
                        gui_sniff_traces = analog_trace.AnalogTrace.generate_sniff_traces(filtered_trace_keys, h5)

                    filtered_trace_keys = manual_curation.launch_sniff_gui(gui_sniff_traces, filtered_trace_keys)

                for trial_number in tqdm(filtered_trace_keys, total=len(filtered_trace_keys), leave=True, position=0):
                    filtered_trimmed_trace = filtered_traces[trial_number].loc[PRE_FV_TIME:]
                    raw_data = h5.sniff[trial_number].loc[PRE_FV_TIME:]

                    # Find inhales/exhales and select true inhales
                    inhales, exhales, crossings = preprocessing.get_trace_features(filtered_trimmed_trace)
                    true_inhales = preprocessing.filter_sniff_peaks(inhales, exhales)

                    true_inhales_post_fv = true_inhales.loc[0:]
                    if len(true_inhales_post_fv) == 0:
                        print(f'{trial_number} has no inhales after the FV!')
                        continue

                    # Gather the flanking exhale for each inhale and find the inhalation durations
                    flanking_exhales = preprocessing.get_flanking_exhales(true_inhales.index, exhales)
                    trial_inhalation_duration = preprocessing.get_inhalation_durations(true_inhales, flanking_exhales,
                                                                                       filtered_trimmed_trace)
                    trial_inhalation_duration = trial_inhalation_duration.reset_index(names='timestamps')
                    new_columns = pd.MultiIndex.from_product(
                        [[str(trial_number)], trial_inhalation_duration.columns.values], names=['Trial', 'Data'])
                    trial_inhalation_duration.columns = new_columns
                    trial_inhalation_duration = trial_inhalation_duration.infer_objects().fillna('X')
                    inhale_durations = pd.concat(
                        (inhale_durations, trial_inhalation_duration), axis=1
                    )

                    pre_odor_sniffs = true_inhales.loc[PRE_ODOR_COUNT_TIME_MS:0]
                    post_odor_sniffs = true_inhales.loc[0:POST_ODOR_COUNT_TIME_MS]
                    pre_odor_sniff_count = pre_odor_sniffs.shape[0]
                    post_odor_sniff_count = post_odor_sniffs.shape[0]
                    _counts = pd.Series((pre_odor_sniff_count, post_odor_sniff_count), name=trial_number)

                    inhale_counts = pd.concat((inhale_counts, _counts), axis=1)

                    post_odor_latencies = pd.Series(np.zeros(3), index=[1, 2, 3], name=trial_number)
                    _latencies = post_odor_sniffs.index[:3]
                    post_odor_latencies.loc[:_latencies.shape[0]] = _latencies

                    inhale_latencies = pd.concat((inhale_latencies, post_odor_latencies), axis=1)

                    plot_output_dir = file_output_dir.joinpath('figures')
                    plot_output_dir.mkdir(exist_ok=True, parents=True)
                    if plot_figs:
                        plotting.plot_traces(raw_data, filtered_trimmed_trace, true_inhales, inhales, exhales,
                                             trial_number, plot_output_dir, display_plots)

                # go_trial_counts.to_excel(file_output_dir.joinpath('go_trial_counts.xlsx'))
                # false_alarm_counts.to_excel(file_output_dir.joinpath('false_alarm_counts.xlsx'))
                # correct_rejection_counts.to_excel(file_output_dir.joinpath('correct_rejection_counts.xlsx'))
                # missed_counts.to_excel(file_output_dir.joinpath('missed_counts.xlsx'))
                # inhalation_durations.to_excel(file_output_dir.joinpath('inhalation_durations.xlsx'))
                # fig.savefig(file_output_dir.joinpath(f'binned_frequency_hist.pdf'))
                output

                output.repack_data(h5, inhale_counts, inhale_latencies, inhale_durations, file_output_dir)
                h5.export(file_output_dir)


        except Exception as e:
            import traceback
            print(f'Error processing H5 file {h5_file_path}')
            print(traceback.format_exc())
            if not ignore_errors:
                raise e
