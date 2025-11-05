import logging

from dewan_h5 import DewanH5
from dewan_manual_curation import manual_curation
from dewan_manual_curation._components import analog_trace
from dewan_utils import async_io
from .helpers import frequency, preprocessing, output, plotting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm.auto import tqdm

LOWER_FILTER_BAND = 0.01  # Lower Frequency (Hz)
UPPER_FILTER_BAND = 100  # Upper Frequency (Hz)
PRE_FV_TIME = -1000  # ms before FV
MAX_POST_FV_TIME = 2000  # ms after FV
BIN_SIZE = 100  # ms
BIN_STEPS = 50  # ms

PRE_ODOR_COUNT_TIME_MS = -2000
POST_ODOR_COUNT_TIME_MS = 350

logging.basicConfig(level=logging.WARNING)
plt.set_loglevel(level="warning")


def process_files(
    h5_files,
    output_dir,
    run_manual_curation=False,
    filtered_manual_curation=True,
    plot_figs=False,
    display_plots=False,
    ignore_errors=False,
):
    logger = logging.getLogger(__name__)
    h5_stats = pd.DataFrame()
    processing_params = pd.DataFrame()
    tpe = async_io.AsyncIO(logger=logger)
    if not h5_files:
        logger.error("No H5 Files provided!")
        return None
    for h5_file_path in tqdm(
        h5_files, total=len(h5_files), desc="Processing H5 Files:"
    ):
        try:
            print(f"Processing {h5_file_path.name}")

            with DewanH5(
                h5_file_path, drop_early_lick_trials=False, drop_cheating_trials=True
            ) as h5:
                _total_original_trials = (
                    h5.total_trials
                    + len(h5.early_lick_trials)
                    + len(h5.short_trials)
                    + len(h5.missing_packet_trials)
                    + len(h5.cheat_check_trials)
                    + len(h5.zero_trials)
                )
                _perc_loss = round(
                    100
                    * (_total_original_trials - h5.total_trials)
                    / _total_original_trials,
                    2,
                )
                h5_trial_info = pd.Series(
                    (
                        h5.mouse,
                        h5.concentration,
                        h5.total_trials,
                        len(h5.early_lick_trials),
                        len(h5.short_trials),
                        len(h5.missing_packet_trials),
                        len(h5.cheat_check_trials),
                        len(h5.zero_trials),
                        _total_original_trials,
                        _perc_loss,
                    )
                )
                h5_stats = pd.concat((h5_stats, h5_trial_info), axis=1)

                params_output = pd.Series(
                    (
                        h5.mouse,
                        h5.concentration,
                        LOWER_FILTER_BAND,
                        UPPER_FILTER_BAND,
                        PRE_FV_TIME,
                        MAX_POST_FV_TIME,
                        BIN_SIZE,
                        BIN_STEPS,
                        PRE_ODOR_COUNT_TIME_MS,
                        POST_ODOR_COUNT_TIME_MS,
                    )
                )
                processing_params = pd.concat((processing_params, params_output), axis=1)


                inhale_counts = pd.DataFrame(
                    index=[PRE_ODOR_COUNT_TIME_MS, POST_ODOR_COUNT_TIME_MS]
                )
                inhale_latencies = pd.DataFrame(index=[1, 2, 3])
                inhale_durations = pd.DataFrame()

                binned_counts = pd.DataFrame()

                file_output_dir = output_dir.joinpath(
                    f"mouse-{h5.mouse}", h5.concentration
                )
                file_output_dir.mkdir(exist_ok=True, parents=True)

                bp_filter = signal.cheby2(
                    2,
                    40,
                    [LOWER_FILTER_BAND, UPPER_FILTER_BAND],
                    "bandpass",
                    output="sos",
                    fs=1000,
                )
                filtered_traces = preprocessing.filter_sniff_traces(
                    h5.sniff, bp_filter, baseline=True, z_score=True
                )

                filtered_trace_keys = list(filtered_traces.keys())

                if run_manual_curation:
                    if filtered_manual_curation:
                        gui_sniff_traces = (
                            analog_trace.AnalogTrace.generate_sniff_traces(
                                filtered_trace_keys, h5, filtered_traces=filtered_traces
                            )
                        )
                    else:
                        gui_sniff_traces = (
                            analog_trace.AnalogTrace.generate_sniff_traces(
                                filtered_trace_keys, h5
                            )
                        )

                    filtered_trace_keys = manual_curation.launch_sniff_gui(
                        gui_sniff_traces, filtered_trace_keys
                    )

                all_trimmed_traces = pd.DataFrame()

                for trial_number in tqdm(
                    filtered_trace_keys,
                    total=len(filtered_trace_keys),
                    leave=True,
                    position=0,
                ):
                    filtered_trimmed_trace = filtered_traces[trial_number].loc[
                        PRE_FV_TIME:
                    ]
                    raw_data = h5.sniff[trial_number].loc[PRE_FV_TIME:]

                    # Find inhales/exhales and select true inhales
                    inhales, exhales, crossings = preprocessing.get_trace_features(
                        filtered_trimmed_trace
                    )
                    true_inhales = preprocessing.filter_sniff_peaks(inhales, exhales)

                    true_inhales_post_fv = true_inhales.loc[0:]
                    if len(true_inhales_post_fv) == 0:
                        print(f"{trial_number} has no inhales after the FV!")
                        continue

                    # Gather the flanking exhale for each inhale and find the inhalation durations
                    flanking_exhales = preprocessing.get_flanking_exhales(
                        true_inhales.index, exhales
                    )
                    trial_inhalation_duration = preprocessing.get_inhalation_durations(
                        true_inhales, flanking_exhales, filtered_trimmed_trace
                    )
                    trial_inhalation_duration = trial_inhalation_duration.reset_index(
                        names="timestamps"
                    )
                    new_columns = pd.MultiIndex.from_product(
                        [
                            [str(trial_number)],
                            trial_inhalation_duration.columns.to_numpy(),
                        ],
                        names=["Trial", "Data"],
                    )
                    trial_inhalation_duration.columns = new_columns
                    trial_inhalation_duration = (
                        trial_inhalation_duration.infer_objects().fillna("X")
                    )
                    inhale_durations = pd.concat(
                        (inhale_durations, trial_inhalation_duration), axis=1
                    )

                    pre_odor_sniffs = true_inhales.loc[PRE_ODOR_COUNT_TIME_MS:0]
                    post_odor_sniffs = true_inhales.loc[0:POST_ODOR_COUNT_TIME_MS]
                    pre_odor_sniff_count = pre_odor_sniffs.shape[0]
                    post_odor_sniff_count = post_odor_sniffs.shape[0]
                    _counts = pd.Series(
                        (pre_odor_sniff_count, post_odor_sniff_count),
                        index=[PRE_ODOR_COUNT_TIME_MS, POST_ODOR_COUNT_TIME_MS],
                        name=trial_number,
                    )

                    _trimmed_timestamps = np.hstack(
                        (
                            pre_odor_sniffs.index.to_numpy(),
                            post_odor_sniffs.index.to_numpy(),
                        )
                    )
                    bin_centers, binned_sniff_counts, _ = (
                        frequency.oneside_moving_window_counts(
                            _trimmed_timestamps,
                            np.array((PRE_ODOR_COUNT_TIME_MS, POST_ODOR_COUNT_TIME_MS)),
                            BIN_SIZE,
                            BIN_STEPS,
                        )
                    )

                    trial_binned_counts = pd.Series(
                        binned_sniff_counts, index=bin_centers, name=trial_number
                    )
                    binned_counts = pd.concat(
                        (binned_counts, trial_binned_counts), axis=1
                    )

                    inhale_counts = pd.concat((inhale_counts, _counts), axis=1)

                    post_odor_latencies = pd.Series(
                        np.zeros(3), index=[1, 2, 3], name=trial_number
                    )
                    _latencies = post_odor_sniffs.index[:3]
                    post_odor_latencies.loc[: _latencies.shape[0]] = _latencies

                    inhale_latencies = pd.concat(
                        (inhale_latencies, post_odor_latencies), axis=1
                    )

                    plot_output_dir = file_output_dir.joinpath("figures")
                    plot_output_dir.mkdir(exist_ok=True, parents=True)
                    if plot_figs:
                        fig, path = plotting.plot_traces(
                            raw_data,
                            filtered_trimmed_trace,
                            h5.lick1[trial_number],
                            true_inhales,
                            exhales,
                            trial_number,
                            plot_output_dir,
                            display_plots,
                            tpe=True,
                        )
                        tpe.queue_save_plot(fig, path)
                    filtered_trimmed_trace = filtered_trimmed_trace.rename(trial_number)
                    all_trimmed_traces = pd.concat(
                        (all_trimmed_traces, filtered_trimmed_trace), axis=1
                    )

                trial_lengths = h5.trial_durations

                output.repack_data(
                    h5,
                    inhale_counts,
                    inhale_latencies,
                    inhale_durations,
                    trial_lengths,
                    binned_counts,
                    file_output_dir,
                    PRE_ODOR_COUNT_TIME_MS,
                    POST_ODOR_COUNT_TIME_MS,
                    tpe,
                )
                h5.export(file_output_dir)

                binned_sniff_counts_path = file_output_dir.joinpath(
                    "binned_sniff_counts.xlsx"
                )
                binned_counts = binned_counts.fillna("X").infer_objects()

                all_trimmed_traces = all_trimmed_traces.fillna("X").infer_objects()
                all_trimmed_traces_path = file_output_dir.joinpath(
                    "all_trimmed_traces.xlsx"
                )

                tpe.queue_save_df(binned_counts, binned_sniff_counts_path)
                tpe.queue_save_df(all_trimmed_traces, all_trimmed_traces_path)

        except Exception as e:
            import traceback

            print(f"Error processing H5 file {h5_file_path}")
            print(traceback.format_exc())
            if not ignore_errors:
                tpe.shutdown(wait=False)
                raise e

    h5_stats = h5_stats.T
    h5_stats.columns = [
        "ID",
        "CONC",
        "GOOD",
        "EARLY",
        "SHORT",
        "PACKETS",
        "CHEATCHK",
        "ZERO",
        "TOTAL",
        "PERCENT_LOSS",
    ]
    stats_output_path = file_output_dir.parent.joinpath("all_h5_stats.xlsx")
    tpe.queue_save_df(h5_stats, stats_output_path)

    processing_params = processing_params.T
    processing_params.columns = [
        "Animal",
        "Conc",
        "Lower Filter Band (Hz)",
        "Upper Filter Band (Hz)",
        "PRE FV Time",
        "MAX Post FV Time",
        "Count Bin Size",
        "Bin Step Size",
        "Pre Odor Count Time (mS)",
        "Post Odor Count Time (mS)"
    ]

    params_path = file_output_dir.parent.joinpath('all_processing_params.xlsx')
    tpe.queue_save_df(processing_params, params_path)

    tpe.shutdown(wait=True)
    return h5_stats
