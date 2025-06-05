import pandas as pd
import numpy as np

from dewan_h5 import DewanH5

COLUMNS = [
    # Repeating Data
    "ID",
    "odor",
    "conc",
    # Trial Data
    "type",
    "result",
    # Counts
    "pre_odor_sniffs",
    "post_odor_sniffs"
    # Latencies
    "sniff_1_latency",
    "sniff_2_latency",
    "sniff_3_latency",
    # Durations
    "pre_sniff_dur_3",
    "pre_sniff_dur_2",
    "pre_sniff_dur_1",
    "post_sniff_dur_1",
    "post_sniff_dur_2",
    "post_sniff_dur_3",
]

TRIAL_TYPE = {1: "GO", 2: "NOGO"}


def repack_data(
    h5_file: DewanH5, inhale_counts, inhale_latencies, inhale_durations, output_dir
):
    animal_ID = h5_file.mouse
    odor = h5_file.odors
    odor = odor[odor != "blank"]
    concentration = h5_file.concentration
    trials = inhale_counts.columns.values

    trial_type = h5_file.trial_parameters.loc[trials, "trial_type"]
    trial_results = h5_file.trial_parameters.loc[trials, "result"]
    trial_type = trial_type.replace(TRIAL_TYPE)

    combined_df = pd.DataFrame(index=trials, columns=COLUMNS)
    combined_df.loc[:, "ID"] = animal_ID
    combined_df.loc[:"odor"] = odor
    combined_df.loc[:, "conc"] = concentration
    combined_df.loc[:, "type"] = trial_type
    combined_df.loc[:, "result"] = trial_results
    # inhale_counts.index = ['pre_odor_sniffs', 'post_odor_sniffs']
    combined_df.loc[:, ["pre_odor_sniffs", "post_odor_sniffs"]] = inhale_counts.T
    combined_df.loc[:, ["sniff_1_latency", "sniff_2_latency", "sniff_3_latency"]] = (
        inhale_latencies.T
    )

    _pre_fv_inhale_durations = pd.Series(np.zeros(3))
    _post_fv_inhale_durations = pd.Series(np.zeros(3))
    pre_fv_inhale_durations = inhale_durations.loc[inhale_durations.index < 0].iloc[-3:]
    post_fv_inhale_durations = inhale_durations.loc[inhale_durations.index >= 0].iloc[
        :3
    ]

    unpack_inhale_durations(inhale_durations, trials)

    _pre_fv_inhale_durations.iloc[: pre_fv_inhale_durations.shape[0]] = (
        pre_fv_inhale_durations
    )
    _post_fv_inhale_durations.iloc[: post_fv_inhale_durations.shape[0]] = (
        post_fv_inhale_durations
    )
    combined_df.loc[:, ["pre_sniff_dur_3", "pre_sniff_dur_2", "pre_sniff_dur_1"]] = (
        _pre_fv_inhale_durations
    )
    combined_df.loc[:, ["post_sniff_dur_1", "post_sniff_dur_2", "post_sniff_dur_3"]] = (
        _post_fv_inhale_durations
    )

    filename = f"{animal_ID}-{concentration}-combined.xlsx"
    output_path = output_dir.joinpath(filename)

    combined_df.to_excel(output_path)


def unpack_inhale_durations(inhale_durations, trials):
    _pre_fv_inhale_durations = pd.Series(np.zeros(3))
    _post_fv_inhale_durations = pd.Series(np.zeros(3))
    _inhale_durations = inhale_durations[[trials, "duration"]]

    return _pre_fv_inhale_durations, _post_fv_inhale_durations
