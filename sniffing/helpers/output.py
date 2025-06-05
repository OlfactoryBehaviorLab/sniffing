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
    "post_odor_sniffs",
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
    trials = inhale_counts.columns.to_numpy()

    trial_type = h5_file.trial_parameters.loc[trials, "trial_type"]
    trial_results = h5_file.trial_parameters.loc[trials, "result"]
    trial_type = trial_type.replace(TRIAL_TYPE)

    combined_df = pd.DataFrame(index=trials, columns=COLUMNS)
    combined_df.loc[:, "ID"] = animal_ID
    combined_df.loc[:, "odor"] = odor
    combined_df.loc[:, "conc"] = concentration
    combined_df.loc[:, "type"] = trial_type
    combined_df.loc[:, "result"] = trial_results
    # inhale_counts.index = ['pre_odor_sniffs', 'post_odor_sniffs']

    inhale_counts.index = ["pre_odor_sniffs", "post_odor_sniffs"]
    inhale_latencies.index = ["sniff_1_latency", "sniff_2_latency", "sniff_3_latency"]

    combined_df.loc[:, ["pre_odor_sniffs", "post_odor_sniffs"]] = inhale_counts.T
    combined_df.loc[:, ["sniff_1_latency", "sniff_2_latency", "sniff_3_latency"]] = (
        inhale_latencies.T
    )
    pre_fv_inhalation_durations, post_fv_inhalation_durations = unpack_inhale_durations(inhale_durations, trials)
    combined_df.loc[:, ["pre_sniff_dur_3","pre_sniff_dur_2","pre_sniff_dur_1",]] = pre_fv_inhalation_durations
    combined_df.loc[:, ["post_sniff_dur_1", "post_sniff_dur_2", "post_sniff_dur_3",]] = post_fv_inhalation_durations

    filename = f"{animal_ID}-{concentration}-combined.xlsx"
    output_path = output_dir.joinpath(filename)

    combined_df = combined_df.infer_objects().fillna('X')

    combined_df.to_excel(output_path)


def unpack_inhale_durations(inhale_durations, trials):
    _inhale_durations = inhale_durations.loc[:, trials]
    _inhale_groupby_trial = _inhale_durations.T.groupby("Trial")
    pre_fv_inhalation_durations = _inhale_groupby_trial.apply(_get_pre_fv_inhales)
    post_fv_inhalation_durations = _inhale_groupby_trial.apply(_get_post_fv_inhales)
    pre_fv_inhalation_durations.columns = ["pre_sniff_dur_3","pre_sniff_dur_2","pre_sniff_dur_1",]
    post_fv_inhalation_durations.columns = ["post_sniff_dur_1", "post_sniff_dur_2", "post_sniff_dur_3",]

    return pre_fv_inhalation_durations, post_fv_inhalation_durations

def _get_pre_fv_inhales(trial_df: pd.DataFrame):
    trial_name = trial_df.index.get_level_values(0).unique()[0]
    trial_df.index = trial_df.index.droplevel(0)
    durations = trial_df.loc["duration"]
    timestamps = trial_df.loc["timestamps"]
    pre_fv_durations = durations[timestamps < 0].iloc[-3:].reset_index(drop=True)
    pre_fv_series = pd.Series(pre_fv_durations, name=trial_name, index=[2, 1, 0])

    return pre_fv_series

def _get_post_fv_inhales(trial_df: pd.DataFrame):
    trial_name = trial_df.index.get_level_values(0).unique()[0]
    trial_df.index = trial_df.index.droplevel(0)
    durations = trial_df.loc["duration"]
    timestamps = trial_df.loc["timestamps"]
    post_fv_durations = durations[timestamps >= 0].iloc[:3].reset_index(drop=True)
    post_fv_series = pd.Series(post_fv_durations, name=trial_name, index=[0, 1, 2])

    return post_fv_series
