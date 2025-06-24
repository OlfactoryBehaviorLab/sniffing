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

SUMMARY_COLUMN = [
    'TOTAL_PRE_SNIFFS',
    'TOTAL_POST_SNIFFS',
    'TOTAL_PRE_GO_SNIFFS',
    'TOTAL_PRE_NOGO_SNIFFS',
    'TOTAL_POST_GO_SNIFFS',
    'TOTAL_POST_NOGO_SNIFFS',

    'AVG_LATENCY_1',
    'AVG_LATENCY_2',
    'AVG_LATENCY_3',
    'AVG_GO_LATENCY_1',
    'AVG_GO_LATENCY_2',
    'AVG_GO_LATENCY_3',
    'AVG_NOGO_LATENCY_1',
    'AVG_NOGO_LATENCY_2',
    'AVG_NOGO_LATENCY_3',

    'AVG_PRE_GO_SNIFF_DUR_1',
    'AVG_PRE_GO_SNIFF_DUR_2',
    'AVG_PRE_GO_SNIFF_DUR_3',
    'AVG_PRE_NOGO_SNIFF_DUR_1',
    'AVG_PRE_NOGO_SNIFF_DUR_2',
    'AVG_PRE_NOGO_SNIFF_DUR_3',

    'AVG_POST_GO_SNIFF_DUR_1',
    'AVG_POST_GO_SNIFF_DUR_2',
    'AVG_POST_GO_SNIFF_DUR_3',
    'AVG_POST_NOGO_SNIFF_DUR_1',
    'AVG_POST_NOGO_SNIFF_DUR_2',
    'AVG_POST_NOGO_SNIFF_DUR_3',
]

def repack_data(
    h5_file: DewanH5,
    inhale_counts,
    inhale_latencies,
    inhale_durations,
    output_dir,
    PRE_ODOR_COUNT_TIME_MS,  # noqa: N803
    POST_ODOR_COUNT_TIME_MS,  # noqa: N803
):
    animal_ID = h5_file.mouse
    odor = h5_file.odors
    odor = odor[odor != "blank"]
    concentration = h5_file.concentration
    trials = inhale_counts.columns.to_numpy()

    trial_type = h5_file.trial_parameters.loc[trials, "trial_type"]
    trial_results = h5_file.trial_parameters.loc[trials, "result"]

    combined_df = pd.DataFrame(index=trials, columns=COLUMNS)

    combined_df.loc[:, "ID"] = animal_ID
    combined_df.loc[:, "odor"] = odor
    combined_df.loc[:, "conc"] = concentration
    combined_df.loc[:, "type"] = trial_type
    combined_df.loc[:, "result"] = trial_results

    inhale_counts.index = ["pre_odor_sniffs", "post_odor_sniffs"]

    inhale_latencies.index = ["sniff_1_latency", "sniff_2_latency", "sniff_3_latency"]

    combined_df.loc[:, ["pre_odor_sniffs", "post_odor_sniffs"]] = inhale_counts.T
    combined_df.loc[:, ["sniff_1_latency", "sniff_2_latency", "sniff_3_latency"]] = (
        inhale_latencies.T
    )
    pre_fv_inhalation_durations, post_fv_inhalation_durations = unpack_inhale_durations(
        inhale_durations, trials, PRE_ODOR_COUNT_TIME_MS, POST_ODOR_COUNT_TIME_MS
    )
    combined_df.loc[:, ["pre_sniff_dur_3", "pre_sniff_dur_2", "pre_sniff_dur_1"]] = (
        pre_fv_inhalation_durations
    )

    combined_df.loc[:, ["post_sniff_dur_1", "post_sniff_dur_2", "post_sniff_dur_3"]] = (
        post_fv_inhalation_durations
    )



    filename = f"{animal_ID}-{concentration}-combined.xlsx"
    filename_spss = f"{animal_ID}-{concentration}-spss.xlsx"
    output_path = output_dir.joinpath(filename)
    output_path_spss = output_dir.joinpath(filename_spss)

    combined_df = combined_df.infer_objects().fillna("X")

    combined_df.to_excel(output_path)


def unpack_inhale_durations(
    inhale_durations,
    trials,
    PRE_ODOR_COUNT_TIME_MS,  # noqa: N803
    POST_ODOR_COUNT_TIME_MS,  # noqa: N803
):
    _inhale_durations = inhale_durations.loc[:, trials]
    _inhale_groupby_trial = _inhale_durations.T.groupby("Trial")
    pre_fv_inhalation_durations = _inhale_groupby_trial.apply(
        lambda x: _get_pre_fv_inhales(x, PRE_ODOR_COUNT_TIME_MS)
    )
    post_fv_inhalation_durations = _inhale_groupby_trial.apply(
        lambda x: _get_post_fv_inhales(x, POST_ODOR_COUNT_TIME_MS)
    )
    pre_fv_inhalation_durations.columns = [
        "pre_sniff_dur_3",
        "pre_sniff_dur_2",
        "pre_sniff_dur_1",
    ]
    post_fv_inhalation_durations.columns = [
        "post_sniff_dur_1",
        "post_sniff_dur_2",
        "post_sniff_dur_3",
    ]

    return pre_fv_inhalation_durations, post_fv_inhalation_durations


def _get_pre_fv_inhales(trial_df: pd.DataFrame, PRE_ODOR_COUNT_TIME_MS):  # noqa: N803
    trial_name = trial_df.index.get_level_values(0).unique()[0]
    pre_fv_durations = pd.Series(
        np.zeros(3), name=trial_name, index=[2, 1, 0], dtype=object
    )
    trial_df.index = trial_df.index.droplevel(0)
    durations = trial_df.loc["duration"]
    timestamps = trial_df.loc["timestamps"]

    timestamps_in_window = (timestamps >= PRE_ODOR_COUNT_TIME_MS) & (
        timestamps < 0
    )  # What timestamps are in our window
    durations_in_window = durations[timestamps_in_window].reset_index(
        drop=True
    )  # Grab those durations and reset index
    durations_in_window.index = durations_in_window.index.to_numpy()[
        ::-1
    ]  # Flip the index so the last duration becomes the first
    shared_index = np.intersect1d(
        pre_fv_durations.index, durations_in_window.index
    )  # To avoid out-of-bounds indexing, find the durations that share an index with our series
    pre_fv_durations.loc[shared_index] = durations_in_window.loc[shared_index].astype(
        object
    )

    return pre_fv_durations


def _get_post_fv_inhales(trial_df: pd.DataFrame, POST_ODOR_COUNT_TIME_MS):  # noqa: N803
    trial_name = trial_df.index.get_level_values(0).unique()[0]
    post_fv_durations = pd.Series(
        np.zeros(3), name=trial_name, index=[0, 1, 2], dtype=object
    )
    trial_df.index = trial_df.index.droplevel(0)
    durations = trial_df.loc["duration"]
    timestamps = trial_df.loc["timestamps"]

    timestamps_in_window = (timestamps < POST_ODOR_COUNT_TIME_MS) & (timestamps >= 0)
    durations_in_window = durations[timestamps_in_window].reset_index(drop=True)
    shared_index = np.intersect1d(post_fv_durations.index, durations_in_window.index)
    post_fv_durations.loc[shared_index] = durations_in_window.loc[shared_index].astype(
        object
    )

    return post_fv_durations


def calculate_summary_stats(combined_df: pd.DataFrame) -> pd.DataFrame:
    go_trials_mask = combined_df['type'] == 1
    nogo_trials_mask = combined_df['type'] == 0

    summary_stats = pd.DataFrame(index=SUMMARY_COLUMN, columns=['Stat'])

    summary_stats.loc['TOTAL_PRE_SNIFFS'] = combined_df['pre_odor_sniffs'].sum()
    summary_stats.loc['TOTAL_POST_SNIFFS'] = combined_df['post_odor_sniffs'].sum()
    summary_stats.loc['TOTAL_PRE_GO_SNIFFS'] = combined_df.loc[go_trials_mask, 'pre_odor_sniffs'].sum()
    summary_stats.loc['TOTAL_PRE_NOGO_SNIFFS'] = combined_df.loc[nogo_trials_mask, 'pre_odor_sniffs'].sum()
    summary_stats.loc['TOTAL_POST_GO_SNIFFS'] = combined_df.loc[go_trials_mask, 'post_odor_sniffs'].sum()
    summary_stats.loc['TOTAL_POST_NOGO_SNIFFS'] = combined_df.loc[nogo_trials_mask, 'post_odor_sniffs'].sum()

    summary_stats.loc['AVG_LATENCY_1'] = combined_df['sniff_1_latency'].mean()
    summary_stats.loc['AVG_LATENCY_2'] = combined_df['sniff_2_latency'].mean()
    summary_stats.loc['AVG_LATENCY_3'] = combined_df['sniff_3_latency'].mean()
    summary_stats.loc['AVG_GO_LATENCY_1'] = combined_df.loc[go_trials_mask, 'sniff_1_latency'].mean()
    summary_stats.loc['AVG_GO_LATENCY_2'] = combined_df.loc[go_trials_mask, 'sniff_2_latency'].mean()
    summary_stats.loc['AVG_GO_LATENCY_3'] = combined_df.loc[go_trials_mask, 'sniff_3_latency'].mean()
    summary_stats.loc['AVG_NOGO_LATENCY_1'] = combined_df.loc[nogo_trials_mask, 'sniff_1_latency'].mean()
    summary_stats.loc['AVG_NOGO_LATENCY_2'] = combined_df.loc[nogo_trials_mask, 'sniff_2_latency'].mean()
    summary_stats.loc['AVG_NOGO_LATENCY_3'] = combined_df.loc[nogo_trials_mask, 'sniff_3_latency'].mean()

    summary_stats.loc['AVG_PRE_GO_SNIFF_DUR_1'] = combined_df.loc[go_trials_mask, "pre_sniff_dur_1"].mean()
    summary_stats.loc['AVG_PRE_GO_SNIFF_DUR_2'] = combined_df.loc[go_trials_mask, "pre_sniff_dur_2"].mean()
    summary_stats.loc['AVG_PRE_GO_SNIFF_DUR_3'] = combined_df.loc[go_trials_mask, "pre_sniff_dur_3"].mean()
    summary_stats.loc['AVG_PRE_NOGO_SNIFF_DUR_1'] = combined_df.loc[nogo_trials_mask, "pre_sniff_dur_1"].mean()
    summary_stats.loc['AVG_PRE_NOGO_SNIFF_DUR_2'] = combined_df.loc[nogo_trials_mask, "pre_sniff_dur_2"].mean()
    summary_stats.loc['AVG_PRE_NOGO_SNIFF_DUR_3'] = combined_df.loc[nogo_trials_mask, "pre_sniff_dur_3"].mean()

    summary_stats.loc['AVG_POST_GO_SNIFF_DUR_1'] = combined_df.loc[go_trials_mask, "post_sniff_dur_1"].mean()
    summary_stats.loc['AVG_POST_GO_SNIFF_DUR_2'] = combined_df.loc[go_trials_mask, "post_sniff_dur_1"].mean()
    summary_stats.loc['AVG_POST_GO_SNIFF_DUR_3'] = combined_df.loc[go_trials_mask, "post_sniff_dur_1"].mean()
    summary_stats.loc['AVG_POST_NOGO_SNIFF_DUR_1'] = combined_df.loc[nogo_trials_mask, "post_sniff_dur_1"].mean()
    summary_stats.loc['AVG_POST_NOGO_SNIFF_DUR_2'] = combined_df.loc[nogo_trials_mask, "post_sniff_dur_1"].mean()
    summary_stats.loc['AVG_POST_NOGO_SNIFF_DUR_3'] = combined_df.loc[nogo_trials_mask, "post_sniff_dur_1"].mean()

    summary_stats = summary_stats.reset_index(drop=False, inplace=False)
    summary_stats.columns = ['Summary', 'Stat']
    return summary_stats