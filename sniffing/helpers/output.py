import pandas as pd
import numpy as np
from dewan_h5 import DewanH5
from dewan_utils.async_io import AsyncIO

pd.options.mode.copy_on_write = True

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

SHEET_1_COLUMNS = [
    # Repeating Data
    "ID",
    "odor",
    "conc",
    # Trial Data
    "type",
    "result",
    "correct",
    # Counts
    "pre-post",
    "count",
]

SHEET_2_COLUMNS = [
    # Repeating Data
    "ID",
    "odor",
    "conc",
    # Trial Data
    "type",
    "result",
    "correct",
    # Durations
    "pre-post",
    "sniff_num",
    "duration",
]

SHEET_3_COLUMNS = [
    # Repeating Data
    "ID",
    "odor",
    "conc",
    # Trial Data
    "type",
    "result",
    "correct",
    # ISI
    "NUM",
    "ISI",
]

SHEET_4_COLUMNS = [
    # Repeating Data
    "ID",
    "odor",
    "conc",
    # Trial Data
    "type",
    "result",
    "correct",
    "LEN",
]

SHEET_5_COLUMNS = [
    # Repeating Data
    "ID",
    "odor",
    "conc",
    # Trial Data
    "type",
    "result",
    "correct",
    "bin",
    "count",
]

SHEET_6_COLUMNS = [
    # Repeating Data
    "ID",
    "odor",
    "conc",
    # Trial Data
    "type",
    "result",
    "correct",
    "sniff_num",
    "sniff_dur"
]


SUMMARY_COLUMN = [
    "TOTAL_PRE_SNIFFS",
    "TOTAL_POST_SNIFFS",
    "TOTAL_PRE_GO_SNIFFS",
    "TOTAL_PRE_NOGO_SNIFFS",
    "TOTAL_POST_GO_SNIFFS",
    "TOTAL_POST_NOGO_SNIFFS",
    "AVG_GO_LATENCY_1",
    "AVG_GO_LATENCY_2",
    "AVG_GO_LATENCY_3",
    "AVG_NOGO_LATENCY_1",
    "AVG_NOGO_LATENCY_2",
    "AVG_NOGO_LATENCY_3",
    "AVG_PRE_GO_SNIFF_DUR_1",
    "AVG_PRE_GO_SNIFF_DUR_2",
    "AVG_PRE_GO_SNIFF_DUR_3",
    "AVG_PRE_NOGO_SNIFF_DUR_1",
    "AVG_PRE_NOGO_SNIFF_DUR_2",
    "AVG_PRE_NOGO_SNIFF_DUR_3",
    "AVG_POST_GO_SNIFF_DUR_1",
    "AVG_POST_GO_SNIFF_DUR_2",
    "AVG_POST_GO_SNIFF_DUR_3",
    "AVG_POST_NOGO_SNIFF_DUR_1",
    "AVG_POST_NOGO_SNIFF_DUR_2",
    "AVG_POST_NOGO_SNIFF_DUR_3",
]

CORRECT_MAP = {
    1: 1,  # 1 is correct
    2: 1,  # 1 is correct
    3: 0,  # 0 is incorrect
    5: 0,  # 0 is incorrect
}


def repack_data(
    h5_file: DewanH5,
    inhale_counts,
    inhale_latencies,
    inhale_durations,
    trial_lengths: pd.Series,
    bin_counts: pd.DataFrame,
    output_dir,
    PRE_ODOR_COUNT_TIME_MS,  # noqa: N803
    POST_ODOR_COUNT_TIME_MS,  # noqa: N803
    tpe: AsyncIO,
):
    animal_ID = h5_file.mouse
    odor = h5_file.odors
    odor = odor[odor != "blank"]
    concentration = h5_file.concentration
    trials = inhale_counts.columns.to_numpy()
    trial_type = h5_file.trial_parameters.loc[trials, "trial_type"]
    trial_results = h5_file.trial_parameters.loc[trials, "result"]

    inhale_counts.index = ["pre_odor_sniffs", "post_odor_sniffs"]
    inhale_latencies.index = ["sniff_1_latency", "sniff_2_latency", "sniff_3_latency"]

    pre_fv_inhalation_durations, post_fv_inhalation_durations = unpack_three_durations(
        inhale_durations, trials, PRE_ODOR_COUNT_TIME_MS, POST_ODOR_COUNT_TIME_MS
    )

    all_sniff_durations = unpack_all_durations(inhale_durations, trials, PRE_ODOR_COUNT_TIME_MS, POST_ODOR_COUNT_TIME_MS)
    all_sniff_durations = all_sniff_durations.sort_index()

    correct_nogo_trials = trial_results.loc[trial_results==2].index
    correct_nogo_durations = unpack_all_durations(inhale_durations, correct_nogo_trials, PRE_ODOR_COUNT_TIME_MS, 2000)
    correct_nogo_durations = correct_nogo_durations.sort_index()

    correct_nogo_bins = bin_counts[correct_nogo_trials]
    correct_nogo_bins = correct_nogo_bins.sort_index()

    sheet_1 = output_sheet_1(
        animal_ID, odor, concentration, trial_type, trial_results, inhale_counts
    )
    sheet_2 = output_sheet_2(
        animal_ID, odor, concentration, trial_type, trial_results, pre_fv_inhalation_durations, post_fv_inhalation_durations,
    )
    sheet_3 = output_sheet_3(
        animal_ID, odor, concentration, trial_type, trial_results, inhale_latencies
    )
    sheet_4 = output_sheet_4(
        animal_ID, odor, concentration, trial_type, trial_results, trial_lengths
    )
    sheet_5 = output_sheet_5(
        animal_ID, odor, concentration, trial_type, trial_results, bin_counts
    )
    sheet_5_correct_nogo = output_sheet_5(
        animal_ID, odor, concentration, trial_type, trial_results, correct_nogo_bins
    )
    sheet_6 = output_sheet_6(
        animal_ID, odor, concentration, trial_type, trial_results, all_sniff_durations
    )
    sheet_6_correct_nogo = output_sheet_6(
        animal_ID, odor, concentration, trial_type, trial_results, correct_nogo_durations
    )

    sheet_1_path = output_dir.joinpath(
        f"1_{animal_ID}-{concentration}-sniff_count.xlsx"
    )
    sheet_2_path = output_dir.joinpath(
        f"2_{animal_ID}-{concentration}-sniff_duration.xlsx"
    )
    sheet_3_path = output_dir.joinpath(f"3_{animal_ID}-{concentration}-ISI.xlsx")
    sheet_4_path = output_dir.joinpath(f"4_{animal_ID}-{concentration}-lengths.xlsx")
    sheet_5_path = output_dir.joinpath(f"5_{animal_ID}-{concentration}-bins.xlsx")
    sheet_5_path_correct_nogo_path = output_dir.joinpath(f"5_{animal_ID}-{concentration}-bins_correct_nogo.xlsx")
    sheet_6_path = output_dir.joinpath(f"6_{animal_ID}-{concentration}-all_sniff_durs.xlsx")
    sheet_6_correct_nogo_path = output_dir.joinpath(f"6_{animal_ID}-{concentration}-all_sniff_durs_correct_nogo.xlsx")

    tpe.queue_save_df(sheet_1, sheet_1_path)
    tpe.queue_save_df(sheet_2, sheet_2_path)
    tpe.queue_save_df(sheet_3, sheet_3_path)
    tpe.queue_save_df(sheet_4, sheet_4_path)
    tpe.queue_save_df(sheet_5, sheet_5_path)
    tpe.queue_save_df(sheet_5_correct_nogo, sheet_5_path_correct_nogo_path)
    tpe.queue_save_df(sheet_6, sheet_6_path)
    tpe.queue_save_df(sheet_6_correct_nogo, sheet_6_correct_nogo_path)


def unpack_all_durations(
        inhale_durations,
        trials,
        PRE_ODOR_COUNT_TIME_MS,  # noqa: N803
        POST_ODOR_COUNT_TIME_MS,  # noqa: N803
):
    _inhale_durations = inhale_durations.loc[:, trials]
    _inhale_groupby_trial = _inhale_durations.T.groupby("Trial")

    all_durations = pd.DataFrame()

    for trial, data in _inhale_groupby_trial:
        data = data.loc[trial].T
        ts = data['timestamps'].to_numpy()
        dur = data['duration'].to_numpy()
        pre_FV_ts = (PRE_ODOR_COUNT_TIME_MS <= ts) & (ts < 0)
        post_FV_ts = (0 <= ts) & (ts < POST_ODOR_COUNT_TIME_MS)

        good_dur = dur != 'X'

        pre_fv_sniff_dur = dur[pre_FV_ts & good_dur]
        post_fv_sniff_dur = dur[post_FV_ts & good_dur]

        dur_index_pre = np.arange(-pre_fv_sniff_dur.shape[0], 0)
        dur_index_post = np.arange(1, post_fv_sniff_dur.shape[0]+1)
        durations = pd.Series(np.hstack((pre_fv_sniff_dur, post_fv_sniff_dur)), index=np.hstack((dur_index_pre, dur_index_post)), name=trial)
        all_durations = pd.concat((all_durations, durations), axis=1)

    return all_durations.T



def unpack_three_durations(
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


def output_sheet_1(
    animal_ID: int,  # NOQA N803
    odor: str,
    concentration: str,
    trial_type: pd.Series,
    trial_results: pd.Series,
    inhale_counts: pd.DataFrame,
):
    index = inhale_counts.columns.to_numpy()

    sheet_1_pre_df = pd.DataFrame(index=index, columns=SHEET_1_COLUMNS)
    sheet_1_pre_df.loc[:, "ID"] = animal_ID
    sheet_1_pre_df.loc[:, "odor"] = odor
    sheet_1_pre_df.loc[:, "conc"] = concentration[-1]
    sheet_1_pre_df.loc[:, "type"] = trial_type
    sheet_1_pre_df.loc[:, "result"] = trial_results
    sheet_1_pre_df.loc[:, "correct"] = trial_results.replace(CORRECT_MAP)
    pre_sniff_count = inhale_counts.loc["pre_odor_sniffs"].T
    count_type = np.repeat(-1, len(pre_sniff_count))
    sheet_1_pre_df.loc[:, "pre-post"] = count_type
    sheet_1_pre_df.loc[:, "count"] = pre_sniff_count

    sheet_1_post_df = sheet_1_pre_df.copy()
    post_sniff_count = inhale_counts.loc["post_odor_sniffs"].T
    count_type = np.repeat(1, len(post_sniff_count))
    sheet_1_post_df.loc[:, "pre-post"] = count_type
    sheet_1_post_df.loc[:, "count"] = post_sniff_count

    sheet_1_df = pd.concat([sheet_1_pre_df.T, sheet_1_post_df.T], axis=1).T
    return sheet_1_df.sort_index(inplace=False)


def output_sheet_2(
    animal_ID: int,  # NOQA N803
    odor: str,
    concentration: str,
    trial_type: pd.Series,
    trial_results: pd.Series,
    pre_fv_inhalation_durations: pd.DataFrame,
    post_fv_inhalation_durations: pd.DataFrame,
):
    sheet_2_df = pd.DataFrame(index=SHEET_2_COLUMNS)

    for sniff_num, data in pre_fv_inhalation_durations.items():
        sniff_num_int = int(sniff_num[-1])

        sheet_2_pre_df = pd.DataFrame(index=data.index, columns=SHEET_2_COLUMNS)
        sheet_2_pre_df.loc[:, "ID"] = animal_ID
        sheet_2_pre_df.loc[:, "odor"] = odor
        sheet_2_pre_df.loc[:, "conc"] = concentration[-1]
        sheet_2_pre_df.loc[:, "type"] = trial_type
        sheet_2_pre_df.loc[:, "result"] = trial_results
        sheet_2_pre_df.loc[:, "correct"] = trial_results.replace(CORRECT_MAP)
        sheet_2_pre_df.loc[:, "pre-post"] = -1
        sheet_2_pre_df.loc[:, "sniff_num"] = sniff_num_int
        data.loc[data == 0] = np.nan
        sheet_2_pre_df.loc[:, "duration"] = data
        sheet_2_df = pd.concat([sheet_2_df, sheet_2_pre_df.T], axis=1)

    for sniff_num, data in post_fv_inhalation_durations.items():
        sniff_num_int = int(sniff_num[-1])

        sheet_2_post_df = pd.DataFrame(index=data.index, columns=SHEET_2_COLUMNS)
        sheet_2_post_df.loc[:, "ID"] = animal_ID
        sheet_2_post_df.loc[:, "odor"] = odor
        sheet_2_post_df.loc[:, "conc"] = concentration[-1]
        sheet_2_post_df.loc[:, "type"] = trial_type
        sheet_2_post_df.loc[:, "result"] = trial_results
        sheet_2_post_df.loc[:, "correct"] = trial_results.replace(CORRECT_MAP)
        sheet_2_post_df.loc[:, "pre-post"] = 1
        sheet_2_post_df.loc[:, "sniff_num"] = sniff_num_int
        data.loc[data == 0] = np.nan
        sheet_2_post_df.loc[:, "duration"] = data
        sheet_2_df = pd.concat([sheet_2_df, sheet_2_post_df.T], axis=1)

    return sheet_2_df.T.sort_index(inplace=False)


def output_sheet_3(
    animal_ID: int,  # NOQA N803
    odor: str,
    concentration: str,
    trial_type: pd.Series,
    trial_results: pd.Series,
    inhale_latencies: pd.DataFrame,
):
    ISI_1 = inhale_latencies.loc["sniff_1_latency", :]
    ISI_2 = (
        inhale_latencies.loc["sniff_2_latency", :]
        - inhale_latencies.loc["sniff_1_latency", :]
    )
    ISI_3 = (
        inhale_latencies.loc["sniff_3_latency", :]
        - inhale_latencies.loc["sniff_2_latency", :]
    )
    ISI_1.loc[ISI_1 <= 0] = np.nan
    ISI_2.loc[ISI_2 <= 0] = np.nan
    ISI_3.loc[ISI_3 <= 0] = np.nan
    # NUM, ISI
    ISI_1_df = pd.DataFrame(index=inhale_latencies.columns, columns=SHEET_3_COLUMNS)
    ISI_1_df.loc[:, "ID"] = animal_ID
    ISI_1_df.loc[:, "odor"] = odor
    ISI_1_df.loc[:, "conc"] = concentration[-1]
    ISI_1_df.loc[:, "type"] = trial_type
    ISI_1_df.loc[:, "result"] = trial_results
    ISI_1_df.loc[:, "correct"] = trial_results.replace(CORRECT_MAP)
    ISI_1_df.loc[:, "NUM"] = 1
    ISI_1_df.loc[:, "ISI"] = ISI_1.to_numpy()

    ISI_2_df = pd.DataFrame(index=inhale_latencies.columns, columns=SHEET_3_COLUMNS)
    ISI_2_df.loc[:, "ID"] = animal_ID
    ISI_2_df.loc[:, "odor"] = odor
    ISI_2_df.loc[:, "conc"] = concentration[-1]
    ISI_2_df.loc[:, "type"] = trial_type
    ISI_2_df.loc[:, "result"] = trial_results
    ISI_2_df.loc[:, "correct"] = trial_results.replace(CORRECT_MAP)
    ISI_2_df.loc[:, "NUM"] = 2
    ISI_2_df.loc[:, "ISI"] = ISI_2.to_numpy()

    ISI_3_df = pd.DataFrame(index=inhale_latencies.columns, columns=SHEET_3_COLUMNS)
    ISI_3_df.loc[:, "ID"] = animal_ID
    ISI_3_df.loc[:, "odor"] = odor
    ISI_3_df.loc[:, "conc"] = concentration[-1]
    ISI_3_df.loc[:, "type"] = trial_type
    ISI_3_df.loc[:, "result"] = trial_results
    ISI_3_df.loc[:, "correct"] = trial_results.replace(CORRECT_MAP)
    ISI_3_df.loc[:, "NUM"] = 3
    ISI_3_df.loc[:, "ISI"] = ISI_3.to_numpy()

    sheet_3_df = pd.concat((ISI_1_df.T, ISI_2_df.T, ISI_3_df.T), axis=1).T
    return sheet_3_df.sort_index(inplace=False)


def output_sheet_4(
    animal_ID: int,  # NOQA N803
    odor: str,
    concentration: str,
    trial_type: pd.Series,
    trial_results: pd.Series,
    trial_lengths: pd.Series,
):
    sheet_4_df = pd.DataFrame(index=trial_lengths.index, columns=SHEET_4_COLUMNS)
    sheet_4_df.loc[:, "ID"] = animal_ID
    sheet_4_df.loc[:, "odor"] = odor
    sheet_4_df.loc[:, "conc"] = concentration[-1]
    sheet_4_df.loc[:, "type"] = trial_type
    sheet_4_df.loc[:, "result"] = trial_results
    sheet_4_df.loc[:, "correct"] = trial_results.replace(CORRECT_MAP)
    sheet_4_df.loc[:, "LEN"] = trial_lengths

    return sheet_4_df


def output_sheet_5(
    animal_ID: int,  # NOQA N803
    odor: str,
    concentration: str,
    trial_type: pd.Series,
    trial_results: pd.Series,
    binned_counts: pd.DataFrame,
):
    sheet_5_df = pd.DataFrame(index=SHEET_5_COLUMNS)
    for _bin, data in binned_counts.T.items():
        bin_df = pd.DataFrame(index=data.index, columns=SHEET_5_COLUMNS)
        bin_df.loc[:, "ID"] = animal_ID
        bin_df.loc[:, "odor"] = odor
        bin_df.loc[:, "conc"] = concentration[-1]
        bin_df.loc[:, "type"] = trial_type
        bin_df.loc[:, "result"] = trial_results
        bin_df.loc[:, "correct"] = trial_results.replace(CORRECT_MAP)
        bin_df.loc[:, "bin"] = _bin
        bin_df.loc[:, "count"] = data
        sheet_5_df = pd.concat((sheet_5_df, bin_df.T), axis=1)

    return sheet_5_df.T.sort_index(inplace=False)


def output_sheet_6(
        animal_ID: int,  # NOQA N803
        odor: str,
        concentration: str,
        trial_type: pd.Series,
        trial_results: pd.Series,
        sniff_durations: pd.DataFrame
):

    sheet_6_df = pd.DataFrame(index=SHEET_6_COLUMNS)

    for trial, data in sniff_durations.iterrows():
        trial_df = pd.DataFrame(index=np.repeat(trial, data.shape[0]), columns=SHEET_6_COLUMNS)
        trial_df.loc[:, "ID"] = animal_ID
        trial_df.loc[:, "odor"] = odor
        trial_df.loc[:, "conc"] = concentration[-1]
        trial_df.loc[:, "type"] = trial_type
        trial_df.loc[:, "result"] = trial_results
        trial_df.loc[:, "correct"] = trial_results.replace(CORRECT_MAP)
        trial_df.loc[:, "sniff_num"] = data.index
        trial_df.loc[:, "sniff_dur"] = data.to_numpy()

        sheet_6_df = pd.concat((sheet_6_df, trial_df.T), axis=1)

    return sheet_6_df.T.sort_index(inplace=False)


def calculate_summary_stats(combined_df: pd.DataFrame) -> pd.DataFrame:
    go_trials_mask = combined_df["type"] == 1
    nogo_trials_mask = combined_df["type"] == 2

    summary_stats = pd.DataFrame(index=SUMMARY_COLUMN, columns=["Stat"])

    summary_stats.loc["TOTAL_PRE_SNIFFS"] = combined_df.loc[:, "pre_odor_sniffs"].sum()
    summary_stats.loc["TOTAL_POST_SNIFFS"] = combined_df.loc[
        :, "post_odor_sniffs"
    ].sum()
    summary_stats.loc["TOTAL_PRE_GO_SNIFFS"] = combined_df.loc[
        go_trials_mask, "pre_odor_sniffs"
    ].sum()
    summary_stats.loc["TOTAL_PRE_NOGO_SNIFFS"] = combined_df.loc[
        nogo_trials_mask, "pre_odor_sniffs"
    ].sum()
    summary_stats.loc["TOTAL_POST_GO_SNIFFS"] = combined_df.loc[
        go_trials_mask, "post_odor_sniffs"
    ].sum()
    summary_stats.loc["TOTAL_POST_NOGO_SNIFFS"] = combined_df.loc[
        nogo_trials_mask, "post_odor_sniffs"
    ].sum()

    _sniff_latency_1 = combined_df.loc[:, "sniff_1_latency"].replace({"X": -1})
    summary_stats.loc["AVG_LATENCY_1"] = _sniff_latency_1[_sniff_latency_1 > 0].mean()
    _sniff_latency_2 = combined_df.loc[:, "sniff_2_latency"].replace({"X": -1})
    summary_stats.loc["AVG_LATENCY_2"] = _sniff_latency_2[_sniff_latency_2 > 0].mean()
    _sniff_latency_3 = combined_df.loc[:, "sniff_3_latency"].replace({"X": -1})
    summary_stats.loc["AVG_LATENCY_3"] = _sniff_latency_3[_sniff_latency_3 > 0].mean()

    _sniff_1_latency = combined_df.loc[go_trials_mask, "sniff_1_latency"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_GO_LATENCY_1"] = _sniff_1_latency[
        _sniff_1_latency > 0
    ].mean()
    _sniff_2_latency = combined_df.loc[go_trials_mask, "sniff_2_latency"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_GO_LATENCY_2"] = _sniff_2_latency[
        _sniff_2_latency > 0
    ].mean()
    _sniff_3_latency = combined_df.loc[go_trials_mask, "sniff_3_latency"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_GO_LATENCY_3"] = _sniff_3_latency[
        _sniff_3_latency > 0
    ].mean()

    _sniff_1_latency = combined_df.loc[nogo_trials_mask, "sniff_1_latency"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_NOGO_LATENCY_1"] = _sniff_1_latency[
        _sniff_1_latency > 0
    ].mean()
    _sniff_2_latency = combined_df.loc[nogo_trials_mask, "sniff_2_latency"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_NOGO_LATENCY_2"] = _sniff_2_latency[
        _sniff_2_latency > 0
    ].mean()
    _sniff_3_latency = combined_df.loc[nogo_trials_mask, "sniff_3_latency"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_NOGO_LATENCY_3"] = _sniff_3_latency[
        _sniff_3_latency > 0
    ].mean()

    _sniff_dur_1 = combined_df.loc[go_trials_mask, "pre_sniff_dur_1"].replace({"X": -1})
    summary_stats.loc["AVG_PRE_GO_SNIFF_DUR_1"] = _sniff_dur_1[_sniff_dur_1 > 0].mean()
    _sniff_dur_2 = combined_df.loc[go_trials_mask, "pre_sniff_dur_2"].replace({"X": -1})
    summary_stats.loc["AVG_PRE_GO_SNIFF_DUR_2"] = _sniff_dur_2[_sniff_dur_2 > 0].mean()
    _sniff_dur_3 = combined_df.loc[go_trials_mask, "pre_sniff_dur_3"].replace({"X": -1})
    summary_stats.loc["AVG_PRE_GO_SNIFF_DUR_3"] = _sniff_dur_3[_sniff_dur_3 > 0].mean()
    _sniff_dur_1 = combined_df.loc[nogo_trials_mask, "pre_sniff_dur_1"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_PRE_NOGO_SNIFF_DUR_1"] = _sniff_dur_1[
        _sniff_dur_1 > 0
    ].mean()
    _sniff_dur_2 = combined_df.loc[nogo_trials_mask, "pre_sniff_dur_2"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_PRE_NOGO_SNIFF_DUR_2"] = _sniff_dur_2[
        _sniff_dur_2 > 0
    ].mean()
    _sniff_dur_3 = combined_df.loc[nogo_trials_mask, "pre_sniff_dur_3"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_PRE_NOGO_SNIFF_DUR_3"] = _sniff_dur_3[
        _sniff_dur_3 > 0
    ].mean()

    _sniff_dur_1 = combined_df.loc[go_trials_mask, "post_sniff_dur_1"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_POST_GO_SNIFF_DUR_1"] = _sniff_dur_1[_sniff_dur_1 > 0].mean()
    _sniff_dur_2 = combined_df.loc[go_trials_mask, "post_sniff_dur_2"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_POST_GO_SNIFF_DUR_2"] = _sniff_dur_2[_sniff_dur_2 > 0].mean()
    _sniff_dur_3 = combined_df.loc[go_trials_mask, "post_sniff_dur_3"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_POST_GO_SNIFF_DUR_3"] = _sniff_dur_3[_sniff_dur_3 > 0].mean()
    _sniff_dur_1 = combined_df.loc[nogo_trials_mask, "post_sniff_dur_1"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_POST_NOGO_SNIFF_DUR_1"] = _sniff_dur_1[
        _sniff_dur_1 > 0
    ].mean()
    _sniff_dur_2 = combined_df.loc[nogo_trials_mask, "post_sniff_dur_2"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_POST_NOGO_SNIFF_DUR_2"] = _sniff_dur_2[
        _sniff_dur_2 > 0
    ].mean()
    _sniff_dur_3 = combined_df.loc[nogo_trials_mask, "post_sniff_dur_3"].replace(
        {"X": -1}
    )
    summary_stats.loc["AVG_POST_NOGO_SNIFF_DUR_3"] = _sniff_dur_3[
        _sniff_dur_3 > 0
    ].mean()

    summary_stats = summary_stats.reset_index(drop=False, inplace=False)
    summary_stats.columns = ["Summary", "Stat"]
    return summary_stats


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
