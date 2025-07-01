import logging
from functools import reduce
from warnings import warn
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from .process_files import PRE_ODOR_COUNT_TIME_MS, POST_ODOR_COUNT_TIME_MS

from .helpers import classifiers, async_io

pd.set_option("future.no_silent_downcasting", True)
logging.basicConfig(level=logging.NOTSET, filename="trial_nums.log", encoding="utf-8")


def animals_to_skip(concentration_files: dict[str, dict]) -> tuple[list, list]:
    all_keys = [
        list(concentration_files[concentration].keys())
        for concentration in concentration_files
    ]
    animals = np.unique(np.hstack(all_keys))
    good_animals = reduce(np.intersect1d, all_keys)
    bad_animals = np.setdiff1d(animals, good_animals)

    return good_animals, bad_animals


def process_combined(concentration_files: dict[str, dict], output_dir):
    good_animals, bad_animals = animals_to_skip(concentration_files)
    warn(
        f"{bad_animals} are missing some concentrations and will be skipped!",
        stacklevel=2,
    )

    concentration_dfs = {}
    tpe = async_io.AsyncIO()

    all_f1_df = pd.DataFrame()
    all_f2_df = pd.DataFrame()
    all_f3_df = pd.DataFrame()
    all_f4_df = pd.DataFrame()
    all_f5_df = pd.DataFrame()

    for concentration in tqdm(
        concentration_files,
        desc="Processing concentrations: ",
        total=len(concentration_files),
        leave=True,
        position=1,
    ):
        all_go_trial_traces = pd.DataFrame()
        all_nogo_trial_traces = pd.DataFrame()

        animal_files = concentration_files[concentration]



        for animal in animal_files:
            animal_data_matrix = pd.read_excel(
                animal_files[animal]["combined"], index_col=[0]
            )
            windowed_bin_counts = pd.read_excel(
                animal_files[animal]["window"], index_col=[0]
            )
            all_trimmed_traces = pd.read_excel(
                animal_files[animal]["traces"], index_col=[0]
            )

            if animal_files[animal]["1"] is not None:
                file1_df = pd.read_excel(
                    animal_files[animal]["1"], index_col=[0]
                )
                all_f1_df = pd.concat((all_f1_df, file1_df.T), axis=1)
            if animal_files[animal]["2"] is not None:
                file2_df = pd.read_excel(
                    animal_files[animal]["2"], index_col=[0]
                )
                all_f2_df = pd.concat((all_f2_df, file2_df.T), axis=1)
            if animal_files[animal]["3"] is not None:
                file3_df = pd.read_excel(
                    animal_files[animal]["3"], index_col=[0]
                )
                all_f3_df = pd.concat((all_f3_df, file3_df.T), axis=1)
            if animal_files[animal]["4"] is not None:
                file4_df = pd.read_excel(
                    animal_files[animal]["4"], index_col=[0]
                )
                all_f4_df = pd.concat((all_f4_df, file4_df.T), axis=1)
            if animal_files[animal]["5"] is not None:
                file5_df = pd.read_excel(
                    animal_files[animal]["5"], index_col=[0]
                )
                all_f5_df = pd.concat((all_f5_df, file5_df.T), axis=1)

            good_trials = windowed_bin_counts.columns

            trial_types = animal_data_matrix["trial_type"]
            go_trials = animal_data_matrix.loc[trial_types == 1].index
            nogo_trials = animal_data_matrix.loc[trial_types == 2].index

            go_trials = np.intersect1d(good_trials, go_trials)
            nogo_trials = np.intersect1d(good_trials, nogo_trials)

            # go_trial_counts = windowed_bin_counts.loc[PRE_ODOR_COUNT_TIME_MS:POST_ODOR_COUNT_TIME_MS, go_trials]
            # nogo_trial_counts = windowed_bin_counts.loc[PRE_ODOR_COUNT_TIME_MS:POST_ODOR_COUNT_TIME_MS, nogo_trials]

            go_trial_traces = all_trimmed_traces.loc[
                PRE_ODOR_COUNT_TIME_MS:POST_ODOR_COUNT_TIME_MS, go_trials
            ]
            nogo_trial_traces = all_trimmed_traces.loc[
                PRE_ODOR_COUNT_TIME_MS:POST_ODOR_COUNT_TIME_MS, nogo_trials
            ]

            go_trial_traces = go_trial_traces.replace("X", np.nan).infer_objects(
                copy=False
            )
            nogo_trial_traces = nogo_trial_traces.replace("X", np.nan).infer_objects(
                copy=False
            )
            go_trial_traces = go_trial_traces.dropna(axis=1).infer_objects(copy=False)
            nogo_trial_traces = nogo_trial_traces.dropna(axis=1).infer_objects(
                copy=False
            )

            zscore_go_trial_traces = classifiers.zscore_data(go_trial_traces)
            zscore_nogo_trial_traces = classifiers.zscore_data(nogo_trial_traces)

            all_go_trial_traces = pd.concat(
                (all_go_trial_traces, zscore_go_trial_traces), axis=1
            )
            all_nogo_trial_traces = pd.concat(
                (all_nogo_trial_traces, zscore_nogo_trial_traces), axis=1
            )

        # Do combined analysis here

        all_go_trial_traces.columns = np.repeat("go", all_go_trial_traces.shape[1])
        all_nogo_trial_traces.columns = np.repeat(
            "nogo", all_nogo_trial_traces.shape[1]
        )

        all_concentration_trials = pd.concat(
            [all_go_trial_traces, all_nogo_trial_traces], axis=1
        )
        concentration_dfs[concentration] = all_concentration_trials.T
        logging.info(
            "\nConcentration %s has %i go trials and %i nogo trials",
            concentration,
            all_go_trial_traces.shape[1],
            all_nogo_trial_traces.shape[1],
        )

    # all_concentration_labels = list(concentration_dfs.keys())

    # all_scores = pd.DataFrame(
    #     index=all_concentration_labels, columns=["score", "shuffle_score"]
    # )
    # all_individual_scores = pd.DataFrame(
    #     index=all_concentration_labels, columns=[np.arange(0, 20)]
    # )
    # for concentration in all_concentration_labels:
    #     concentration_df = concentration_dfs[concentration]
    #
    #     scores, individual_scores, individual_CMS = (
    #         classifiers.decode_trial_type_single(concentration_df, concentration)
    #     )
    #
    #     shuffled_concentration_df = classifiers.shuffle_labels(concentration_df)
    #
    #     shuffled_scores, shuffled_individual_scores, shuffled_individual_CMS = (
    #         classifiers.decode_trial_type_single(
    #             shuffled_concentration_df, concentration
    #         )
    #     )
    #
    #     all_scores.loc[concentration] = scores.to_numpy().round(3)
    #     all_individual_scores.loc[concentration] = individual_scores
    #     all_scores.loc[concentration, "shuffle_score"] = (
    #         shuffled_scores.to_numpy().round(3)
    #     )

    all_scores_path = output_dir.joinpath("all_scores.xlsx")
    all_individual_scores_path = output_dir.joinpath("all_individual_scores.xlsx")

    f1_path = output_dir.joinpath("combined_count.xlsx")
    f2_path = output_dir.joinpath("combined_duration.xlsx")
    f3_path = output_dir.joinpath("combined_ISI.xlsx")
    f4_path = output_dir.joinpath("combined_lengths.xlsx")
    f5_path = output_dir.joinpath("combined_bins.xlsx")

    # tpe.queue_save_df(all_scores, all_scores_path)
    # tpe.queue_save_df(all_individual_scores, all_individual_scores_path)

    tpe.queue_save_df(all_f1_df.T, f1_path)
    tpe.queue_save_df(all_f2_df.T, f2_path)
    tpe.queue_save_df(all_f3_df.T, f3_path)
    tpe.queue_save_df(all_f4_df.T, f4_path)
    tpe.queue_save_df(all_f5_df.T, f5_path)

    tpe.shutdown(wait=True)
