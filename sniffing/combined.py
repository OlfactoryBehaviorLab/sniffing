import logging
from functools import reduce
from warnings import warn
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

from .process_files import PRE_ODOR_COUNT_TIME_MS, POST_ODOR_COUNT_TIME_MS

from .helpers import classifiers

logging.basicConfig(level=logging.NOTSET, filename='trial_nums.log', encoding='utf-8')

def animals_to_skip(concentration_files: dict[str, dict]) -> tuple[list, list]:
    all_keys = [list(concentration_files[concentration].keys()) for concentration in concentration_files]
    animals = np.unique(np.hstack(all_keys))
    good_animals = reduce(np.intersect1d, all_keys)
    bad_animals = np.setdiff1d(animals, good_animals)

    return good_animals, bad_animals


def process_combined(concentration_files: dict[str, dict], output_dir):

    good_animals, bad_animals = animals_to_skip(concentration_files)
    warn(f'{bad_animals} are missing some concentrations and will be skipped!', stacklevel=2)

    concentration_dfs = {}

    for concentration in tqdm(concentration_files, desc="Processing concentrations: ", total=len(concentration_files),
                              leave=True, position=1):
        all_go_trial_traces = pd.DataFrame()
        all_nogo_trial_traces = pd.DataFrame()

        animal_files = concentration_files[concentration]


        for animal in animal_files:
            animal_data_matrix = pd.read_excel(animal_files[animal]['combined'], index_col=[0])
            windowed_bin_counts = pd.read_excel(animal_files[animal]['window'], index_col=[0])
            all_trimmed_traces = pd.read_excel(animal_files[animal]['traces'], index_col=[0])

            good_trials = windowed_bin_counts.columns

            trial_types = animal_data_matrix['trial_type']
            go_trials = animal_data_matrix.loc[trial_types == 1].index
            nogo_trials = animal_data_matrix.loc[trial_types == 2].index

            go_trials = np.intersect1d(good_trials, go_trials)
            nogo_trials = np.intersect1d(good_trials, nogo_trials)

            #go_trial_counts = windowed_bin_counts.loc[PRE_ODOR_COUNT_TIME_MS:POST_ODOR_COUNT_TIME_MS, go_trials]
            #nogo_trial_counts = windowed_bin_counts.loc[PRE_ODOR_COUNT_TIME_MS:POST_ODOR_COUNT_TIME_MS, nogo_trials]

            go_trial_traces = all_trimmed_traces.loc[PRE_ODOR_COUNT_TIME_MS:POST_ODOR_COUNT_TIME_MS, go_trials]
            nogo_trial_traces = all_trimmed_traces.loc[PRE_ODOR_COUNT_TIME_MS:POST_ODOR_COUNT_TIME_MS, nogo_trials]

            all_go_trial_traces = pd.concat((all_go_trial_traces, go_trial_traces), axis=1)
            all_nogo_trial_traces = pd.concat((all_nogo_trial_traces, nogo_trial_traces), axis=1)

        # Do combined analysis here

        all_go_trial_traces.columns = np.repeat('go', all_go_trial_traces.shape[1])
        all_nogo_trial_traces.columns = np.repeat('nogo', all_nogo_trial_traces.shape[1])

        all_concentration_trials = pd.concat([all_go_trial_traces, all_nogo_trial_traces], axis=1)
        concentration_dfs[concentration] = all_concentration_trials.T
        logging.info("\nConcentration %s has %i go trials and %i nogo trials", concentration, all_go_trial_traces.shape[1], all_nogo_trial_traces.shape[1])


    all_concentration_labels = list(concentration_dfs.keys())
    # scores, individual_scores, individual_CMS = classifiers.decode_trial_type(concentration_dfs[all_concentration_labels[0]])
    _cols = concentration_dfs[all_concentration_labels[0]].columns
    all_scores = pd.DataFrame(index=all_concentration_labels, columns=_cols)
    all_variance = pd.DataFrame(index=all_concentration_labels, columns=_cols)
    all_means = pd.DataFrame(index=all_concentration_labels, columns=_cols)

    for concentration in all_concentration_labels:
        concentration_df = concentration_dfs[concentration]
        all_variance.loc[concentration] = concentration_df.var()
        all_means.loc[concentration] = concentration_df.mean()

        scores, individual_scores, individual_CMS = classifiers.decode_trial_type_single(concentration_df)
        all_scores.loc[concentration] = scores

    all_scores_path = output_dir.joinpath('all_scores.xlsx')
    all_variance_path = output_dir.joinpath('all_variance.xlsx')
    all_means_path = output_dir.joinpath('all_means.xlsx')

    all_scores.to_excel(all_scores_path)
    all_variance.to_excel(all_variance_path)
    all_means.to_excel(all_means_path)