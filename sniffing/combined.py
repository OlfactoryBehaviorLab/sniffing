from functools import reduce
from warnings import warn
import pandas as pd
import numpy as np

from tqdm.auto import tqdm


def animals_to_skip(concentration_files: dict['str', dict]) -> tuple[list, list]:
    all_keys = [list(concentration_files[concentration].keys()) for concentration in concentration_files]
    animals = np.unique(np.hstack(all_keys))
    good_animals = reduce(np.intersect1d, all_keys)
    bad_animals = np.setdiff1d(animals, good_animals)

    return good_animals, bad_animals


def process_combined(concentration_files: dict['str', dict]):

    good_animals, bad_animals = animals_to_skip(concentration_files)
    warn(f'{bad_animals} are missing some concentrations and will be skipped!', stacklevel=2)

    for concentration in tqdm(concentration_files, desc="Processing concentrations: ", total=len(concentration_files),
                              leave=True, position=1):
        print(concentration)
