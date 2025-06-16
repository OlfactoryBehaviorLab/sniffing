from typing import Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

MIN_TRIAL_NUMBER = 40
RANDOM_SEED = 1749483131


def _run_svm(
    iter_data: tuple[object, pd.DataFrame],
    test_train_split: float = 0.2,
    num_splits: int = 20,
) -> tuple[str, float, Iterable[float], Iterable[np.ndarray]]:
    individual_scores: list[float] = []
    individual_CMS: list[np.ndarray] = []

    window_name, sniff_count_window = iter_data
    window_name = int(window_name)

    sniff_count_window = pd.DataFrame(sniff_count_window)

    # sniff_count_window = zscore_data(sniff_count_window)

    x_train, x_test, y_train, y_test = train_test_split(
        sniff_count_window,
        sniff_count_window.index,
        test_size=test_train_split,
        random_state=RANDOM_SEED,
        stratify=sniff_count_window.index
    )

    svm = LinearSVC(dual="auto", random_state=RANDOM_SEED)
    bagging_classifier = BaggingClassifier(
        svm, n_estimators=num_splits, random_state=1000, n_jobs=-1
    )
    bagging_classifier = bagging_classifier.fit(x_train, y_train)
    bagged_score = bagging_classifier.score(x_test, y_test)

    for num, sub_estimator in enumerate(bagging_classifier.estimators_):
        logging.info("Getting scores from subestimator %i for window %s", num, window_name)
        sub_estimator_predictions = sub_estimator.predict(x_test)
        sub_estimator_score = sub_estimator.score(x_test, convert_results(y_test))
        _converted_predictions = convert_results(sub_estimator_predictions)
        sub_estimator_cm = confusion_matrix(y_test, _converted_predictions)

        individual_scores.append(sub_estimator_score)
        individual_CMS.append(sub_estimator_cm)

    return window_name, bagged_score, individual_scores, individual_CMS

def convert_results(predictions: np.ndarray):
    converted_results = []

    for prediction in predictions:
            if prediction == 0:
                converted_results.append('go')
            elif prediction == 1:
                converted_results.append('nogo')
            elif prediction == 'go':
                converted_results.append(0)
            elif prediction == 'nogo':
                converted_results.append(1)

    return np.array(converted_results)


def zscore_data(windowed_sniff_counts: pd.DataFrame) -> pd.DataFrame:
    transformed_data = StandardScaler().fit_transform(windowed_sniff_counts)
    return pd.DataFrame(transformed_data, index = windowed_sniff_counts.index, columns=windowed_sniff_counts.columns)


def decode_trial_type(
    windowed_sniff_counts: pd.DataFrame,
    test_train_split: float = 0.2,
    num_splits: int = 20,
):

    all_scores: pd.Series = pd.Series(index=windowed_sniff_counts.columns)
    all_individual_scores: dict[str, pd.DataFrame] = dict.fromkeys(windowed_sniff_counts.columns)
    all_individual_CMS: dict[str, list[np.ndarray]] = dict.fromkeys(windowed_sniff_counts.columns)

    partial_function = partial(_run_svm, test_train_split=test_train_split, num_splits=num_splits)

    with ProcessPoolExecutor() as ppe:
        for name, bagged_score, individual_scores, individual_CMS in ppe.map(
            partial_function, windowed_sniff_counts.items()
        ):
            logging.info("Received results for %s", name)
            all_scores.loc[name] = bagged_score
            all_individual_scores[name] = pd.DataFrame(
                individual_scores, index=np.arange(len(individual_scores))
            )
            all_individual_CMS[name] = individual_CMS

    return all_scores, all_individual_scores, all_individual_CMS


def decode_trial_type_single(
    windowed_sniff_counts: pd.DataFrame,
    test_train_split: float = 0.2,
    num_splits: int = 20
) -> tuple[pd.Series, dict[str, pd.DataFrame], dict[str, list[np.ndarray]]]:

    all_scores: pd.Series = pd.Series(index=windowed_sniff_counts.columns)
    all_individual_scores: dict[str, pd.DataFrame] = dict.fromkeys(windowed_sniff_counts.columns)
    all_individual_CMS: dict[str, list[np.ndarray]] = dict.fromkeys(windowed_sniff_counts.columns)

    for data in windowed_sniff_counts.items():
        name, bagged_score, individual_scores, individual_CMS = _run_svm(data,
                                                                         test_train_split=test_train_split,
                                                                         num_splits=num_splits)
        all_scores.loc[name] = bagged_score
        all_individual_scores[name] = pd.DataFrame(
            individual_scores, index=np.arange(len(individual_scores))
        )
        all_individual_CMS[name] = individual_CMS

    return all_scores, all_individual_scores, all_individual_CMS
