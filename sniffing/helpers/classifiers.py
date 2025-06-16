from typing import Iterable
from concurrent.futures import ProcessPoolExecutor
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
    window_name = str(window_name)

    logging.info("Running SVM for %s", window_name)
    x_train, x_test, y_train, y_test = train_test_split(
        sniff_count_window,
        sniff_count_window.index,
        test_size=test_train_split,
        random_state=RANDOM_SEED,
    )

    svm = LinearSVC(dual="auto", random_state=RANDOM_SEED)
    bagging_classifier = BaggingClassifier(
        svm, n_estimators=num_splits, random_state=RANDOM_SEED, n_jobs=-1
    )
    bagging_classifier.fit(x_train, y_train)
    bagged_score = bagging_classifier.score(x_test, y_test)

    for num, sub_estimator in enumerate(bagging_classifier.estimators_):
        logging.info("Getting scores from subestimator %i for concentration %s", num, window_name)
        sub_estimator_score = sub_estimator.score(x_test, y_test)
        sub_estimator_predictions = sub_estimator.predict(x_test)
        sub_estimator_cm = confusion_matrix(y_test, sub_estimator_predictions)

        individual_scores.append(sub_estimator_score)
        individual_CMS.append(sub_estimator_cm)

    return window_name, bagged_score, individual_scores, individual_CMS


def zscore_data(windowed_sniff_counts: pd.DataFrame) -> pd.DataFrame:
    transformed_data = StandardScaler().fit_transform(windowed_sniff_counts)
    return pd.DataFrame(transformed_data, columns=windowed_sniff_counts.columns)


def decode_trial_type(
    windowed_sniff_counts: pd.DataFrame,
    test_train_split: float = 0.2,
    num_splits: int = 20,
):
    scores: pd.Series = pd.Series(index=windowed_sniff_counts.columns)
    individual_scores: dict[str, pd.DataFrame] = dict.fromkeys(
        windowed_sniff_counts.columns
    )
    individual_CMS: dict[str, list[np.ndarray]] = dict.fromkeys(
        windowed_sniff_counts.columns
    )

    scaled_windowed_sniff_counts = zscore_data(windowed_sniff_counts)

    def svm_function(iterable):
        _run_svm(iterable, test_train_split=test_train_split, num_splits=num_splits)

    with ProcessPoolExecutor() as ppe:
        for name, bagged_score, individual_scores, individual_CMS in ppe.map(
            svm_function, scaled_windowed_sniff_counts.items()
        ):
            logging.info("Received results for %s", name)
            scores.loc[name] = bagged_score
            individual_scores[name] = pd.DataFrame(
                individual_scores, index=np.arange(len(individual_scores))
            )
            individual_CMS[name] = individual_CMS

    return scores, individual_scores, individual_CMS
