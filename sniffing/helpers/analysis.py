import numpy as np
import pandas as pd
from scipy import signal


def _instantaneous_frequency(peaks):
    pairs = zip(peaks.iloc[:-1].index, peaks.iloc[1:].index)
    frequencies = []
    times = []

    for inhale1, inhale2 in pairs:
        delta_t = abs(inhale2 - inhale1)
        frequency = round(1 / (delta_t / 1000), 2)
        frequencies.append(frequency)
        times.append(inhale1 + (delta_t / 2))

    return frequencies, times


def calc_frequencies(true_inhales, true_exhales):
    inhale_frequencies, inhale_times = _instantaneous_frequency(true_inhales)
    exhale_frequencies, exhale_times = _instantaneous_frequency(true_exhales)

    return inhale_frequencies, exhale_frequencies, inhale_times, exhale_times