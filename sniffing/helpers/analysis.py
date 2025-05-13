import numpy as np
import pandas as pd
from scipy import signal, fft
import matplotlib.pyplot as plt

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


def trial_fft(trace):
    num_samples = trace.shape[0]
    sample_space = 1/1000

    # stft = signal.ShortTimeFFT.from_window('hann', fs=sample_rate, nperseg=50, noverlap=25)
    # stft = signal.ShortTimeFFT.from_window('hann', fs=1000, nperseg=200, noverlap=100)
    # fft_results = stft.stft(trace.values)
    # window = signal.windows.hann(num_samples)
    # trace_window = trace * window
    fft_results = fft.rfft(trace)[1:]
    x_vals = fft.rfftfreq(num_samples, sample_space)[1:]
    largest_freq = x_vals[np.argmax(np.abs(fft_results))]

    plt.suptitle(f'Largest Freq {largest_freq}')
    plt.plot(x_vals, np.abs(fft_results) * 1/num_samples)
    plt.xlim(-3, 40)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (percentage)')
    # plt.imshow(np.abs(fft_results), aspect='auto', origin='lower', extent=stft.extent(num_samples))
    # plt.colorbar()
    # plt.ylim(0,50)
    plt.show()
