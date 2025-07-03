import matplotlib.pyplot as plt
import numpy as np
from pandas import Series


def plot_multi_traces(traces: list[Series], trial_name) -> None:
    """
    Plots multiple traces in individual rows. Useful for visualizing raw v. preprocessed traces
    :param traces: (list[Series]) List of Pandas Series. Each series is a trace that is plotted on an individual subplot
    :return:
    """
    colors = plt.rcParams["axes.prop_cycle"]()
    figs, axs = plt.subplots(len(traces), figsize=(10, 10), sharex=True)
    for i, trace in enumerate(traces):
        color = next(colors)["color"]
        _ = axs[i].plot(trace, color=color)
        _ = axs[i].vlines(x=0, ymin=min(trace), ymax=max(trace), color="k")
        _ = axs[i].hlines(
            y=np.mean(trace), xmin=trace.index[0], xmax=trace.index[-1], color="m"
        )
    plt.suptitle(trial_name)

    plt.show()


def plot_crossing_frequencies(
    trace,
    true_inhales,
    true_exhales,
    inhale_frequencies,
    exhale_frequencies,
    inhale_times,
    exhale_times,
    crossings,
    trial_number,
    output_dir=None,
    display_plots=True,
):
    fig, ax = plt.subplots(2, sharex=True, figsize=(10, 7))
    ax[0].title.set_text(f"Trace w/ Peaks and Crossings | Trial: {trial_number}")
    ax[1].set_ylabel("Frequency (Hz)")
    ax[1].set_xlabel("Time (ms)")
    ax[1].title.set_text("Instantaneous Frequency")
    _ = ax[0].plot(trace, color="c")
    _ = ax[1].plot(inhale_times, inhale_frequencies, color="red", label="inhale")
    _ = ax[1].plot(exhale_times, exhale_frequencies, color="orange", label="exhale")
    _ = ax[0].vlines(x=0, ymin=min(trace), ymax=max(trace), color="k")
    _ = ax[0].hlines(
        y=np.mean(trace),
        xmin=trace.index[0],
        xmax=trace.index[-1],
        alpha=0.3,
        color="m",
    )
    _ = ax[0].scatter(crossings, np.zeros(len(crossings)), marker="o", color="g")
    # _ = ax[0].scatter(true_inhales.index, true_inhales['magnitude'].values, marker='x', color='r', label='inhale')
    # _ = ax[0].scatter(true_exhales.index, true_exhales['magnitude'].values, marker='x', color='orange', label='exhale')
    _ = ax[0].scatter(
        true_inhales, true_inhales.index, marker="x", color="r", label="inhale"
    )
    _ = ax[0].scatter(
        true_exhales, true_exhales.index, marker="x", color="orange", label="exhale"
    )
    _ = ax[0].legend()

    _ = ax[1].legend(["inhale", "exhale"])

    if output_dir is not None:
        output_path = output_dir.joinpath(f"trial_{trial_number}.pdf")
        fig.savefig(output_path, dpi=600)
        plt.close()

    if display_plots:
        plt.show()


def plot_true_sniffs(
    trace,
    true_inhales,
    all_inhales,
    all_exhales,
    crossings,
    trial_number,
    output_dir=None,
    display_plots=True,
):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.title.set_text(f"Trace w/ Peaks and Crossings | Trial: {trial_number}")
    ax.set_xlabel("Time (ms)")
    _ = ax.plot(trace, color="c")
    _ = ax.vlines(x=0, ymin=min(trace), ymax=max(trace), color="k")
    _ = ax.hlines(
        y=np.mean(trace),
        xmin=trace.index[0],
        xmax=trace.index[-1],
        alpha=0.3,
        color="m",
    )
    _ = ax.scatter(crossings, np.zeros(len(crossings)), marker="o", color="g")
    _ = ax.scatter(
        all_inhales, all_inhales.index, marker="x", color="r", label="inhale"
    )
    _ = ax.scatter(
        all_exhales, all_exhales.index, marker="x", color="orange", label="exhale"
    )
    _ = ax.scatter(
        true_inhales.index, true_inhales, marker="*", color="m", label="true_inhale"
    )
    _ = ax.legend()

    if output_dir is not None:
        output_path = output_dir.joinpath(f"trial_{trial_number}.pdf")
        fig.savefig(output_path, dpi=600)
        plt.close()

    if display_plots:
        plt.show()


def plot_traces(
    raw_trace,
    filtered_trace,
    lick_timestamps,
    true_inhales,
    all_exhales,
    trial_number,
    output_dir=None,
    display=False,
    tpe: bool = False,
):
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    plt.suptitle(
        f"Raw vs. Filtered Trace | Trial: {trial_number}",
        fontsize=16,
        fontweight="bold",
    )
    _ = axs[0].title.set_text("Raw Trace")
    _ = axs[0].plot(raw_trace, color="k")
    _ = axs[0].set_ylabel("Signal (au)")

    _ = axs[1].title.set_text("Filtered Trace with Sniffs")
    _ = axs[1].plot(filtered_trace, color="c", zorder=2)
    _ = axs[1].vlines(
        x=0, ymin=min(filtered_trace), ymax=max(filtered_trace), zorder=1, color="k"
    )
    _ = axs[1].set_ylabel("Signal (Standard Deviations)")
    _ = axs[1].set_xlabel("Time Relative to Odor Onset (ms)")
    _ = axs[1].scatter(
        all_exhales,
        all_exhales.index,
        marker="*",
        color="red",
        s=50,
        zorder=3,
        label="exhale",
    )
    _ = axs[1].scatter(
        true_inhales.index,
        true_inhales,
        marker="*",
        color="m",
        s=50,
        zorder=3,
        label="inhale",
    )

    max_y = max(filtered_trace)
    _trimmed_lick_timestamps = np.intersect1d(filtered_trace.index, lick_timestamps)
    # Don't plot any licks after the sniff data is done
    _ = axs[1].vlines(
        x=_trimmed_lick_timestamps,
        ymin=max_y * 0.9,
        ymax=max_y,
        zorder=10,
        color="r",
        label="sniffs",
    )

    fig.legend(fontsize=12, bbox_to_anchor=(0.988, 1))
    plt.tight_layout()

    if display:
        plt.show()

    if output_dir is not None:
        output_path = output_dir.joinpath(f"trial_{trial_number}.pdf")
        if tpe:
            return fig, output_path

        fig.savefig(output_path, dpi=600)
        plt.close()

    return None, None


def plot_binned_frequencies(
    traces, titles, num_trials, mouse, concentration, display_plots=True
):
    colors = ["m", "c", "g", "orange"]
    num_traces = len(traces)
    fig, ax = plt.subplots(num_traces, 1, figsize=(10, 10), sharex=True, sharey=True)

    max_y = [max(trace) for trace in traces if len(trace) > 0]
    max_y = max(max_y) * 1.1

    for i, trace in enumerate(traces):
        ax[i].plot(trace, color=colors[i])
        ax[i].set_title(f"{titles[i]} - Num Trials: {num_trials[i]}")
        ax[i].set_ylim(0, max_y)
    plt.suptitle(f"Mouse: {mouse} - Concentration: {concentration}")
    fig.supxlabel("Time (ms, relative to odor onset)")
    fig.supylabel("Frequency (Hz)")

    # for i, trace in enumerate(traces):
    #     ax[i].bar(trace.index.values, trace.values, width=1.5, color=colors[i])
    #     ax[i].set_title(f'{titles[i]} - Num Trials: {num_trials[i]}')
    #     # ax[i].set_ylim(0, max_y)
    # plt.suptitle(f'Mouse: {mouse} - Concentration: {concentration}')
    # fig.supxlabel('Time (ms, relative to odor onset)')
    # fig.supylabel('Frequency (Hz)')

    plt.tight_layout()

    if display_plots:
        plt.show()

    return fig, ax
