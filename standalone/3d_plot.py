import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np

from pathlib import Path
import itertools

FILE_PATH = "/mnt/r2d2/5_Projects/1_Sniffing/3d_graph.xlsx"


def log_tick_formatter(val, pos=None):
    return r"$10^{{{:.0f}}}$".format(val)


def main():
    data = pd.read_excel(FILE_PATH)
    odors = ["Isobutanol", "Sec-Butyl-Acetate"]
    col_labels = ["X", "X_SEM", "Y", "Y_SEM", "Z", "Z_SEM"]
    new_cols = pd.MultiIndex.from_product((odors, col_labels))
    data.columns = new_cols
    data = data.drop(0, axis=0)
    data = data.infer_objects()

    isb_threshold_ppm = data["Isobutanol"]["X"]
    isb_threshold_ppm_log10 = np.log(isb_threshold_ppm)

    isb_threshold_perf = data["Isobutanol"]["Y"]
    isb_threshold_sniffs = data["Isobutanol"]["Z"]
    ax = plt.figure().add_subplot(projection="3d")

    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    # Since matplotlib doesn't support log-scale in 3D
    # Workaround by https://github.com/matplotlib/matplotlib/issues/209#issuecomment-836684647

    ax.set_zlim(0, 3)

    ax.set_xlabel("Concentration (ppm)")
    ax.set_ylabel("Behavioral Performance")
    ax.set_zlabel("Sniff Count")

    # ax.plot(isb_threshold_ppm_log10, isb_threshold_perf, c='magenta', label='Isobutanol Performance')
    markerline, stemlines, baseline = ax.stem(
        isb_threshold_ppm_log10,
        isb_threshold_perf,
        isb_threshold_sniffs,
        label="Isobutanol Performance",
    )
    baseline.set_color('m')
    baseline.set_linewidth(2)
    ax.legend()
    plt.show(dpi=600)


if __name__ == "__main__":
    main()
