import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np

from pathlib import Path
import itertools

FILE_PATH = "/mnt/r2d2/5_Projects/1_Sniffing/3d_graph.xlsx"
ISB_BOTTOM = 51.85
ISB_SLOPE = 1.813
ISB_TOP = 93.31
ISB_EC50 = 0.06146

def hill_func(bottom: float, slope: float, top: float, ec50: float, x: float) -> float:
    exponent = np.power(x, slope)
    frac_top = np.subtract(top, bottom)
    frac_bottom = (exponent + np.power(ec50, slope))
    frac = np.divide(frac_top, frac_bottom)
    mult = np.multiply(exponent, frac)
    return bottom + mult


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
    isb_threshold_ppm_log10 = np.log10(isb_threshold_ppm)
    isb_threshold_sniffs = data["Isobutanol"]["Z"]
    isb_threshold_perf_SEM = data["Isobutanol"]["Y_SEM"]

    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    # Since matplotlib doesn't support log-scale in 3D
    # Workaround by https://github.com/matplotlib/matplotlib/issues/209#issuecomment-836684647

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    ax.view_init(elev=20)
    ax.set_ylim(50, 100)
    ax.set_zlim(1.6, 2.6)
    ax.set_xlabel("Concentration (ppm)")
    ax.set_ylabel("Behavioral Performance")
    ax.set_zlabel("Sniff Count")

    ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
    # Since matplotlib doesn't support log-scale in 3D
    # Workaround by https://github.com/matplotlib/matplotlib/issues/209#issuecomment-836684647


    ISB_X_GEN = np.logspace(-6, 3, 100, base=10, dtype=float)
    ISB_Y_GEN = np.vectorize(lambda x: hill_func(ISB_BOTTOM, ISB_SLOPE, ISB_TOP, ISB_EC50, x))(ISB_X_GEN)
    ISB_X_GEN_LOG10 = np.log10(ISB_X_GEN)
    ax.plot(ISB_X_GEN_LOG10, ISB_Y_GEN, zs=1.6, color="#FF6000",label="Isobutanol Performance")

    ISB_MARKER_Y = np.vectorize(lambda x: hill_func(ISB_BOTTOM, ISB_SLOPE, ISB_TOP, ISB_EC50, x))(isb_threshold_ppm)
    ax.fill_between(x1=isb_threshold_ppm_log10, x2=isb_threshold_ppm_log10, y1=ISB_MARKER_Y+isb_threshold_perf_SEM,  y2=ISB_MARKER_Y-isb_threshold_perf_SEM, z1=1.6, z2=1.6, alpha=0.4, color="#FF6000")
    markerline, stemlines, baseline = ax.stem(
        isb_threshold_ppm_log10,
        ISB_MARKER_Y,
        isb_threshold_sniffs,
        bottom=1.6,
        label="Isobutanol Sniffing",
    )


    markerline.set_markerfacecolor("blue")
    markerline.set_markeredgecolor("blue")
    stemlines.set_color("blue")
    baseline.set_color((0,0,0,0))

    ax.legend()
    plt.tight_layout()
    fig.tight_layout()
    plt.show(dpi=600)


if __name__ == "__main__":
    main()
