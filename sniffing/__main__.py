import argparse
import os
import pathlib
import sys
import warnings
import logging
from typing import Tuple

import pandas as pd
from PySide6.QtWidgets import QFileDialog, QApplication
from pathlib import Path

from sniffing.process_files import process_files
from sniffing.combined import process_combined

DEFAULT_DIR = Path("/mnt/r2d2/5_Projects/Concentration_Sniffing_Dynamics/Raw_Data")

logging.basicConfig(level=logging.DEBUG)


def select_dialog(file=False):
    dialog = QFileDialog()
    if file:
        _val, _ = dialog.getOpenFileName(
            None, "Select H5 File: ", filter="H5 Files (*.h5)"
        )
    else:
        _val = dialog.getExistingDirectory(None, "Select Data Directory: ")
    dialog.hide()
    dialog.close()
    return _val


def get_spss_files(concentration_dir: pathlib.Path, animal) -> Tuple:
    file1 = _check_exists_or_warn(concentration_dir, "1*count.xlsx", animal)
    file2 = _check_exists_or_warn(concentration_dir, "2*duration.xlsx", animal)
    file3 = _check_exists_or_warn(concentration_dir, "3*ISI.xlsx", animal)
    file4 = _check_exists_or_warn(concentration_dir, "4*lengths.xlsx", animal)
    file5 = _check_exists_or_warn(concentration_dir, "5*bins.xlsx", animal)
    file6 = _check_exists_or_warn(concentration_dir, "6*all_sniff_durs.xlsx", animal)
    file6_nogo = _check_exists_or_warn(concentration_dir, "6*durs_correct_nogo.xlsx", animal)

    return file1, file2, file3, file4, file5, file6, file6_nogo


def _check_exists_or_warn(conc_dir: pathlib.Path, file_name: str, animal: str):
    glob_list = list(conc_dir.glob(file_name))

    if not glob_list:
        warnings.warn(
            f"{conc_dir.name}-{animal} is missing {file_name}! Skipping...",
            stacklevel=2,
        )
        return None

    return glob_list[0]


def main():
    data_dir = None
    file_path = None
    output_dir = None

    qt_available = False

    if "WSL_DISTRO_NAME" not in os.environ:
        qt_available = True
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)

    parser = argparse.ArgumentParser(description="Sniffing Analysis")
    parser.add_argument(
        "-s",
        "--single",
        help="Path to single file to process",
        nargs="?",
        const="",
        default=None,
    )
    parser.add_argument(
        "-c", "--combined", action="store_true", help="Run combined analysis"
    )
    # parser.add_argument(
    #     "-b", "--batch", action="store_true", help="Batch process H5 files"
    # )
    parser.add_argument(
        "-d", "--data_dir", help="Path to data directory, implies --batch"
    )
    parser.add_argument(
        "-i",
        "--ignore_errors",
        default=False,
        help="Continue processing files and ignore any errors",
        action="store_true",
    )
    parser.add_argument("-o", "--output_dir", help="Path to output directory")
    args = parser.parse_args()

    # if args.single and args.batch:
    #     warnings.warn(
    #         "Cannot batch process a single file! Defaulting to single file",
    #         stacklevel=2,
    #     )
    #     args.batch = False

    if args.single and args.combined:
        raise argparse.ArgumentError(
            args.single,
            "Cannot run combined processing on a single file! Please specify a data directory!",
        )

    # if args.combined and args.batch:
    #     warnings.warn('Batch processing (-b) is implied for combined processing!', stacklevel=2)

    # If we specify single, we're only processing one file
    if args.single is not None:
        if len(args.single) == 0:
            if qt_available:
                _path = select_dialog(file=True)
            else:
                raise RuntimeError("QT is unavailable! Cannot open selection dialog")
            if len(_path) == 0:
                raise FileNotFoundError("No file selected!")
            file_path = Path(_path)
        else:
            _file_path = Path(args.single)
            # Is the file the user passed real?
            if not _file_path.exists():
                raise FileNotFoundError(f"File {args.single} does not exist!")
            file_path = _file_path
    # The default is batch processing, so it is the mode if the user didn't specify a single file
    else:
        print("Batch Processing!")
        # Do we need/have a data directory?
        if args.data_dir:
            _path = Path(args.data_dir)
            if not _path.exists():
                raise FileNotFoundError(
                    f"Data directory {args.data_dir} does not exist"
                )
            data_dir = _path
        else:
            if qt_available:
                _path = select_dialog()
            else:
                raise RuntimeError("QT is unavailable! Cannot open selection dialog")
            if len(_path) > 0:
                data_dir = Path(_path)
            else:
                warnings.warn(
                    f"No data directory was selected! Using {DEFAULT_DIR}", stacklevel=2
                )
                _path = Path(DEFAULT_DIR)
                if not _path.exists():
                    raise FileNotFoundError(
                        f"Default Directory {DEFAULT_DIR} not found!"
                    )
                data_dir = _path

    # Use or create output directory
    if args.output_dir:
        _path = Path(args.output_dir)
        _path.mkdir(exist_ok=True, parents=True)
        # if not _path.exists():
        #     raise FileNotFoundError(f"Output directory {args.output_dir} does not exist")
        output_dir = _path
    else:
        if file_path:
            output_dir = file_path.parent.joinpath("output")
        elif data_dir:
            output_dir = data_dir.parent.joinpath("output")
        output_dir.mkdir(exist_ok=True)

    if not args.combined:
        if data_dir:
            animal_dirs = data_dir.iterdir()
            animal_dirs = [_dir for _dir in animal_dirs if "Z" not in _dir.name]
            all_h5_stats = pd.DataFrame()
            for animal_dir in animal_dirs:
                h5_files = list(animal_dir.glob("*.h5"))
                _animal_h5_stats = process_files(
                    h5_files, output_dir, ignore_errors=args.ignore_errors
                )
                all_h5_stats = pd.concat(
                    [all_h5_stats, _animal_h5_stats], ignore_index=True
                )
            h5_stats_output_path = output_dir.joinpath("all_h5_stats.xlsx")
            all_h5_stats.to_excel(h5_stats_output_path)
        elif file_path:
            _ = process_files([file_path], output_dir)
    else:
        concentration_files = {}
        animal_dirs = [_dir for _dir in data_dir.iterdir() if _dir.is_dir()]

        for animal_dir in animal_dirs:
            concentration_dirs = [
                _dir for _dir in animal_dir.iterdir() if _dir.is_dir()
            ]

            for concentration_dir in concentration_dirs:
                if concentration_dir.name not in concentration_files:
                    concentration_files[concentration_dir.name] = {}

                windowed_bin_counts = concentration_dir.joinpath(
                    "binned_sniff_counts.xlsx"
                )
                combined_data_matrix = _check_exists_or_warn(
                    concentration_dir, "*TrialParams.xlsx", animal_dir.name
                )

                if combined_data_matrix is None:
                    continue

                all_traces = _check_exists_or_warn(
                    concentration_dir, "all_trimmed_traces.xlsx", animal_dir.name
                )
                if all_traces is None:
                    continue

                file1, file2, file3, file4, file5, file6, file6_nogo = get_spss_files(
                    concentration_dir, animal_dir.name
                )

                concentration_files[concentration_dir.name][animal_dir.name] = {
                    "combined": combined_data_matrix,
                    "window": windowed_bin_counts,
                    "traces": all_traces,
                    "1": file1,
                    "2": file2,
                    "3": file3,
                    "4": file4,
                    "5": file5,
                    "6": file6,
                    "6_nogo": file6_nogo
                }

        process_combined(concentration_files, output_dir)


if __name__ == "__main__":
    main()
