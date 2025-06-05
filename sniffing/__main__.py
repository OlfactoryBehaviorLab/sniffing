from .process_files import process_files

import argparse
import sys
import warnings

from PySide6.QtWidgets import QFileDialog, QApplication
from pathlib import Path

DEFAULT_DIR = Path("/mnt/r2d2/5_Projects/Concentration_Sniffing_Dynamics/Raw_Data")


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


def main():
    data_dir = None
    file_path = None
    output_dir = None

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
    parser.add_argument(
        "-b", "--batch", action="store_true", help="Batch process H5 files"
    )
    parser.add_argument(
        "-d", "--data_dir", help="Path to data directory, implies --batch"
    )
    parser.add_argument("-o", "--output_dir", help="Path to output directory")
    args = parser.parse_args()

    if args.single and args.batch:
        warnings.warn("Cannot batch process a single file! Defaulting to single file", stacklevel=2)
        args.batch = False

    # If we specify single, we're only processing one file
    if args.single is not None:
        if len(args.single) == 0:
            _path = select_dialog(file=True)
            if len(_path) == 0:
                raise FileNotFoundError("No file selected!")
            file_path = Path(_path)
        else:
            _file_path = Path(args.single)
            # Is the file the user passed real?
            if not _file_path.exists():
                raise FileNotFoundError(f"File {args.single} does not exist!")
            file_path = _file_path

    # The default is batch processing, so see if the user explicitly specified, or didn't specify a single file
    elif args.batch or not args.single:
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
            _path = select_dialog()
            if len(_path) > 0:
                data_dir = Path(_path)
            else:
                warnings.warn(f"No data directory was selected! Using {DEFAULT_DIR}", stacklevel=2)
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

            for animal_dir in animal_dirs:
                h5_files = list(animal_dir.glob("*.h5"))
                process_files(h5_files, output_dir)
        elif file_path:
            process_files([file_path], output_dir)
    else:
        print("Combined!")


if __name__ == "__main__":
    main()
