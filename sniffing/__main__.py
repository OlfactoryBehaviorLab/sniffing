from .process_files import process_files

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Sniffing Analysis')
    parser.add_argument('-c', '--combined', action='store_true', help="Run combined analysis")
    parser.add_argument('-d', '--data_dir', help="Path to data directory")
    parser.add_argument('-o', '--output_dir', help="Path to output directory")
    args = parser.parse_args()

    if args.data_dir:
        _path = Path(args.data_dir)
        if not _path.exists():
            raise FileNotFoundError(f"Data directory {args.data_dir} does not exist")
        data_dir = _path
    else:
        data_dir = Path('/mnt/r2d2/11_Data/GoodSniffData')

    animal_dirs = data_dir.iterdir()
    animal_dirs = [_dir for _dir in animal_dirs if 'Z' not in _dir.name]

    if args.output_dir:
        _path = Path(args.output_dir)
        if not _path.exists():
            raise FileNotFoundError(f"Output directory {args.output_dir} does not exist")
        output_dir = _path
    else:
        output_dir = data_dir.joinpath('output')

    if not args.combined:
        for animal_dir in animal_dirs:
            h5_files = list(animal_dir.glob('*.h5'))
            output_dir.mkdir(exist_ok=True)
            process_files(h5_files, output_dir)
            break
    else:
        print('Combined!')



if __name__ == '__main__':
    main()