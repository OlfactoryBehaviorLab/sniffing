from pathlib import Path
import pandas as pd

# Process file 1 and rename its output directory
# Process file 2 and leave the output directory
# set data_dir to the file_2 output directory and update file_1_path to be the name of the renamed first directory
# Combined data files will be output to the file 1 directory



OFFSET = 84

file_2_dir = Path(r'R:\5_Projects\1_Sniffing\3_benzaldehyde\Raw_Data_8-26-2025\212\output\mouse-212\1.e-06') # Second data file output directory
file_1_dir = file_2_dir.with_name('1.e-06-f1') # First data file output directory
f2_excel = file_2_dir.glob('*.xlsx') # Get all second file data files
output_dir = file_1_dir.parent.joinpath('fixed')
output_dir.mkdir(parents=True, exist_ok=True)


# Get the H5 params to combine them
file_2_params_path = list(file_2_dir.glob('*TrialParams*'))[0]
file_1_params_path = list(file_1_dir.glob('*TrialParams*'))[0]

files_to_correct = []
files_to_correct_2 = []

for excel in f2_excel:
    try:
        # Get the data files that are prefixed with numbers; these have trial number as the index
        int(excel.name[0])
        files_to_correct.append(excel)
    except:
        # These data files have Trial number as the column name; skip trial params
        if "TrialParams" not in excel.name:
            files_to_correct_2.append(excel)


def reindex(trial_numbers: list) -> list:
    unique_trials_num = [int(trial[5:]) + OFFSET for trial in trial_numbers]
    new_index = [f'Trial0{num}' for num in unique_trials_num]

    return new_index

for excel in files_to_correct:
    # Renumber and combine all file 2 trials
    df = pd.read_excel(excel, index_col=0)
    original_file_path = file_1_dir.joinpath(excel.name)
    original_file_df = pd.read_excel(original_file_path, index_col=0)

    df.index = reindex(df.index)

    new_df = pd.concat([original_file_df, df], axis=0)
    new_df_path = output_dir.joinpath(original_file_path.name)
    new_df.to_excel(new_df_path)

for excel in files_to_correct_2:
    # renumber and combine all file 2 trials but in the other direction
    df = pd.read_excel(excel, index_col=0)
    original_file_path = file_1_dir.joinpath(excel.name)
    original_file_df = pd.read_excel(original_file_path, index_col=0)

    df.columns = reindex(df.columns)

    new_df = pd.concat([original_file_df, df], axis=1)
    new_df_path = output_dir.joinpath(original_file_path.name)
    new_df.to_excel(new_df_path)

# Reindex h5 params file
file_1_params_df = pd.read_excel(file_1_params_path, index_col=0)
file_2_params_df = pd.read_excel(file_2_params_path, index_col=0)
file_2_params_df.index = reindex(file_2_params_df.index)

new_params = pd.concat([file_1_params_df, file_2_params_df], axis=0)
new_params_path = output_dir.joinpath(file_1_params_path.name)
new_params.to_excel(new_params_path)