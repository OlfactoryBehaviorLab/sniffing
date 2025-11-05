from pathlib import Path
import pandas as pd

OFFSET = 84

data_dir = Path(r'R:\5_Projects\1_Sniffing\3_benzaldehyde\Raw_Data_8-26-2025\212\output\mouse-212\1.e-06')
excels = data_dir.glob('*.xlsx')

files_to_correct = []

for excel in excels:
    try:
        int(excel.name[0])
        files_to_correct.append(excel)
    except:
        pass


for excel in files_to_correct:
    df = pd.read_excel(excel, index_col=0)
    index = df.index
    unique_trials_num = [int(trial[5:]) + OFFSET for trial in index]
    new_index = [f'Trial0{num}' for num in unique_trials_num]
    df.index = new_index
    df.to_excel(excel.with_stem(excel.stem+'-2'))