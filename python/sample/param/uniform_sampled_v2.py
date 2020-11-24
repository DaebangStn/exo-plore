import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

#============================Configurations============================
# Specify parameters as: {param_name: [start, end, num_intervals]}
param_specs = {
    # "Phase": [0.75, 1.25, 10], # pws
    "Phase": [0.7, 1.4, 15], # b20
    # "Phase": [0.6, 1.1, 22], # p20
    # "Phase": [1.013],
    # "Stride": [0.75, 1.25, 10], # pws
    "Stride": [0.2, 1.4, 15], # b20
    # "Stride": [0.4, 1.1, 22], # p20
    # "Stride": [1.013],
    # "K": [0.0], # k0
    # "K": [0.2], # k2
    # "K": [0.0, 0.4, 21], # df
    "K": [0.0, 0.7, 13], # df
    # "K": [0.0, 0.5, 15], # di
    # "K": [0.0, 0.5, 20], # di
    # "Delay": [0.0],
    # "Delay": [0.15], # d2
    # "Delay": [0.25], # d3
    # "Delay": [0.0, 0.5, 10], # d[f, g, h, i]
    "Delay": [0.0, 0.5, 20], # d[f, g, h, i]
    # "Delay": [0.0, 0.5, 50], # d[f, g, h, i]
    # "Mass": [1.4], # m
    # "Mass": [0.8, 1.2, 5], # m
    # "muscle_force_Calcaneal": [0.0], # 0.2 ~ 1.0
    # "muscle_force_Waddling": [0.0], # 0.3 ~ 1.0
    # "muscle_force_Footdrop": [0.0], # 0.0 ~ 1.0
    # "muscle_length_Hyperlordosis": [0.6], # 0.6 ~ 1.0
    # "muscle_length_Equinus": [0.73], # 0.75 ~ 1.0
}

SAVE_PARQUET = True
PARAM_NAME = 'exo_no_hei2'

#============================Sampling============================
param_names = list(param_specs.keys())
param_ranges = []
for spec in param_specs.values():
    if len(spec) == 1:
        # Single value case
        param_ranges.append([spec[0]])
    else:
        # Range case: [start, end, num]
        start, end, num = spec
        param_ranges.append(np.linspace(start, end, num))

# Create grid of all combinations
all_combinations = list(product(*param_ranges))

# Build DataFrame
param_idx = np.arange(len(all_combinations), dtype=np.int32)
data = np.column_stack([param_idx, np.array(all_combinations)])
columns = ['param_idx'] + param_names
df = pd.DataFrame(data, columns=columns)

# Convert float64 to float32 for all except param_idx
for col in param_names:
    if df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')

num_sample = len(df)

PARAM_PATH = f'experiment_data/params/{PARAM_NAME}_U{num_sample}'

#============================Save============================
if SAVE_PARQUET:
    parquet_path = Path(PARAM_PATH).with_suffix('.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"Uniform sampled parameters (v2) saved to {parquet_path}")
else:
    csv_path = Path(PARAM_PATH).with_suffix('.csv')
    df.to_csv(csv_path, index=False, sep=',', float_format='%.4f')
    print(f"Uniform sampled parameters (v2) saved to {csv_path}") 