from python.util import *

#============================Configurations============================
NUM_SAMPLE = 100
DUP_NUM = 10
# FILE_PATH = f'experiment_data/params/b21_U{NUM_SAMPLE}x{DUP_NUM}.parquet'
FILE_PATH = f'experiment_data/params/p21_calc4_U{NUM_SAMPLE}x{DUP_NUM}.parquet'

# if false, fix the phase and stride
CHANGE_GAIT = True
USE_EXO = False
CHANGE_EXO = False

#=============================Parameters==============================
# K_range = [0.0, 0.6] # da
# K_range = [0.2, 0.5] # db
# K_range = [0.2, 0.7] # dc
# K_range = [0.2, 0.6] # dd
# K_range = [0.1, 0.65] # de
# K_range = [0.0, 0.7] # df
# K_range = [0.0, 1.0] # dg
# K_range = [0.0, 1.5] # dh
# K_range = [0.0, 0.5] # di

# Delay_range = [0.0, 0.4] # d[a, b, c, d] 
# Delay_range = [0.0, 0.45] # de 
Delay_range = [0.0, 0.5] # d[f, g, h, i]

# Phase_range = [0.8, 1.4] # b19
# Phase_range = [0.7, 1.4] # b20
Phase_range = [0.6, 1.1] # p20

# Stride_range = [0.4, 1.4] # b19
# Stride_range = [0.2, 1.4] # b20
Stride_range = [0.4, 1.1] # p20

fixed_phase = [1]
fixed_stride = [1]
fixed_k = [0.26667]
fixed_delay = [0.25]

extras = {
    "muscle_force_Calcaneal": [0.4], # 0.2 ~ 1.0
    # "muscle_force_Waddling": [0.4], # 0.3 ~ 1.0
    # "muscle_force_Footdrop": [0.5], # 0.0 ~ 1.0
    # "muscle_length_Hyperlordosis": [1.0], # 0.6 ~ 1.0
    # "muscle_length_Equinus": [1.0], # 0.75 ~ 1.0
}

#======================================================================

true_count = sum([2 * CHANGE_GAIT, 2 * (CHANGE_EXO and USE_EXO)])
assert true_count > 0, "At least one of USE_GAIT and USE_EXO must be true"
assert len(extras.keys()) < 2, "Only one extra parameter is allowed"

grid_size = int(NUM_SAMPLE ** (1 / true_count))
NUM_SAMPLE = grid_size ** (true_count)

info = f"exponent: {true_count}  "
for i in range(true_count):
    info += f"{grid_size}x"
info = info[:-1]
info += f", Sample#: {grid_size ** true_count}\n"
info += "\n"
if CHANGE_GAIT:
    info += f"Phase: {Phase_range}, Stride: {Stride_range}\n"
else:
    info += f"Phase: {fixed_phase}, Stride: {fixed_stride}\n"
if USE_EXO:
    if CHANGE_EXO:
        info += f"K: {K_range}, Delay: {Delay_range}\n"
    else:
        info += f"K: {fixed_k}, Delay: {fixed_delay}\n"
if len(extras.keys()) > 0:
    info += f"Extra: {extras.keys()}\n"
print(info)

param_lists = []
if CHANGE_GAIT:
    Phase_value = np.linspace(*Phase_range, grid_size)
    Stride_value = np.linspace(*Stride_range, grid_size)
else:
    Phase_value = fixed_phase
    Stride_value = fixed_stride
param_lists.extend([Phase_value, Stride_value])

if USE_EXO:
    if CHANGE_EXO:
        K_value = np.linspace(*K_range, grid_size)
        Delay_value = np.linspace(*Delay_range, grid_size)        
    else:
        K_value = fixed_k
        Delay_value = fixed_delay
    param_lists.extend([K_value, Delay_value])

if len(extras.keys()) > 0:
    for extra_key in extras.keys():
        param_lists.extend([extras[extra_key]])
permutes_array = np.array(list(product(*param_lists)))
permutes_array = np.repeat(permutes_array, DUP_NUM, axis=0)
columns = ['param_idx', 'Phase', 'Stride']
param_idx = np.arange(len(permutes_array))
Phase_values = permutes_array[:, 0]
Stride_values = permutes_array[:, 1]
data = [param_idx, Phase_values, Stride_values]

if USE_EXO:
    columns.extend(['K', 'Delay'])
    K_values = permutes_array[:, 2]
    Delay_values = permutes_array[:, 3]
    data.extend([K_values, Delay_values])

if len(extras.keys()) > 0:
    columns.extend(extras.keys())
    for i, extra_key in enumerate(extras.keys()):
        extra_values = permutes_array[:, len(columns) + i - 2]
        data.extend([extra_values])

# Transpose data to have correct shape for DataFrame
data = np.array(data).T
df = pd.DataFrame(data, columns=columns)

file_path = Path(FILE_PATH)
df.to_parquet(file_path, index=False)
print(f"Uniform sampled parameters saved to {file_path}")
