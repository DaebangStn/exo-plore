from python.util import *

#============================Configurations============================
NUM_SAMPLE = 500
# CSV_PATH = 'experiment_data/params/p20_df_equi_U3000'
CSV_PATH = f'experiment_data/params/b21hyp60_k0_U{NUM_SAMPLE}'

# if false, fix the phase and stride
CHANGE_GAIT = True
USE_EXO = True
CHANGE_EXO = False

SAVE_PARQUET = True
#=============================Parameters==============================
# K_range = [0.0, 0.6] # da
# K_range = [0.2, 0.5] # db
# K_range = [0.2, 0.7] # dc
# K_range = [0.2, 0.6] # dd
# K_range = [0.1, 0.65] # de
K_range = [0.0, 0.7] # df
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

# positive gain
# fixed_k = [0.26667]
# fixed_k = [0.5]
fixed_k = [0.0]
# negative gain
# fixed_k = [-0.13333]
fixed_delay = [0.0]

extras = {
    # "muscle_force_Calcaneal": [0.2, 1.0],
    # "muscle_force_Waddling": [0.3, 1.0],
    # "muscle_force_Footdrop": [0.0, 1.0],
    # "muscle_length_Equinus": [0.75, 1.0],
    "muscle_length_Hyperlordosis": [0.6, 0.6],
    # "mass": [0.7, 1.4],
}

#======================================================================

true_count = sum([2 * CHANGE_GAIT, 2 * (CHANGE_EXO and USE_EXO), len(extras.keys())])
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
        extra_value = np.linspace(*extras[extra_key], grid_size)
        param_lists.extend([extra_value])
permutes_array = np.array(list(product(*param_lists)))

columns = ['param_idx', 'Phase', 'Stride']
param_idx = np.arange(len(permutes_array), dtype=np.int32)
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

# if float64, convert to float32
cols = df.columns
for col in cols:
    if df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')

if SAVE_PARQUET:
    parquet_path = Path(CSV_PATH).with_suffix('.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"Uniform sampled parameters saved to {parquet_path}")
else:
    csv_path = Path(CSV_PATH).with_suffix('.csv')
    df.to_csv(csv_path, index=False, sep=',', float_format='%.4f')
    print(f"Uniform sampled parameters saved to {csv_path}")
