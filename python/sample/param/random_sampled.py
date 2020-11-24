import numpy as np

from python.util import *

#============================Configurations============================
# CSV_PATH = "experiment_data/params/S5K_exo_fix_gait.csv"
CSV_PATH = "experiment_data/params/debug.csv"
NUM_SAMPLE = 200

USE_VELOCITY = False
USE_GAIT = False  # if false, fix the phase and stride
USE_EXO = True
SAMPLE_DIAG = False

TIMESTAMP = False
NUM_DUPLICATE = 1

num_zero_exo = 5
#=============================Parameters==============================
K_range = [0.0, 0.5]
Delay_range = [0.0, 0.75]
Phase_range = [0.64, 1.3]
Stride_range = [0.64, 1.3]
Velocity_range = [0.7, 1.9]

diag_alpha = 0.05

#======================================================================
data = {
    "param_idx": np.arange(NUM_SAMPLE),
}

if USE_VELOCITY:
    data["Velocity"] = np.random.uniform(*Velocity_range, NUM_SAMPLE)
else:
    if USE_GAIT:
        if SAMPLE_DIAG:
            first_random = np.random.rand(NUM_SAMPLE)
            phase_random = np.random.standard_normal(NUM_SAMPLE) * diag_alpha
            stride_random = np.random.standard_normal(NUM_SAMPLE) * diag_alpha
            phase = first_random * (max(Phase_range) - min(Phase_range)) + min(Phase_range) + phase_random
            stride = first_random * (max(Stride_range) - min(Stride_range)) + min(Stride_range) + stride_random
            data["Phase"] = np.clip(phase, *Phase_range)
            data["Stride"] = np.clip(stride, *Stride_range)
        else:
            data["Phase"] = np.random.uniform(*Phase_range, NUM_SAMPLE)
            data["Stride"] = np.random.uniform(*Stride_range, NUM_SAMPLE)
    else:
        fixed_phase = 1
        fixed_stride = 1
        data["Phase"] = np.repeat(fixed_phase, NUM_SAMPLE)
        data["Stride"] = np.repeat(fixed_stride, NUM_SAMPLE)

if USE_EXO:
    # data["K"] = np.random.uniform(*K_range, NUM_SAMPLE - num_zero_exo).tolist() + [0] * num_zero_exo
    data["K"] = np.repeat(0.5, NUM_SAMPLE - num_zero_exo).tolist() + [0] * num_zero_exo
    data["Delay"] = np.random.uniform(*Delay_range, NUM_SAMPLE - num_zero_exo).tolist() + [0] * num_zero_exo

duplicated_data = {
    "param_idx": np.arange(NUM_SAMPLE * NUM_DUPLICATE),
}

if USE_VELOCITY:
    duplicated_data["Velocity"] = np.repeat(data["Velocity"], NUM_DUPLICATE)
else:
    duplicated_data["Phase"] = np.repeat(data["Phase"], NUM_DUPLICATE)
    duplicated_data["Stride"] = np.repeat(data["Stride"], NUM_DUPLICATE)

if USE_EXO:
    duplicated_data["K"] = np.repeat(data["K"], NUM_DUPLICATE)
    duplicated_data["Delay"] = np.repeat(data["Delay"], NUM_DUPLICATE)



# Create a DataFrame
df = pd.DataFrame(duplicated_data)

csv_path = Path(CSV_PATH)
if TIMESTAMP:
    csv_filename = csv_path.stem + "_" + timestamp() + ".csv"
    csv_path = csv_path.parent / csv_filename

df.to_csv(csv_path, index=False, sep=',')

