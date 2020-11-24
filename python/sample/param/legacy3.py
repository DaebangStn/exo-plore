import numpy as np

from python.util import *

#============================Configurations============================
CSV_PATH = "experiment_data/params/S500L3.csv"
NUM_SAMPLE = 500

USE_VELOCITY = False
USE_EXO = False

TIMESTAMP = False
NUM_DUPLICATE = 1

#=============================Parameters==============================
K_range = [0.0, 0.5]
Delay_range = [0.0, 0.5]
Phase_range = [0.64, 1.3]
Stride_range = [0.64, 1.3]
Velocity_range = [0.7, 1.9]

sigma = 0.1

#======================================================================
data = {
    "param_idx": np.arange(NUM_SAMPLE),
}

if USE_VELOCITY:
    data["Velocity"] = np.random.uniform(*Velocity_range, NUM_SAMPLE)
else:
    phase_sigma = np.random.standard_normal(NUM_SAMPLE) * sigma * (Phase_range[1] - Phase_range[0])
    data["Phase"] = np.where(phase_sigma > 0, phase_sigma + Phase_range[0], phase_sigma + Phase_range[1])
    stride_sigma = np.random.standard_normal(NUM_SAMPLE) * sigma * (Stride_range[1] - Stride_range[0])
    data["Stride"] = np.where(stride_sigma > 0, stride_sigma + Stride_range[0], stride_sigma + Stride_range[1])

if USE_EXO:
    data["K"] = np.random.uniform(*K_range, NUM_SAMPLE)
    data["Delay"] = np.random.uniform(*Delay_range, NUM_SAMPLE)

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

