from python.util import *

#============================Configurations============================
CSV_PATH = "experiment_data/params/U4k_gems.csv"
NUM_SAMPLE = 4000

#=============================Parameters==============================
K = [0.2667, 0.4]
Delay = [0.05, 0.15, 0.25, 0.35]
# Phase_range = [0.8, 1.2]
# Stride_range = [0.8, 1.2]
Phase_range = [0.4, 1.4]
Stride_range = [0.4, 1.4]

#======================================================================
grid_size = int((NUM_SAMPLE / (len(K) * len(Delay))) ** (1 / 2))
Phase_range = np.linspace(*Phase_range, grid_size)
Stride_range = np.linspace(*Stride_range, grid_size)

param_combinations = list(product(K, Delay, Phase_range, Stride_range))
num_samples = len(param_combinations)
data = {
    "param_idx": np.arange(num_samples),
    "K": [p[0] for p in param_combinations],
    "Delay": [p[1] for p in param_combinations],
    "Phase": [p[2] for p in param_combinations],
    "Stride": [p[3] for p in param_combinations],
}

# Create a DataFrame
df = pd.DataFrame(data)
csv_path = Path(CSV_PATH)
df.to_csv(csv_path, index=False, sep=',')
