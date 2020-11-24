from python.util import *


#============================Configurations============================
CSV_PATH = "experiment_data/params/S50D.csv"
NUM_SAMPLE = 50


Phase_range = [0.64, 1.3]
Stride_range = [0.64, 1.3]


#======================================================================
data = {
    "param_idx": np.arange(NUM_SAMPLE),
    "Phase": np.linspace(*Phase_range, NUM_SAMPLE),
    "Stride": np.linspace(*Stride_range, NUM_SAMPLE),
}


df = pd.DataFrame(data)
csv_path = Path(CSV_PATH)
df.to_csv(csv_path, index=False, sep=',')
