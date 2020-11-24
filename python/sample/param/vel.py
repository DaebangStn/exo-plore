from python.util import *

#============================Configurations============================
CSV_PATH = "experiment_data/params/vel_ms.csv"

#=============================Parameters==============================
Velocity_ms = np.linspace(0.5, 2.0, 20)

#======================================================================
num_samples = len(Velocity_ms)
data = {
    "param_idx": np.arange(num_samples),
    "Velocity": Velocity_ms,
}

# Create a DataFrame
df = pd.DataFrame(data)
csv_path = Path(CSV_PATH)
df.to_csv(csv_path, index=False, sep=',')

