# Add Exo parameters to the Selected Parameters from ExoOptimization
from python.sample.util import *

#============================Configurations============================
CSV_PATH = "experiment_data/params/cot/selected_on_preprocess.csv"
K = 0.2666667
Delay = 0.25
selected_col = ["Phase", "Stride"]

#======================================================================
path = PROJECT_ROOT / CSV_PATH
df = pd.read_csv(CSV_PATH)
df = df[selected_col]
df["K"] = K
df["Delay"] = Delay

new_path = path.parent / "selected_exo.csv"
df.to_csv(new_path, index=False)
print(f"Saved to {new_path}")
