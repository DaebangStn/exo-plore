from python.util import *


#============================Configurations============================
CSV1_PATH = "experiment_data/params/U50_exo.csv"
CSV2_PATH = "experiment_data/params/U50_exo2.csv"
OUTPUT_PATH = "experiment_data/params/U100_exo.csv"

#======================================================================
csv1 = pd.read_csv(CSV1_PATH)
csv2 = pd.read_csv(CSV2_PATH)

col1 = set(csv1.columns)
col2 = set(csv2.columns)
assert col1 == col2, f"Merge failed: columns are not the same. Col1 = {col1}, Col2 = {col2}"

csv = pd.concat([csv1, csv2], ignore_index=True, axis=0)
csv['param_idx'] = np.arange(len(csv))
csv.to_csv(OUTPUT_PATH, index=False)
print(f"Merge successful: {len(csv)} rows are written to {OUTPUT_PATH}")

