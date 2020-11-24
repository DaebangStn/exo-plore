# Add Exo parameters to the Selected Parameters from ExoOptimization
from python.sample.util import *

#============================Configurations============================
CSV_IN = "experiment_data/params/b13_U200_k2_d4.csv"

# CSV_OUT = "experiment_data/params/U200_patho_hype.csv"
# CSV_OUT = "experiment_data/params/U200_patho_wadd.csv"
# CSV_OUT = "experiment_data/params/U200_patho_equi.csv"
# CSV_OUT = "experiment_data/params/U200_patho_foot.csv"
# CSV_OUT = "experiment_data/params/U200_patho_calc.csv"
CSV_OUT = "experiment_data/params/b13_U200_k2_d4_virtual.csv"
# col_name = "muscle_length_Waddling"
# col_val = 0.325
# col_name = "muscle_length_Equinus"
# col_val = 0.73
# col_name = "muscle_force_Footdrop"
# col_val = 0.0
# col_name = "muscle_length_Calcaneal"
# col_val = 0.3
# col_name = "muscle_length_Hyperlordosis"
# col_val = 0.58
# col_name = "device_force_weight"
# col_val = 1.0
col_name = "device_virtual_coupling"
col_val = 1.0

#======================================================================
path = PROJECT_ROOT / CSV_IN
df = pd.read_csv(CSV_IN)
df[col_name] = col_val

df.to_csv(CSV_OUT, index=False)
print(f"Saved to {CSV_OUT}")
