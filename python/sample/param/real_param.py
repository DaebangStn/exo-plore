from python.sample.util import *

#============================Configurations============================
CSV_PATH = "experiment_data/params/Real50.csv"
NUM_SAMPLE = 5

USE_EXO = True
num_exo_param = 10

TIMESTAMP = False
NUM_DUPLICATE = 1

#=============================Parameters==============================
Cadence_range = [1.143, 2.564]
Step_range = [0.436, 0.788]
K_range = [0.0, 0.5]
Delay_range = [0.0, 0.5]


#======================================================================
Phase_ratio = [cad / REF_CADENCE for cad in Cadence_range]
Stride_ratio = [step / REF_STEP for step in Step_range]

data = {
    "param_idx": np.arange(NUM_SAMPLE),
    "Phase": np.linspace(Phase_ratio[0], Phase_ratio[1], NUM_SAMPLE),
    "Stride": np.linspace(Stride_ratio[0], Stride_ratio[1], NUM_SAMPLE)
}

if USE_EXO:
    data["param_idx"] = np.arange(NUM_SAMPLE * num_exo_param)
    data["Phase"] = np.repeat(data["Phase"], num_exo_param)
    data["Stride"] = np.repeat(data["Stride"], num_exo_param)
    data["K"] = np.random.uniform(*K_range, NUM_SAMPLE * num_exo_param)
    data["Delay"] = np.random.uniform(*Delay_range, NUM_SAMPLE * num_exo_param)

# Create a DataFrame
df = pd.DataFrame(data)

csv_path = Path(CSV_PATH)
if TIMESTAMP:
    csv_filename = csv_path.stem + "_" + timestamp() + ".csv"
    csv_path = csv_path.parent / csv_filename

df.to_csv(csv_path, index=False, sep=',')

