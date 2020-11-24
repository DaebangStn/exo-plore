# Phase and Stride from digitized plot output
from python.sample.util import *


#============================Configurations============================
GEMS_PAPER = True
K = 0.2666667
Delay = [  # Single element creates fig1+2, multiple elements creates fig3
    0.05, 0.15, 0.35,
    0.25,
]

#=============================Parameters==============================
# step and cadence is from half of the cycle
step_m = [
    0.435904255319149,
    0.546808510638298,
    0.6497340425531916,
    0.7255319148936172,
    0.7877659574468087,
]

cadence_hz = [
    1.1428571428571423,
    1.5394088669950738,
    1.9088669950738915,
    2.2389162561576352,
    2.5640394088669947,
]

ref_step = 0.67
ref_cadence = 2 / 1.1
gems_velocity_kmh = [1, 2, 3, 4, 5]
gems_velocity_ms = [v / 3.6 for v in gems_velocity_kmh]

if GEMS_PAPER:
    if len(Delay) == 1:
        output_path = "experiment_data/params/gems/fig1+2.csv"
    else:
        output_path = "experiment_data/params/gems/fig3.csv"
else:
    output_path = "experiment_data/params/cot/real.csv"

#======================================================================

if GEMS_PAPER:
    slope, intercept, _, _, _ = linregress(step_m, cadence_hz)
    points = [find_line_cross_xy_hyperbola(slope, intercept, vel_ms) for vel_ms in gems_velocity_ms]
    step_found, cadence_found = zip(*points)
    stride_ratio = [s / ref_step for s in step_found]
    phase_ratio = [c / ref_cadence for c in cadence_found]
    gait_condition = zip(stride_ratio, phase_ratio)
    param_combinations = list(product(gait_condition, [K], Delay))
    df = pd.DataFrame({
        "Stride": [p[0][0] for p in param_combinations],
        "Phase": [p[0][1] for p in param_combinations],
        "K": [p[1] for p in param_combinations],
        "Delay": [p[2] for p in param_combinations],
    })
else:
    stride_ratio = [s / ref_step for s in step_m]
    phase_ratio = [c / ref_cadence for c in cadence_hz]
    df = pd.DataFrame({
        "Stride": stride_ratio,
        "Phase": phase_ratio,
    })

df["param_idx"] = np.arange(len(df))
df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
