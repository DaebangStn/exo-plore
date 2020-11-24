# Sample Gait parameter for given velocity as a product constant. vel / REF_VEL = Phase * Stride
import numpy as np

from python.sample.util import *

#============================Configurations============================
CSV_PATH = "experiment_data/params/P100.csv"
NUM_SAMPLE = 100


#=============================Parameters==============================
Phase_range = [0.45, 1.3]
Stride_range = [0.45, 1.3]
Velocity_kmh = np.arange(1.0, 5.1, 1.0)
SAMPLE_PER_VEL = NUM_SAMPLE // len(Velocity_kmh)

Product_const = Velocity_kmh / (3.6 * REF_VELOCITY)
initial_sigma = 0.1
#======================================================================

def sample_gait_normal(phase_mu: float, product_const: float, _sigma: float, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
    phase_sampled = np.random.normal(phase_mu, _sigma, num_sample)
    phase_sampled = np.where(np.isclose(phase_sampled, 0, atol=0.01), 1e3, phase_sampled)
    stride_sampled = product_const / phase_sampled
    phase_valid_flag = (min(Phase_range) <= phase_sampled) & (phase_sampled <= max(Phase_range))
    stride_valid_flag = (min(Stride_range) <= stride_sampled) & (stride_sampled <= max(Stride_range))
    valid = phase_valid_flag & stride_valid_flag
    return phase_sampled[valid], stride_sampled[valid]


# Find the cross point of the hyperbola and the line
slope, intercept, _, _, _ = linregress(Phase_range, Stride_range)
points = [find_line_cross_xy_hyperbola(slope, intercept, constraint) for constraint in Product_const]
Phase_found, Stride_found = zip(*points)

params = []  # List[Dict[str, float]], ["Phase", "Stride"] in Dict.keys()

for idx, const in enumerate(Product_const):
    phase_cross = Phase_found[idx]
    stride_cross = Stride_found[idx]
    params.append({
        "Phase": phase_cross,
        "Stride": stride_cross
    })

    phase_sampled = np.array([])
    stride_sampled = np.array([])
    sigma = initial_sigma
    while len(phase_sampled) < SAMPLE_PER_VEL / 2:
        print(f"Sampling Velocity: {Velocity_kmh[idx]} km/h, sigma: {sigma}")
        phase_sampled, stride_sampled = sample_gait_normal(phase_cross, const, sigma, SAMPLE_PER_VEL)
        sigma *= 0.7

    print(f"valid samples: {len(phase_sampled)}")

    for phase, stride in zip(phase_sampled, stride_sampled):
        params.append({
            "Phase": phase,
            "Stride": stride
        })

print(f"Total samples: {len(params)}")
df = pd.DataFrame(params)
df["param_idx"] = np.arange(len(df))

df.to_csv(CSV_PATH, index=False, sep=',', float_format='%.4f')
print(f"Product-constrained sampled parameters saved to {CSV_PATH}")
