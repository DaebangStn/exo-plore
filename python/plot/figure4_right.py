import numpy as np
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from python.surrogate.preprocess import load_shift_processed, filter_lastrun, filter_prerun_w_cycle
from python.surrogate.optimizer import Optimizer
from python.surrogate.batched_optimizer import Optimizer as BOptimizer
from python.surrogate.util import tensor_to_df
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import matplotlib.lines as mlines
from python.plot.util import *
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from typing import List, Tuple
import polars as pl
import torch
from tqdm import tqdm

################ CONFIG ################
EXP_NAMEs = [
# [
# 'no_exo_meta_ma_0620_135347_10000+memo__b20_U500_k0+_on_0731_123329',
# 'no_exo_meta_ma_0622_124343_10000+memo__b20_U500_k0+_on_0731_123646',
# 'no_exo_meta_ma_0625_131439_10000+memo__b20_U500_k0+_on_0731_123014',
# 'no_exo_meta_ma_0629_113826_10000+memo__b20_U500_k0+_on_0731_123954',
# 'no_exo_meta_ma_0705_120847_10000+memo__b20_U500_k0+_on_0731_124306',
# 'no_exo_meta_ma_10000_0729_211733+memo_b20_U500_k0_no_mod_state_ma_on_0801_094603',
# 'no_exo_meta_ma_10000_0730_064237+memo_b20_U500_k0_no_mod_state_ma_on_0801_095000',
# 'no_exo_meta_ma_10000_0731_015140+memo_b20_U500_k0_no_mod_state_ma_on_0801_095754',
# 'no_exo_meta_ma_10000_0730_161202+memo_b20_U500_k0_no_mod_state_ma_on_0801_095409',
# ],
# [
# 'no_exo_meta_ma3_0620_174352_10000+memo__b20_U500_k0+_on_0731_125031',
# 'no_exo_meta_ma3_0622_125627_10000+memo__b20_U500_k0+_on_0731_124044',
# 'no_exo_meta_ma3_0624_141640_10000+memo__b20_U500_k0+_on_0731_123730',
# 'no_exo_meta_ma3_0626_124157_10000+memo__b20_U500_k0+_on_0731_124359',
# 'no_exo_meta_ma3_0627_103431_10000+memo__b20_U500_k0+_on_0731_123045',
# 'no_exo_meta_ma3_0627_103440_10000+memo__b20_U500_k0+_on_0731_123415',
# 'no_exo_meta_ma3_0628_093859_10000+memo__b20_U500_k0+_on_0731_124712',
# 'no_exo_meta_ma3_0629_095847_10000+memo__b20_U500_k0+_on_0731_125353',
# ],
[
'no_exo_meta_ma15_10000_0801_111350+memo_b20_U500_k0_no_mod_state_ma15_gems+_on_0822_111223',
'no_exo_meta_ma15_10000_0801_111421+memo_b20_U500_k0_no_mod_state_ma15_gems+_on_0822_111612',
'no_exo_meta_ma15_10000_0801_205540+memo_b20_U500_k0_no_mod_state_ma15_gems+_on_0822_112002',
'no_exo_meta_ma15_10000_0802_064456+memo_b20_U500_k0_no_mod_state_ma15_gems+_on_0822_112417',
'no_exo_meta_ma15_10000_0802_163808+memo_b20_U500_k0_no_mod_state_ma15_gems+_on_0822_112806',
'no_exo_meta_ma15_10000_0805_215352+memo_b20_U500_k0_no_mod_state_ma15_gems+_on_0822_113210',
'no_exo_meta_ma15_10000_0805_215354+memo_b20_U500_k0_no_mod_state_ma15_gems+_on_0822_113601',
'no_exo_meta_ma15_10000_0805_215359+memo_b20_U500_k0_no_mod_state_ma15_gems+_on_0822_114003',
'no_exo_meta_ma15_10000_0806_131413+memo_b20_U500_k0_no_mod_state_ma15_gems+_on_0822_114415',
'no_exo_meta_ma15_10000_0806_131416+memo_b20_U500_k0_no_mod_state_ma15_gems+_on_0822_114818',
'no_exo_meta_ma15_10000_0806_131419+memo_b20_U500_k0_no_mod_state_ma15_gems+_on_0822_115222',
],
]

LENGEND_NAMES = [
# r"$\mathbf{mEE} = ma^{1.0}$",
# r"$\mathbf{mEE} = ma^{3.0}$",
# r"$\mathbf{mEE} = ma^{1.5}$",
"Simulation",
]
COLORS = [
    # 'blue',
    # 'green',
    'red',
]

COT_FIELDS = [
    # "cot_ma", 
    # "cot_ma3", 
    "cot_ma15"
]

# Optimizer configuration
VELOCITIES_KMH = np.linspace(1.6, 7.7, 100)  # 0.1 km/h intervals
VELOCITIES_MS = VELOCITIES_KMH / 3.6
OPTIMIZER_TRIALS = 20

# Batched optimizer configuration
BATCH_CONFIG = {
    "MAX_ITER": 250,
    # "MAX_ITER": 1,
    "TRIAL_SIZE": 128,
    "LR": 5e-2,
    "CONST_REG": 3.0,
    "CHERRY_PICK": False,
}
REGULARIZE = {
    "Phase": 0.0,
    "Stride": 0.0,
}

USED_CYCLE_IDX = 7
POLYNOMIAL_DEGREE = 5  # 5th order polynomial for smoothing
# Using polynomial smoothing with std bands
########################################

REFRESH_BOUNDARY = True
SELECTION_INTERVALS_NUM = 20

real_data_num = 5
capsize = 5

xticks = np.arange(0, 8.1, 2.0)
stepRange = [0.2, 1.0]
cadenceRange = [0.5, 3.0]
stepTick = np.arange(stepRange[0], stepRange[1] + 0.1, 0.4)
cadenceTick = np.arange(cadenceRange[0], cadenceRange[1] + 0.1, 1.0)
xrange = [1, 8]
real_to_sim_field = {
    "Step": "step_length",
    "Cadence": "cadence",
}

sim_data_fields = ["angle_HipR"]
cad_step_fields = ["Step", "Cadence"]

real_color = "k"
selected_sim_color = "r"
raw_sim_color = "gray"
fontsize = 16
label_padsize = 8
########################################


def optimize_gait_params_batched(exp_name: str, cot_field: str) -> pl.DataFrame:
    """
    Use batched optimizer with velocity constraints to find optimal gait parameters

    Args:
        exp_name: Experiment name
        cot_field: CoT field to optimize (e.g., 'cot_ma', 'cot_ma3', 'cot_ma15')

    Returns:
        DataFrame with optimized gait parameters for each velocity
    """
    print(f"Optimizing gait parameters for {exp_name} using {cot_field} (batched)")

    # Initialize batched optimizer
    ckpt_path = opt_ckpt_versioned_path(exp_name)
    optimizer = BOptimizer(ckpt_path, reg_input=REGULARIZE, reg_batch=0.0)

    # Check required fields are in model
    assert "velocity" in optimizer.model.output_col, f"Model missing 'velocity' output field"
    assert cot_field in optimizer.model.output_col, f"Model missing '{cot_field}' output field"
    assert "Phase" in optimizer.model.input_col, f"Model missing 'Phase' input field"
    assert "Stride" in optimizer.model.input_col, f"Model missing 'Stride' input field"

    # Prepare velocity constraints (raw values)
    # Pass raw velocity values - the optimizer will handle normalization internally
    velocity_tensor = torch.tensor(VELOCITIES_MS, dtype=torch.float32, device=optimizer.model.device)
    velocity_tensor = optimizer.model.normalize_tensor(velocity_tensor, ["velocity"])

    print(f"Running batched optimization for {len(VELOCITIES_MS)} velocities...")

    # Run constrained optimization
    found_param_list = optimizer.run_with_constraint(
        opt_field=cot_field,
        const_field="velocity",
        const_values=velocity_tensor,
        const_reg=BATCH_CONFIG["CONST_REG"],
        lr=BATCH_CONFIG["LR"],
        trial_size=BATCH_CONFIG["TRIAL_SIZE"],
        max_iter=BATCH_CONFIG["MAX_ITER"],
        cherry_pick=BATCH_CONFIG["CHERRY_PICK"]
    )

    if not found_param_list:
        print(f"Error: No successful optimizations for {exp_name}")
        return pl.DataFrame()

    # Debug: Check the structure of found_param_list
    print(f"Found {len(found_param_list)} optimization results")
    if found_param_list:
        print(f"First result keys: {list(found_param_list[0].keys())}")
        print(f"First result types: {[type(v) for v in found_param_list[0].values()]}")

        # Convert all values to Python floats to avoid datatype issues
        cleaned_param_list = []
        for result in found_param_list:
            cleaned_result = {}
            for key, value in result.items():
                if hasattr(value, 'item'):  # numpy scalar
                    cleaned_result[key] = float(value.item())
                else:
                    cleaned_result[key] = float(value)
            cleaned_param_list.append(cleaned_result)
    else:
        cleaned_param_list = found_param_list

    # Convert results to DataFrame
    result_df = pl.DataFrame(cleaned_param_list)

    # Add velocity columns for compatibility (velocity is already in m/s from the constraint)
    result_df = result_df.with_columns([
        pl.col("velocity").alias("velocity_ms"),
        (pl.col("velocity") * 3.6).alias("velocity_kmh")
    ])

    print(f"Completed batched optimization for {exp_name}: {len(result_df)} results")
    return result_df


def optimize_gait_params(exp_name: str, cot_field: str) -> pl.DataFrame:
    """
    Wrapper function that tries batched optimizer first, falls back to single optimizer
    """
    # try:
    return optimize_gait_params_batched(exp_name, cot_field)
    # except Exception as e:
    #     print(f"Batched optimization failed: {e}")
    #     print(f"Falling back to single-trial optimization...")
    #     return optimize_gait_params_single(exp_name, cot_field)


def optimize_gait_params_single(exp_name: str, cot_field: str) -> pl.DataFrame:
    """
    Use single-trial optimizer to find optimal gait parameters for various velocities (fallback)

    Args:
        exp_name: Experiment name
        cot_field: CoT field to optimize (e.g., 'cot_ma', 'cot_ma3', 'cot_ma15')

    Returns:
        DataFrame with optimized gait parameters for each velocity
    """
    print(f"Optimizing gait parameters for {exp_name} using {cot_field} (single)")

    # Initialize optimizer
    ckpt_path = opt_ckpt_versioned_path(exp_name)
    optimizer = Optimizer(ckpt_path, trials=OPTIMIZER_TRIALS)

    # Optimize for each velocity
    found_param_list = []
    for velocity in tqdm(VELOCITIES_MS, desc=f"Optimizing:", ncols=80):
        try:
            param, _ = optimizer.run_w_cfield_from_model(cot_field, "velocity", velocity)
            param_df = pl.DataFrame([param])
            param_df = param_df.with_columns(pl.lit(velocity).alias("velocity_ms"))
            param_df = param_df.with_columns(pl.lit(velocity * 3.6).alias("velocity"))
            found_param_list.append(param_df)
        except Exception as e:
            print(f"Warning: Optimization failed for velocity {velocity:.2f} m/s: {e}")
            continue

    if not found_param_list:
        print(f"Error: No successful optimizations for {exp_name}")
        return pl.DataFrame()

    # Combine all results
    found_df = pl.concat(found_param_list)

    # Denormalize the parameters
    found_df_norm = optimizer.model.normalize(found_df)
    found_tensor = torch.tensor(
        found_df_norm[optimizer.model.input_col].to_numpy(),
        dtype=torch.float32
    ).to(optimizer.model.device)

    # Get predictions from model
    with torch.no_grad():
        predict_output = tensor_to_df(optimizer.model(found_tensor), optimizer.model.output_col)
    predict_output = optimizer.model.denormalize(predict_output)

    # Combine parameters with predictions
    result_df = found_df.hstack(predict_output)

    print(f"Completed optimization for {exp_name}: {len(result_df)} successful optimizations")
    return result_df


def select_gait_param(exp_name: str, cot_field: str) -> Tuple[Df, Df]:
    """
    Legacy function - now redirects to optimizer-based approach
    """
    # Use optimizer to get gait parameters
    optimized_df = optimize_gait_params(exp_name, cot_field)

    # Convert to format expected by plotting functions
    # Extract relevant columns for gait parameters
    gait_params = optimized_df.select([
        "velocity",
        pl.col("Phase").alias("cadence"),  # Phase maps to cadence
        pl.col("Stride").alias("step_length")  # Stride maps to step_length
    ])

    # Create empty sim_data as it's not used in the new approach
    sim_data = pl.DataFrame()

    return gait_params, sim_data



def polynomial_smooth_with_std_bands(x_data, y_data, y_std_data, degree=5):
    """
    Smooth data with polynomial regression and compute std bands for each x value

    Args:
        x_data: numpy array of x values (velocities)
        y_data: numpy array of mean y values
        y_std_data: numpy array of std y values
        degree: polynomial degree for smoothing

    Returns:
        x_smooth: smoothed x values
        y_smooth: smoothed y values (polynomial fit)
        y_std_smooth: interpolated std values at smooth x positions
    """
    # Sort data for proper processing
    sorted_indices = np.argsort(x_data)
    x_sorted = x_data[sorted_indices]
    y_sorted = y_data[sorted_indices]
    y_std_sorted = y_std_data[sorted_indices]

    # Create polynomial features and fit
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(x_sorted.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_poly, y_sorted)

    # Create smooth prediction grid
    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
    X_smooth_poly = poly_features.transform(x_smooth.reshape(-1, 1))
    y_smooth = model.predict(X_smooth_poly)

    # Interpolate standard deviation values at smooth x positions
    y_std_smooth = np.interp(x_smooth, x_sorted, y_std_sorted)

    return x_smooth, y_smooth, y_std_smooth


def plot_gait_parameter(axes, grouped_optimized_data: List[pl.DataFrame], category_idx: int, color: str):
    """
    Plot gait parameters from optimized data with averaging across experiments
    """
    print(f"Plotting category {category_idx} with {len(grouped_optimized_data)} experiments")

    # Average across experiments for each velocity
    # First, align all data to the same velocity grid
    velocity_grid_kmh = VELOCITIES_KMH
    averaged_params = {"velocity": velocity_grid_kmh}

    # Initialize parameter storage
    for field in ["Phase", "Stride"]:
        averaged_params[field] = []
        averaged_params[field + "_std"] = []  # Store standard deviations

    # For each velocity in our grid, find corresponding values from all experiments
    for target_vel_kmh in velocity_grid_kmh:
        target_vel_ms = target_vel_kmh / 3.6
        phase_values = []
        stride_values = []

        # Collect values from all experiments for this velocity
        for exp_df in grouped_optimized_data:
            if len(exp_df) == 0:
                continue

            # Find closest velocity in this experiment
            # Handle both velocity_ms and velocity columns
            if "velocity_ms" in exp_df.columns:
                velocities = exp_df.select("velocity_ms").to_numpy().flatten()
            elif "velocity" in exp_df.columns:
                velocities = exp_df.select("velocity").to_numpy().flatten() / 3.6  # Convert km/h to m/s
            else:
                print(f"Warning: No velocity column found in experiment data")
                continue

            if len(velocities) == 0:
                continue

            closest_idx = np.argmin(np.abs(velocities - target_vel_ms))
            closest_vel = velocities[closest_idx]

            # Only use if velocity is reasonably close (within 0.05 m/s)
            if abs(closest_vel - target_vel_ms) <= 0.05:
                try:
                    phase_val = exp_df.select("Phase").to_numpy()[closest_idx]
                    stride_val = exp_df.select("Stride").to_numpy()[closest_idx]
                    phase_values.append(phase_val)
                    stride_values.append(stride_val)
                except Exception as e:
                    print(f"Warning: Failed to extract parameter values: {e}")
                    continue

        # Average the values and compute standard deviation for this velocity
        if phase_values and stride_values:
            averaged_params["Phase"].append(np.mean(phase_values))
            averaged_params["Stride"].append(np.mean(stride_values))
            # Compute standard deviation, use 0 if only one value
            averaged_params["Phase_std"].append(np.std(phase_values) if len(phase_values) > 1 else 0.0)
            averaged_params["Stride_std"].append(np.std(stride_values) if len(stride_values) > 1 else 0.0)
        else:
            # Use NaN for missing data
            averaged_params["Phase"].append(np.nan)
            averaged_params["Stride"].append(np.nan)
            averaged_params["Phase_std"].append(np.nan)
            averaged_params["Stride_std"].append(np.nan)

    # Convert to numpy arrays and remove NaN values
    velocity_array = np.array(averaged_params["velocity"])
    phase_array = np.array(averaged_params["Phase"])
    stride_array = np.array(averaged_params["Stride"])
    phase_std_array = np.array(averaged_params["Phase_std"])
    stride_std_array = np.array(averaged_params["Stride_std"])

    # Remove NaN values
    valid_mask = ~(np.isnan(phase_array) | np.isnan(stride_array) | np.isnan(phase_std_array) | np.isnan(stride_std_array))
    velocity_valid = velocity_array[valid_mask]
    phase_valid = phase_array[valid_mask]
    stride_valid = stride_array[valid_mask]
    phase_std_valid = phase_std_array[valid_mask]
    stride_std_valid = stride_std_array[valid_mask]

    print(f"Category {category_idx}: {len(velocity_valid)} valid velocity points for averaging")

    real_data = load_wpd_dataset("cot")
    ax_idx = 0
    for field in cad_step_fields:
        ax = axes[ax_idx]
        ax_idx += 1

        # Configure axes
        ax.set_xticks(xticks)
        ax.set_xlim(xrange[0], xrange[1])
        if field == "Step":
            ax.set_yticks(stepTick)
            ax.set_ylim(min(stepRange), max(stepRange))
            if category_idx == 0:  # Only set ylabel for first category
                ax.set_ylabel("Step length (m)", fontsize=FONT_SIZE_LABEL, labelpad=label_padsize)
        elif field == "Cadence":
            ax.set_yticks(cadenceTick)
            ax.set_ylim(min(cadenceRange), max(cadenceRange))
            if category_idx == 0:  # Only set ylabel for first category
                ax.set_ylabel("Step frequency (Hz)", fontsize=FONT_SIZE_LABEL, labelpad=label_padsize)

        # Plot real data only once (for first category)
        if category_idx == 0:
            for j in range(real_data_num):
                xerr = real_data[f'{field}{j+1}_X'][1] - real_data[f'{field}{j+1}_X'][0]
                yerr = real_data[f'{field}{j+1}_Y'][2] - real_data[f'{field}{j+1}_Y'][0]
                ax.errorbar(real_data[f'{field}{j+1}_X'][0] * 3.6, real_data[f'{field}{j+1}_Y'][0],
                           xerr=xerr, yerr=yerr, color=real_color, capsize=capsize)

        # Get averaged data and standard deviation for this field
        if field == "Step":
            # Convert stride (normalized) to step length (meters)
            ydata_averaged = stride_valid * REF_STEP  # Convert to physical units
            ydata_std = stride_std_valid * REF_STEP  # Convert std to physical units
        elif field == "Cadence":
            # Convert phase (normalized) to cadence (Hz)
            ydata_averaged = phase_valid * REF_CADENCE  # Convert to physical units
            ydata_std = phase_std_valid * REF_CADENCE  # Convert std to physical units

        if len(velocity_valid) > 3:  # Need enough points for polynomial fitting
            # Apply polynomial smoothing with std bands
            x_smooth, y_smooth, y_std_smooth = polynomial_smooth_with_std_bands(
                velocity_valid, ydata_averaged, ydata_std, degree=POLYNOMIAL_DEGREE)

            # Plot smoothed polynomial line with category-specific color
            ax.plot(x_smooth, y_smooth, color=color, linewidth=2,
                   label=LENGEND_NAMES[category_idx] if ax_idx == 1 else "")

            # Plot standard deviation bands using interpolated std values
            y_lower = y_smooth - y_std_smooth
            y_upper = y_smooth + y_std_smooth
            ax.fill_between(x_smooth, y_lower, y_upper, alpha=0.2, color=color)
        else:
            print(f"Warning: Not enough valid points for polynomial fitting in category {category_idx}, field {field}")

        print(f"Smoothed optimization results with std bands plotted for {field}, category {category_idx}")
    
    
def plot_hip_angle(ax, sim_data: Df, cad_step_data: Df):
    xdata = cad_step_data.select("velocity").to_numpy()
    # Limit velocity range
    vel_min, vel_max = 2.0, 7.0  # Set desired min/max velocity in km/h
    
    # Filter data within velocity range
    valid_velocities = (xdata >= vel_min) & (xdata <= vel_max)
    cad_step_data = cad_step_data.filter(pl.col("velocity") >= vel_min).filter(pl.col("velocity") <= vel_max)
    param_indices = np.unique(cad_step_data.select("param_idx").to_numpy())
    
    norm = mcolors.Normalize(vmin=vel_min, vmax=vel_max)
    cmap = cm.get_cmap("viridis")
    
    for field in sim_data_fields:
        for param in param_indices:
            sim_data_df = sim_data.filter(pl.col("param_idx") == param)
            cycle_indices = np.unique(sim_data_df.select("cycle").to_numpy())
            for cycle_idx in cycle_indices:
                velocity = cad_step_data.filter(pl.col("param_idx") == param).select("velocity").item()
                # Only plot if velocity is within range
                if vel_min <= velocity <= vel_max:
                    color = cmap(norm(velocity))
                    sim_data_field = sim_data_df.filter(pl.col("cycle") == cycle_idx).select(field).to_numpy()
                    xdata = np.linspace(0, 100, len(sim_data_field))
                    ax.plot(xdata, sim_data_field, color=color)
                    
        ax.set_xlabel("Gait cycle (%)", fontsize=FONT_SIZE_LABEL + 4, labelpad=label_padsize)
        ax.set_ylabel(f"Simulated hip flexion angle (°)", fontsize=FONT_SIZE_LABEL + 4, labelpad=label_padsize)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.axvline(x=50, color='k', linestyle='--', alpha=0.5)
        
        # Add colorbar with limited range
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Walking speed (km/h)', fontsize=FONT_SIZE_LABEL + 4, labelpad=10)
        cbar.ax.tick_params(labelsize=fontsize-2)


def main():
    # Create vertically stacked subplots (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(6, 9))
    plt.subplots_adjust(hspace=0.1)  # Add vertical spacing between subplots
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.15, right=0.95)

    # Process each category separately using optimizer
    for category_idx, exp_group in enumerate(EXP_NAMEs):
        print(f"\nProcessing category {category_idx}: {LENGEND_NAMES[category_idx]}")

        optimized_data_list = []
        cot_field = COT_FIELDS[category_idx]
        color = COLORS[category_idx]

        # Run optimizer for each experiment in this category
        for exp_name in exp_group:
            print(f"Running optimizer for experiment: {exp_name}")
            try:
                optimized_df = optimize_gait_params(exp_name, cot_field)
                if len(optimized_df) > 0:
                    optimized_data_list.append(optimized_df)
                else:
                    print(f"Warning: No optimization results for {exp_name}")
            except Exception as e:
                print(f"Error optimizing {exp_name}: {e}")
                continue

        if optimized_data_list:
            # Plot this category with averaged optimization results
            plot_gait_parameter(axes, optimized_data_list, category_idx, color)
        else:
            print(f"Warning: No valid optimization data for category {category_idx}")

    # Add legend after plotting all categories
    # Create legend handles
    legend_handles = []
    legend_labels = []

    # Add real data to legend (only once)
    real_patch = mlines.Line2D([], [], color='black', marker='P', linestyle='None', markersize=15, label="Human exp.")
    legend_handles.append(real_patch)
    legend_labels.append("Human exp.")

    # Add simulation categories
    for category_idx, (color, label) in enumerate(zip(COLORS, LENGEND_NAMES)):
        sim_line = mlines.Line2D([], [], color=color, linewidth=2)
        sim_patch = Patch(facecolor=color, edgecolor=color, alpha=0.2)
        legend_handles.append((sim_line, sim_patch))
        legend_labels.append(label)

    lg = axes[0].legend(legend_handles, legend_labels, loc='upper left', fontsize=FONT_SIZE_LABEL)
    lg.get_frame().set_alpha(0.3)


    # Set x-axis label on the bottom subplot for vertical stacking
    axes[1].set_xlabel("Walking speed (km/h)", fontsize=FONT_SIZE_LABEL)

    # plt.savefig(f"plot/gait_param_optimized_single.png", dpi=150, transparent=True)
    plt.savefig(f"plot/figure4_right.png", dpi=150, transparent=True)

if __name__ == "__main__":
    main()
