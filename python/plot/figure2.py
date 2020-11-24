from python.plot.util import *
from matplotlib.patches import Patch
import matplotlib.lines as mlines
import polars as pl
import numpy as np
from tqdm import tqdm

################ CONFIG ################
EXP_NAMES = [
'no_exo_meta_ma15_10000_0801_111350+memo_b20_ma15_10000_0801_111350_pf_param_no_mod_state_ma15_gems+_on_0826_224549',
'no_exo_meta_ma15_10000_0801_111421+memo_b20_ma15_10000_0801_111421_pf_param_no_mod_state_ma15_gems+_on_0826_224557',
'no_exo_meta_ma15_10000_0801_205540+memo_b20_ma15_10000_0801_205540_pf_param_no_mod_state_ma15_gems+_on_0826_224606',
'no_exo_meta_ma15_10000_0802_064456+memo_b20_ma15_10000_0802_064456_pf_param_no_mod_state_ma15_gems+_on_0826_224615',
'no_exo_meta_ma15_10000_0802_163808+memo_b20_ma15_10000_0802_163808_pf_param_no_mod_state_ma15_gems+_on_0826_224624',
'no_exo_meta_ma15_10000_0805_215352+memo_b20_ma15_10000_0805_215352_pf_param_no_mod_state_ma15_gems+_on_0826_224633',
'no_exo_meta_ma15_10000_0805_215354+memo_b20_ma15_10000_0805_215354_pf_param_no_mod_state_ma15_gems+_on_0826_224641',
'no_exo_meta_ma15_10000_0805_215359+memo_b20_ma15_10000_0805_215359_pf_param_no_mod_state_ma15_gems+_on_0826_224649',
'no_exo_meta_ma15_10000_0806_131413+memo_b20_ma15_10000_0806_131413_pf_param_no_mod_state_ma15_gems+_on_0826_224658',
'no_exo_meta_ma15_10000_0806_131416+memo_b20_ma15_10000_0806_131416_pf_param_no_mod_state_ma15_gems+_on_0826_224707',
'no_exo_meta_ma15_10000_0806_131419+memo_b20_ma15_10000_0806_131419_pf_param_no_mod_state_ma15_gems+_on_0826_224716',

# 'no_exo_meta_ma15_10000_0801_111350+memo_pws_U100_v2_no_mod_state_ma15_gems+_on_0826_111007',
# 'no_exo_meta_ma15_10000_0801_111421+memo_pws_U100_v2_no_mod_state_ma15_gems+_on_0826_111448',
# 'no_exo_meta_ma15_10000_0801_205540+memo_pws_U100_v2_no_mod_state_ma15_gems+_on_0826_111542',
# 'no_exo_meta_ma15_10000_0802_064456+memo_pws_U100_v2_no_mod_state_ma15_gems+_on_0826_111639',
# 'no_exo_meta_ma15_10000_0802_163808+memo_pws_U100_v2_no_mod_state_ma15_gems+_on_0826_111735',
# 'no_exo_meta_ma15_10000_0805_215352+memo_pws_U100_v2_no_mod_state_ma15_gems+_on_0826_111831',
# 'no_exo_meta_ma15_10000_0805_215354+memo_pws_U100_v2_no_mod_state_ma15_gems+_on_0826_111927',
# 'no_exo_meta_ma15_10000_0805_215359+memo_pws_U100_v2_no_mod_state_ma15_gems+_on_0826_112026',
# 'no_exo_meta_ma15_10000_0806_131413+memo_pws_U100_v2_no_mod_state_ma15_gems+_on_0826_112121',
# 'no_exo_meta_ma15_10000_0806_131416+memo_pws_U100_v2_no_mod_state_ma15_gems+_on_0826_112217',
# 'no_exo_meta_ma15_10000_0806_131419+memo_pws_U100_v2_no_mod_state_ma15_gems+_on_0826_112314',
]
########################################

TITLES = [
    "Ankle dorsiflexion (°)",
    "Knee flexion (°)",
    "Hip flexion (°)",
    "Hip abduction (°)",
    "Hip internal rotation (°)",
]


def plot_real_data(axes):
    """
    Load and plot reference data from parquet file for each joint angle.
    Maps TITLES to corresponding joint names in the parquet data.
    """
    # Joint name mapping from TITLES to parquet joint names
    joint_mapping = {
        "Hip flexion (°)": "Hip_Flexion",
        "Hip internal rotation (°)": "Hip_Int_Rot", 
        "Hip abduction (°)": "Hip_Adduction",
        "Knee flexion (°)": "Knee_Flexion",
        "Ankle dorsiflexion (°)": "Ankle_Dorsiflexion",
    }
    
    parquet_path = PROJECT_ROOT / "data/figs/gait_data/averaged_kinematics_data.parquet"
    
    try:
        # Load the parquet data
        ref_data = pl.read_parquet(parquet_path)
        
        for i, title in enumerate(TITLES):
            ax = axes[i]
            joint_name = joint_mapping.get(title)
            
            if joint_name is None:
                print(f"Warning: No mapping found for joint '{title}'")
                continue
                
            # Filter for specific joint and LevelWalking task
            joint_data = ref_data.filter(
                (pl.col("joint") == joint_name) & 
                (pl.col("task") == "LevelWalking")
            )
            
            if not joint_data.is_empty():
                time_vals = joint_data["time_percent"].to_numpy()
                mean_vals = joint_data["mean_angle"].to_numpy()
                std_vals = joint_data["std_angle"].to_numpy()
                
                # Plot mean line
                ax.plot(time_vals, mean_vals, 
                       '-', color='gray', linewidth=2, alpha=0.9, label='Human reference')
                
                # Add std deviation band
                ax.fill_between(time_vals, mean_vals - std_vals, mean_vals + std_vals,
                                color='gray', alpha=0.2)
                                
            ax.set_title(title, fontsize=FONT_SIZE_LABEL)
            ax.grid(True, alpha=0.3)
            
    except Exception as e:
        print(f"Warning: Could not load reference data: {e}")
        for ax in axes:
            ax.set_ylabel("Angle (°)")
            ax.grid(True, alpha=0.3)

    
def plot_sim_data(axes):
    """
    Read angle data from EXP_NAMES experiments with multiple parameters.
    1. Iterate through all experiments and param/cycle combinations to normalize xdata and interpolate for gait cycle (0, 1, ...100)
    2. Compute mean and std across all normalized trajectories from all experiments
    3. Plot simulation results
    """
    # Mapping from TITLES to simulation angle column names
    sim_joint_mapping = {
        "Hip flexion (°)": "angle_HipR",
        "Hip internal rotation (°)": "angle_HipIRR", 
        "Hip abduction (°)": "angle_HipAbR",
        "Knee flexion (°)": "angle_KneeR",
        "Ankle dorsiflexion (°)": "angle_AnkleR",
    }
    
    if not EXP_NAMES:
        print("Warning: No experiments specified in EXP_NAMES")
        return
    
    try:
        # Load simulation data using existing utilities
        from python.surrogate.util import data_and_param_path
        from scipy import interpolate
        
        # Target gait cycle percentages (0 to 100)
        target_gait_cycle = np.arange(0, 101)  # 0, 1, 2, ..., 100
        
        for i, title in enumerate(TITLES):
            ax = axes[i]
            sim_col = sim_joint_mapping.get(title)
            
            if sim_col is None:
                print(f"Warning: No simulation column mapping found for '{title}'")
                continue
                
            # Collect all normalized trajectories across all experiments
            all_normalized_trajectories = []
            
            # Get total parameter count for progress bar (first experiment param count * num experiments)
            total_params = 0
            if EXP_NAMES:
                try:
                    _, first_param = data_and_param_path(EXP_NAMES[0], read_file=True)
                    first_param_count = len(first_param.collect()["param_idx"].unique())
                    total_params = first_param_count * len(EXP_NAMES)
                except:
                    total_params = len(EXP_NAMES) * 100  # fallback estimate
            
            # Create progress bar for all parameters across all experiments
            with tqdm(total=total_params, desc=f"Processing {title}", ncols=100) as pbar:
                # Iterate through all experiments
                for exp_name in EXP_NAMES:
                    try:
                        sim_data, sim_param = data_and_param_path(exp_name, read_file=True)
                        
                        # Check if the column exists in this experiment
                        if sim_col not in sim_data.collect_schema().names():
                            print(f"Warning: Column '{sim_col}' not found in experiment {exp_name}")
                            continue
                        
                        # Filter out prerun and last cycle (following preprocessing patterns)
                        sim_data = (sim_data
                                   .filter(pl.col("cycle") >= 5)  # Remove first 5 cycles
                                   .filter(pl.col("cycle") < pl.max("cycle")))  # Remove last cycle
                        
                        # Get unique parameter indices
                        param_indices = sim_param.collect()["param_idx"].unique().sort()
                        
                        # 1. Iterate through param/cycle combinations to normalize xdata
                        for param_idx in param_indices:
                            # Update progress bar for each parameter
                            pbar.update(1)
                            
                            # Get all cycles for this parameter
                            param_data = (sim_data
                                         .filter(pl.col("param_idx") == param_idx)
                                         .sort(["cycle", "step"])
                                         .collect())
                            
                            if param_data.is_empty():
                                continue
                            
                            # Process each cycle separately
                            cycles = param_data["cycle"].unique().sort()
                            
                            for cycle_num in cycles:
                                cycle_data = param_data.filter(pl.col("cycle") == cycle_num)
                                
                                if len(cycle_data) < 10:  # Skip cycles with too few data points
                                    continue
                                
                                # Get angle data for this cycle
                                angles = cycle_data[sim_col].to_numpy()
                                
                                # Create normalized time axis (0 to 100% of gait cycle)
                                original_time = np.linspace(0, 100, len(angles))
                                
                                # Interpolate to standard gait cycle points (0, 1, 2, ..., 100)
                                if len(angles) > 1:  # Need at least 2 points to interpolate
                                    try:
                                        f = interpolate.interp1d(original_time, angles, 
                                                               kind='linear', bounds_error=False, 
                                                               fill_value='extrapolate')
                                        normalized_angles = f(target_gait_cycle)
                                        all_normalized_trajectories.append(normalized_angles)
                                    except Exception as e:
                                        print(f"Warning: Interpolation failed for {exp_name}, param {param_idx}, cycle {cycle_num}: {e}")
                                        continue
                        
                    except Exception as e:
                        print(f"Error processing experiment {exp_name}: {e}")
                        continue
            
            # 2. Compute mean and std across all normalized trajectories from all experiments
            if all_normalized_trajectories:
                # Convert to numpy array: shape (n_trajectories, 101)
                trajectory_array = np.array(all_normalized_trajectories)
                
                # Compute mean and std across trajectories (axis=0)
                mean_angles = np.mean(trajectory_array, axis=0)
                std_angles = np.std(trajectory_array, axis=0)
                
                # 3. Plot simulation results
                ax.plot(target_gait_cycle, mean_angles, 
                       '-', color='red', linewidth=2, alpha=0.8, label='Simulation')
                
                # Add std deviation band
                ax.fill_between(target_gait_cycle, mean_angles - std_angles, mean_angles + std_angles,
                                color='red', alpha=0.2)
                                
                print(f"Processed {len(all_normalized_trajectories)} trajectories from {len(EXP_NAMES)} experiments for {title}")
            else:
                print(f"Warning: No valid trajectories found for '{title}' across all experiments")
                
    except Exception as e:
        print(f"Error in plot_sim_data: {e}")
        import traceback
        traceback.print_exc()
        
    # Add legend to first subplot
    if len(axes) > 0:
        sim_line = mlines.Line2D([], [], color='red', linewidth=2)
        sim_patch = Patch(facecolor='red', edgecolor='red', alpha=0.2)
        human_line = mlines.Line2D([], [], color='gray', linewidth=2)
        human_patch = Patch(facecolor='gray', edgecolor='gray', alpha=0.2)
        
        # Create combined legend entries
        legend_elements = [
            (sim_line, sim_patch),
            (human_line, human_patch)
        ]
        labels = ['Simulation', 'Human exp.']
        # labels = [r'Simulation ($\pm SD$)', r'Human ref. ($\pm SD$)']
        lg = axes[-1].legend(legend_elements, labels, loc='upper right', fontsize=FONT_SIZE_LABEL-4,)
        lg.get_frame().set_alpha(0.3)


def main():
    fig, axes = plt.subplots(1, len(TITLES), figsize=(4 * len(TITLES), 4), constrained_layout=True)
    for ax in axes:
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    fig.supxlabel("Gait cycle (%)", fontsize=FONT_SIZE_LABEL)
    
    plot_real_data(axes)
    plot_sim_data(axes)
    
    plt.savefig(f"plot/figure2.png", dpi=100, transparent=True)
    print("Plot saved to plot/figure2.png")


if __name__ == "__main__":
    main()
