# Optimize exo parameter (K, Delay) for velocity
# Sampled param is some fixed gait paramimage.pngeter (phase and stride) and varying exo parameters (K and Delay)
from python.surrogate.batched_optimizer import Optimizer as BOptimizer
from python.surrogate.preprocess import load_cot_processed, gt_stat_from_exp_name
from python.plot.util import *
import matplotlib.colors as mcolors

################ CONFIG ################
CKPT_RAW = [
    'exo_hei_resist_21000_0807_131406+memo_exo_benefit_supporting_input_varK_no_mod_state_ma15+_on_0908_111054',
]

CKPT_MODEL = [
    "exo_hei_resist_0807_131406_21000+memo__b21df_LHS60000_no_mod_state_ma15+_on_0821_145627",
    "exo_hei_resist_0807_131406_21000+memo__b21df_LHS60000_no_mod_state_ma15+_on_0821_145627",
    'exo_hei_resist_0807_131406_21000+memo__b21_df_U58500_no_mod_state_ma15+_on_0907_204001',
]

# Checkpoint versions and corresponding titles
CKPT_VERSIONS = [
    "nn_v02",
    "nn_v03",
    "nn_v01", 
]

CKPT_TITLES = [
    "Proposed: Huber loss + LHS",
    "Case1: MSE loss",
    "Case2: Uniform sampling",
]

EPOCH = 10000
# EPOCH = 3000

FIX_K = np.arange(0, 0.41, 0.4/20)
Yrange = (0.88, 1.25)

# Fixed gait parameters (Stride and Phase) 
FIX_STRIDE = 1.013
FIX_PHASE = 1.013

# Single fixed delay value
FIX_DELAY = 0.25

# Delay sweep parameters
DELAY_MIN = 0.0
DELAY_MAX = 0.5
DELAY_STEPS = 50
########################################

point_size = 10
PROCESSOR = "metabolic"

X_LBL = "Velocity"
OPTIM_TARGET = "cot_ma15"
INPUT_VAR = ["K", "Delay"]
INPUT_FIX = ["Stride", "Phase"]


def plot_exo_raw(ax, filtered_data, fix_k, average=False, color=None, divider=1.0):
    """Plot raw data adapted for varied gait parameters with velocity sweep"""
    # Find the nearest K value in the data
    available_k_values = filtered_data["K"].unique().to_list()
    nearest_k = min(available_k_values, key=lambda x: abs(x - fix_k))
    
    # Filter data for the nearest K value
    k_filtered_data = filtered_data.filter(pl.col("K") == nearest_k)
    
    if len(k_filtered_data) == 0:
        print(f"No data found for K={nearest_k} (requested {fix_k})")
        return
    
    # Sort by computed velocity (stride * phase * ref_velocity)
    k_filtered_data = k_filtered_data.with_columns(
        (pl.col("Stride") * pl.col("Phase") * REF_VELOCITY * 3.6).alias("velocity")
    ).sort("velocity")
    
    velocities = k_filtered_data["velocity"].to_numpy()
    cots = k_filtered_data[OPTIM_TARGET].to_numpy()
    
    if average:
        ax.axhline(y=1, color='k', linestyle='--', label=f"exo off", linewidth=2)
        return np.mean(cots)
    else:
        ax.plot(velocities, cots / divider, '-', color=color, linewidth=2)


def plot_exo_model(ax, optimizer, fix_k, average=False, color=None, divider=1.0):
    """Plot model predictions vs velocity with fixed K and delay"""
    model = optimizer.model
    
    # Get velocity range from real gait data
    real_gait_selection = load_gems_selection()
    
    # Create interpolated velocity range
    interpolated_cadence = np.linspace(min(real_gait_selection["Cadence"]), max(real_gait_selection["Cadence"]), 100)
    interpolated_step = np.linspace(min(real_gait_selection["Step"]), max(real_gait_selection["Step"]), 100)
    
    # Create input tensor for velocity sweep
    input_tensor = torch.full((100, len(model.input_col)), float('nan'), dtype=torch.float32, device=model.device)
    input_tensor[:, model.input_col.index("Phase")] = torch.from_numpy(interpolated_cadence / REF_CADENCE)
    input_tensor[:, model.input_col.index("Stride")] = torch.from_numpy(interpolated_step / REF_STEP)
    input_tensor[:, model.input_col.index("K")] = fix_k
    input_tensor[:, model.input_col.index("Delay")] = FIX_DELAY
    
    data_dict = {col: input_tensor.cpu().numpy()[:, i] for i, col in enumerate(model.input_col)}
    df = pl.DataFrame(data_dict)
    df = df.with_row_index("param_idx")
    
    # Compute exo CoT
    base_param = model.normalize_tensor(input_tensor.clone(), model.input_col)
    with torch.no_grad():
        exo_cot = model(base_param)
    exo_cot = model.denormalize_tensor(exo_cot, model.output_col)
    exo_cot = exo_cot.cpu().numpy().squeeze()
    
    # Compute velocities
    base_velocities = input_tensor[:, model.input_col.index("Stride")] * input_tensor[:, model.input_col.index("Phase")] * REF_VELOCITY * 3.6
    base_velocities = base_velocities.cpu().numpy().squeeze()

    if average:
        exo_off_cot = exo_cot.mean(axis=0)
        ax.axhline(y=1, color='k', linestyle='--', label=f"exo off", linewidth=2)
        return exo_off_cot
    else:
        ax.plot(base_velocities, exo_cot / divider, '-', color=color, linewidth=2)



def main():
    assert len(CKPT_MODEL) == 3, "Need exactly three checkpoints for comparison"
    assert len(CKPT_VERSIONS) == 3, "Need exactly three versions for comparison"
    assert len(CKPT_TITLES) == 3, "Need exactly three titles for comparison"
    assert len(CKPT_RAW) == 1, "Need exactly one raw data checkpoint"

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.subplots_adjust(
        left=0.05,
        right=1.1,  # Space for colorbar
        top=0.9,
        bottom=0.15,
        wspace=0.25,
    )
    ax_raw = axes[0]
    ax_model1 = axes[1]
    ax_model2 = axes[2] 
    ax_model3 = axes[3]
    
    # Setup colormap
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=len(FIX_K)-1)
    
    # Load and plot raw data
    ckpt_raw = CKPT_RAW[0]
    try:
        print(f"Loading raw data from: {ckpt_raw}")
        data = load_cot_processed(ckpt_raw, field=[OPTIM_TARGET, "K", "Delay", "Stride", "Phase"]).collect()
        stat = gt_stat_from_exp_name(ckpt_raw, ["K", "Delay", "Stride", "Phase"], [OPTIM_TARGET], is_ckpt=False)
        data = denormalize([data], stat)[0]
        
        # Check available delay values and use all data (since it appears to have single delay value)
        available_delays = sorted(data["Delay"].unique().to_list())
        print(f"Available delay values in raw data: {available_delays}")
        filtered_data = data  # Use all data since delay is constant
        
        if len(filtered_data) == 0:
            print(f"No raw data found for {ckpt_raw}")
        else:
            # Plot raw data sweep for each fixed K value
            exo_off_cot = plot_exo_raw(ax_raw, filtered_data, fix_k=0.0, average=True)
            for idx, fix_k in enumerate(FIX_K):
                color = cmap(norm(idx))
                # plot_exo_raw(ax_raw, filtered_data, fix_k=fix_k, color=color)
                plot_exo_raw(ax_raw, filtered_data, fix_k=fix_k, color=color, divider=exo_off_cot)
                
    except Exception as e:
        print(f"[Error] Raw data {ckpt_raw}: {e}")
        import traceback
        traceback.print_exc()
    
    # Load and plot model predictions for three checkpoints
    ax_models = [ax_model1, ax_model2, ax_model3]
    
    for checkpoint_idx, (ckpt_model, version) in enumerate(zip(CKPT_MODEL, CKPT_VERSIONS)):
        try:
            print(f"Loading model from: {ckpt_model} with version {version}")
            exp_name = ckpt_model + "/" + version
            optimizer = BOptimizer(exp_name, epoch=EPOCH)
            
            missing = [col for col in INPUT_FIX + INPUT_VAR if col not in optimizer.model.input_col]
            assert not missing, f"Missing input for the NN model: {missing}"
            assert OPTIM_TARGET in optimizer.model.output_col, f"The target column {OPTIM_TARGET} is not in the NN model."
            
            # Plot model predictions sweep for each fixed K value
            ax_current = ax_models[checkpoint_idx]
            exo_off_cot = plot_exo_model(ax_current, optimizer, fix_k=0.0, average=True)
            for idx, fix_k in enumerate(FIX_K):
                color = cmap(norm(idx))
                plot_exo_model(ax_current, optimizer, fix_k=fix_k, color=color, divider=exo_off_cot)
                
        except Exception as e:
            print(f"[Error] Model {ckpt_model} with version {version}: {e}")
            import traceback
            traceback.print_exc()
    
    # Configure raw data plot
    ax_raw.set_title("Raw gait data", fontsize=FONT_SIZE_LABEL + 4)
    ax_raw.yaxis.set_major_locator(plt.MaxNLocator(4))
    ticks = ax_raw.get_yticks()
    ticks = list(ticks) + [1]
    ax_raw.set_yticks(ticks)
    ax_raw.grid(True, alpha=0.3)
    ax_raw.set_ylabel("Normalized CoT", fontsize=FONT_SIZE_LABEL + 4)
    ax_raw.set_ylim(Yrange[0], Yrange[1])
    
    # Show legend for raw data (for "Exo off" line)
    handles, labels = ax_raw.get_legend_handles_labels()
    if handles:
        ax_raw.legend(handles, labels, fontsize=FONT_SIZE_LABEL, loc='upper right')
    
    # Configure model prediction plots for all three checkpoints
    for checkpoint_idx, (version, title, ax_model) in enumerate(zip(CKPT_VERSIONS, CKPT_TITLES, ax_models)):
        # Use predefined titles
        ax_model.set_title(title, fontsize=FONT_SIZE_LABEL + 4)
        ax_model.set_ylim(Yrange[0], Yrange[1])
        ax_model.yaxis.set_major_locator(plt.MaxNLocator(4))
        ticks = ax_model.get_yticks()
        ticks = list(ticks) + [1]
        ax_model.set_yticks(ticks)
        ax_model.set_ylim(Yrange[0], Yrange[1])
        ax_model.grid(True, alpha=0.3)
    fig.supxlabel("Walking speed (km/h)", fontsize=FONT_SIZE_LABEL + 4)
        
        # No y-axis label for model plots since raw data plot already has it
        

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02,)
    cbar.set_label(r'Exoskeleton gain, $\kappa$ (Nm)', fontsize=FONT_SIZE_LABEL, labelpad=10)
    
    # Set colorbar ticks to show actual K values in Nm
    k_values_nm = [k * 30 for k in FIX_K]  # Convert to Nm
    cbar.set_ticks(np.linspace(0, len(FIX_K)-1, 3))
    cbar.set_ticklabels([f'{k_values_nm[0]:.0f}', f'{k_values_nm[len(k_values_nm)//2]:.0f}', f'{k_values_nm[-1]:.0f}'])
    cbar.ax.tick_params(labelsize=FONT_SIZE_LEGEND)

    plt.savefig(PLOT_SAVE_DIR / "supp_f6.png", dpi=150, transparent=True, bbox_inches="tight")



if __name__ == "__main__":
    main()
