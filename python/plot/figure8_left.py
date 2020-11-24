# Optimize exo parameter (K, Delay) for velocity
# Sampled param is some fixed gait paramimage.pngeter (phase and stride) and varying exo parameters (K and Delay)
from python.surrogate.optimizer import Optimizer
from python.surrogate.batched_optimizer import Optimizer as BOptimizer
from python.surrogate.preprocess import load_cot_processed
from python.plot.util import *
from tqdm import tqdm
import argparse
import matplotlib.cm as cm
import matplotlib.colors as mcolors

################ CONFIG ################
CKPT_MODEL = [
"exo_hei_resist_0807_131406_21000+memo__b21df_LHS60000_no_mod_state_ma15+_on_0821_145627",
]

# Checkpoint versions to overlay
CKPT_VERSIONS = [
    "nn_v01", 
    "nn_v02"
]

CKPT_RAW = [
'exo_hei_resist_21000_0807_131406+memo_exo_trend101_U1050_no_mod_state_ma15_gems+_on_0831_225143',
]

EPOCH = 10000
# EPOCH = 5000

FIX_K = np.arange(0, 0.41, 0.4/20)
Yrange = (0.95, 1.22)

# Fixed gait parameters (Stride and Phase)
FIX_STRIDE = 1.013
# FIX_STRIDE = 1.0
# FIX_STRIDE = 0.8
FIX_PHASE = 1.013
# FIX_PHASE = 1.0
# FIX_PHASE = 0.8

# Delay sweep parameters
DELAY_MIN = 0.0
DELAY_MAX = 0.5
DELAY_STEPS = 50
########################################

point_size = 10
PROCESSOR = "metabolic"

OPTIM_TARGET = "cot_ma15"
COT_TARGET = "cot_ma15"
INPUTs = ["K", "Delay", "Stride", "Phase"]
MA_OUTPUT = False


def plot_exo_model(ax, optimizer, fix_k, average=False, color=None, divider=1.0):
    """Plot model predictions using the approach from exo_benefit2.py"""
    # Create delay sweep array
    delay_values = np.linspace(DELAY_MIN, DELAY_MAX, DELAY_STEPS)
    
    # Create input tensor for all delay values at once (batch processing)
    input_tensor = torch.full((DELAY_STEPS, len(optimizer.model.input_col)), float('nan'), dtype=torch.float32, device=optimizer.model.device)
    input_tensor[:, optimizer.model.input_col.index("Phase")] = FIX_PHASE
    input_tensor[:, optimizer.model.input_col.index("Stride")] = FIX_STRIDE  
    input_tensor[:, optimizer.model.input_col.index("K")] = fix_k
    input_tensor[:, optimizer.model.input_col.index("Delay")] = torch.from_numpy(delay_values)
    
    # Compute exo CoT (single forward call)
    base_param = optimizer.model.normalize_tensor(input_tensor.clone(), optimizer.model.input_col)
    with torch.no_grad():
        exo_cot = optimizer.model(base_param)
    exo_cot = optimizer.model.denormalize_tensor(exo_cot, optimizer.model.output_col)
    exo_cot = exo_cot.cpu().numpy().squeeze()

    if average:
        exo_off_cot = exo_cot.mean(axis=0)
        ax.axhline(y=1, color='k', linestyle='--', label=r"exo off ($\kappa=0$)", linewidth=2)
        return exo_off_cot
    else:
        ax.plot(delay_values, exo_cot / divider, '-', color=color, linewidth=2)

def plot_exo_raw(ax, filtered_data, fix_k, average=False, color=None, divider=1.0):
    """Plot raw data using the approach from exo_benefit3.py"""
    # Find the nearest K value in the data
    available_k_values = filtered_data["K"].unique().to_list()
    nearest_k = min(available_k_values, key=lambda x: abs(x - fix_k))
    
    # Filter data for the nearest K value
    k_filtered_data = filtered_data.filter(pl.col("K") == nearest_k)
    
    if len(k_filtered_data) == 0:
        print(f"No data found for K={nearest_k} (requested {fix_k})")
        return
    
    k_filtered_data = k_filtered_data.sort("Delay")
    delays = k_filtered_data["Delay"].to_numpy()
    cots = k_filtered_data[COT_TARGET].to_numpy()
    
    if average:
        ax.axhline(y=1, color='k', linestyle='--', label=f"exo off", linewidth=2)
        return np.mean(cots)
    else:
        if MA_OUTPUT:
            cots = moving_average(cots, window_size=int(len(cots) * 0.1))
        ax.plot(delays, cots / divider, '-', color=color, linewidth=2)


def main():
    global CKPT_MODEL
    global CKPT_RAW

    parser = argparse.ArgumentParser(description='Plot exoskeleton benefit comparison by version')
    parser.add_argument('--plot-raw', action='store_true', default=True,
                       help='Include raw data plot (default: True)')
    parser.add_argument('--no-plot-raw', dest='plot_raw', action='store_false',
                       help='Skip raw data plot')
    args = parser.parse_args()

    assert len(CKPT_MODEL) == 1 and len(CKPT_RAW) == 1, "Only one model and one raw data can be plotted at a time"
    CKPT_MODEL = CKPT_MODEL[0]
    CKPT_RAW = CKPT_RAW[0]

    # Create subplots based on whether raw data should be plotted
    if args.plot_raw:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.subplots_adjust(
            left=0.06,
            right=1.07,
            top=0.92,
            bottom=0.115,
            wspace=0.3,
        )
        ax_raw = axes[0]
        ax_model1 = axes[1]
        ax_model2 = axes[2]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.subplots_adjust(
            left=0.09,
            right=1.05,
            top=0.92,
            bottom=0.115,
            wspace=0.3,
        )
        ax_raw = None
        ax_model1 = axes[0]
        ax_model2 = axes[1]
    
    # Setup colormap
    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=len(FIX_K)-1)
    
    # Load and plot raw data (only if requested)
    if args.plot_raw:
        try:
            print(f"Loading raw data from: {CKPT_RAW}")
            data = load_cot_processed(CKPT_RAW, field=[COT_TARGET, "K", "Delay"]).collect()

            # Filter data with fixed parameters
            PARAMS = {
                "Phase": FIX_PHASE,
                "Stride": FIX_STRIDE,
            }
            filtered_data, param_fixed = filter_df(data, PARAMS)

            if len(filtered_data) == 0:
                print(f"No raw data found for {CKPT_RAW} with fixed parameters")
            else:
                # Plot raw data sweep for each fixed K value
                exo_off_cot = plot_exo_raw(ax_raw, filtered_data, fix_k=0.0, average=True)
                for idx, fix_k in enumerate(FIX_K):
                    color = cmap(norm(idx))
                    plot_exo_raw(ax_raw, filtered_data, fix_k=fix_k, color=color, divider=exo_off_cot)

        except Exception as e:
            print(f"[Error] Raw data {CKPT_RAW}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping raw data plot as requested")
    
    # Load and plot model predictions for both versions
    ax_models = [ax_model1, ax_model2]
    
    for version_idx, version in enumerate(CKPT_VERSIONS):
        try:
            print(f"Loading model from: {CKPT_MODEL} with version {version}")
            exp_name = CKPT_MODEL + "/" + version
            optimizer = BOptimizer(exp_name, epoch=EPOCH)
            
            missing = [col for col in INPUTs if col not in optimizer.model.input_col]
            assert not missing, f"Missing input for the NN model: {missing}"
            assert OPTIM_TARGET in optimizer.model.output_col, f"The target column {OPTIM_TARGET} is not in the NN model."
            
            # Plot model predictions sweep for each fixed K value
            ax_current = ax_models[version_idx]
            exo_off_cot = plot_exo_model(ax_current, optimizer, fix_k=0.0, average=True)
            for idx, fix_k in enumerate(FIX_K):
                color = cmap(norm(idx))
                plot_exo_model(ax_current, optimizer, fix_k=fix_k, color=color, divider=exo_off_cot)
                
        except Exception as e:
            print(f"[Error] Model {CKPT_MODEL} with version {version}: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate fixed velocity for display
    fixed_velocity = FIX_STRIDE * FIX_PHASE * REF_VELOCITY * 3.6
    
    # Configure raw data plot (only if it exists)
    if args.plot_raw and ax_raw is not None:
        ax_raw.set_title(f"Raw gait data", fontsize=FONT_SIZE_LABEL)
        ax_raw.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax_raw.grid(True, alpha=0.3)
        ax_raw.set_xticks([DELAY_MIN, (DELAY_MIN + DELAY_MAX) / 2, DELAY_MAX])
        ax_raw.set_xlim(DELAY_MIN, DELAY_MAX)
        yticks = ax_raw.get_yticks()
        yticks = list(yticks) + [1]
        ax_raw.set_yticks(yticks)
        ax_raw.set_ylim(Yrange[0], Yrange[1])
        # Set x label for raw data plot
        ax_raw.set_xlabel("Delay (s)", fontsize=FONT_SIZE_LABEL)
        
    # Configure model prediction plots for both versions
    for version_idx, (version, ax_model) in enumerate(zip(CKPT_VERSIONS, ax_models)):
        # Use version-specific titles
        if version == "nn_v01":
            ax_model.set_title(r"Surrogate ($\lambda_{\mathrm{gp}} = 0.1$)", fontsize=FONT_SIZE_LABEL)
        elif version == "nn_v02":
            ax_model.set_title(r"Surrogate ($\lambda_{\mathrm{gp}} = 0.01$)", fontsize=FONT_SIZE_LABEL)
        else:
            ax_model.set_title(f"Surrogate ({version})", fontsize=FONT_SIZE_LABEL)
        
        ax_model.yaxis.set_major_locator(plt.MaxNLocator(5))
        
        ax_model.grid(True, alpha=0.3)
        ax_model.set_xticks([DELAY_MIN, (DELAY_MIN + DELAY_MAX) / 2, DELAY_MAX])
        ax_model.set_xlim(DELAY_MIN, DELAY_MAX)
        
        # Get yticks from raw data plot if available, otherwise use default range
        if args.plot_raw and ax_raw is not None:
            yticks = ax_raw.get_yticks()
            yticks = list(yticks) + [1]
            ax_model.set_yticks(yticks)
        ax_model.set_ylim(Yrange[0], Yrange[1])
        ax_model.set_xlabel("Delay (s)", fontsize=FONT_SIZE_LABEL)

    # Only show legend for "Exo off" line
    handles, labels = ax_model.get_legend_handles_labels()
    if handles:
        ax_model.legend(handles, labels, fontsize=FONT_SIZE_LABEL, loc='upper left')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02)
    cbar.set_label(r'Exoskeleton gain, $\kappa$ (Nm)', fontsize=FONT_SIZE_LABEL, labelpad=10)

    # Set colorbar ticks to show actual K values in Nm
    k_values_nm = [k * 30 for k in FIX_K]  # Convert to Nm
    cbar.set_ticks(np.linspace(0, len(FIX_K)-1, 3))
    cbar.set_ticklabels([f'{k_values_nm[0]:.0f}', f'{k_values_nm[len(k_values_nm)//2]:.0f}', f'{k_values_nm[-1]:.0f}'])
    cbar.ax.tick_params(labelsize=FONT_SIZE_LEGEND)

    # Set global y label for all subplots
    fig.supylabel("Normalized CoT", fontsize=FONT_SIZE_LABEL + 2)
    plt.savefig(PLOT_SAVE_DIR / "figure8_left.png", dpi=100, transparent=True)



if __name__ == "__main__":
    main()
