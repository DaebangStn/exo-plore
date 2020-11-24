from python.surrogate.preprocess import filter_lastrun
from matplotlib.patches import Patch
import matplotlib.lines as mlines
from scipy import interpolate
from python.plot.util import *

################ CONFIG ################
CKPTs = [
    'no_exo_meta_ma15_10000_0801_111350+memo_b20_ma15_10000_0801_111350_pf_param_act+_on_0829_115420',
    'no_exo_meta_ma15_10000_0801_111421+memo_b20_ma15_10000_0801_111421_pf_param_act+_on_0829_115433',
    'no_exo_meta_ma15_10000_0801_205540+memo_b20_ma15_10000_0801_205540_pf_param_act+_on_0829_115446',
    'no_exo_meta_ma15_10000_0802_064456+memo_b20_ma15_10000_0802_064456_pf_param_act+_on_0829_115458',
    'no_exo_meta_ma15_10000_0802_163808+memo_b20_ma15_10000_0802_163808_pf_param_act+_on_0829_115512',
    'no_exo_meta_ma15_10000_0805_215352+memo_b20_ma15_10000_0805_215352_pf_param_act+_on_0829_115525',
    'no_exo_meta_ma15_10000_0805_215354+memo_b20_ma15_10000_0805_215354_pf_param_act+_on_0829_115537',
    'no_exo_meta_ma15_10000_0805_215359+memo_b20_ma15_10000_0805_215359_pf_param_act+_on_0829_115550',
    'no_exo_meta_ma15_10000_0806_131413+memo_b20_ma15_10000_0806_131413_pf_param_act+_on_0829_115603',
    'no_exo_meta_ma15_10000_0806_131416+memo_b20_ma15_10000_0806_131416_pf_param_act+_on_0829_115616',
    'no_exo_meta_ma15_10000_0806_131419+memo_b20_ma15_10000_0806_131419_pf_param_act+_on_0829_115628',
]
########################################
CYCLE_AFTER = 13
PLOT_MUSCLES = [
    "Bicep_Femoris_Longus",
    "Bicep_Femoris_Short",
    "Gastrocnemius_Lateral_Head",
    "Gastrocnemius_Medial_Head",
    "Soleus",
    "Gluteus_Maximus",
    "Gluteus_Medius",
    "Vastus_Lateralis",
    "Vastus_Medialis",
    "Rectus_Femoris",
    "Tibialis_Anterior",
    "Peroneus_Brevis",
    "Peroneus_Longus",
    "Semitendinosus",
]
PLOT_MUSCLES_TITLE = [
    "Bicep Femoris\nLongus",
    "Bicep Femoris\nShort",
    "Gastrocnemius\nLateral Head",
    "Gastrocnemius\nMedial Head",
    "Soleus",
    "Gluteus Maximus",
    "Gluteus Medius",
    "Vastus Lateralis",
    "Vastus Medialis",
    "Rectus Femoris",
    "Tibialis Anterior",
    "Peroneus Brevis",
    "Peroneus Longus",
    "Semitendinosus",
]
range_sigma = 1
fontsize = 23
xticks = [0, 50, 100]
yticks = [0, 1]
MA_WINDOW = 10

cmap = plt.get_cmap("viridis")

def collect_all_imas_activation():
    """
    Collect activation data from all checkpoints and compute mean/std averaged across all muscle lines
    
    Returns:
        Dict mapping muscle names to (x_data, mean_data, std_data) tuples
    """
    # Use a standard time grid for all data (0 to 100%)
    standard_time_points = 101  # 0, 1, 2, ..., 100%
    x_data = np.linspace(0, 100, standard_time_points)
    
    # Structure: {muscle_name: [checkpoint1_avg_data, checkpoint2_avg_data, ...]}
    all_muscle_avg_data = {}
    
    for muscle_name in PLOT_MUSCLES:
        all_muscle_avg_data[muscle_name] = []
    
    for exp_idx, exp_name in enumerate(CKPTs):
        print(f"Processing checkpoint {exp_idx+1}/{len(CKPTs)}: {exp_name}")
        
        try:
            data, _ = data_and_param_path(exp_name, read_file=True)
            param_data = data.select("param_idx").unique().collect()
            assert len(param_data) == 1, f"Expected single parameter index, got {len(param_data)}"
            data = filter_lastrun(data)
            data = data.filter(pl.col("cycle") == CYCLE_AFTER)
            
            num_step = len(data.select("step").collect())
            if num_step == 0:
                print(f"Warning: No data found for checkpoint {exp_name}")
                continue
                
            # Original time grid for this checkpoint
            original_x = np.linspace(0, 100, num_step + 1)
            
            for muscle_name in PLOT_MUSCLES:
                muscle_data = data.select(pl.col(f"^act_R_{muscle_name}.*$")).collect()
                muscle_data = moving_average(muscle_data, MA_WINDOW)
                
                if muscle_data.shape[0] == 0:
                    print(f"Warning: No muscle data for {muscle_name} in {exp_name}")
                    continue
                
                # Average across all muscle lines for this checkpoint
                muscle_avg = np.mean(muscle_data.to_numpy(), axis=1)
                
                # Interpolate to standard time grid
                f = interpolate.interp1d(original_x, muscle_avg, kind='linear', 
                                       bounds_error=False, fill_value='extrapolate')
                interpolated_avg = f(x_data)
                
                # Store averaged data for this checkpoint
                all_muscle_avg_data[muscle_name].append(interpolated_avg)
                
        except Exception as e:
            print(f"Error processing checkpoint {exp_name}: {e}")
            continue
    
    # Compute mean and std across all checkpoints
    muscle_stats = {}
    for muscle_name in PLOT_MUSCLES:
        if all_muscle_avg_data[muscle_name]:
            # Stack data across checkpoints
            checkpoint_array = np.array(all_muscle_avg_data[muscle_name])
            mean_activation = np.mean(checkpoint_array, axis=0)
            std_activation = np.std(checkpoint_array, axis=0)
            muscle_stats[muscle_name] = (x_data, mean_activation, std_activation)
            print(f"Processed {muscle_name}: {len(all_muscle_avg_data[muscle_name])} checkpoints")
        else:
            print(f"Warning: No data for {muscle_name}")
            muscle_stats[muscle_name] = (x_data, np.zeros_like(x_data), np.zeros_like(x_data))
    
    return muscle_stats


def plot_imas_activation(axes):
    """
    Plot IMAS activation data with mean and standard deviation across all checkpoints
    """
    muscle_stats = collect_all_imas_activation()
    
    for idx, muscle_name in enumerate(PLOT_MUSCLES):
        ax = axes[idx]
        x_data, mean_activation, std_activation = muscle_stats[muscle_name]
        
        # Plot mean line for simulation data in red
        ax.plot(x_data, mean_activation, color='red', linewidth=2, alpha=0.8)
        
        # Add std deviation band for simulation data in red
        ax.fill_between(x_data, mean_activation - std_activation, mean_activation + std_activation, 
                       color='red', alpha=0.2)

def plot_real_activation(axes):
    emgs = load_emg(PLOT_MUSCLES)
    for idx, muscle_name in enumerate(PLOT_MUSCLES):
        ax = axes[idx]
        # Plot human reference data in blue
        ax.plot(emgs['x'], emgs[muscle_name], color='gray', linewidth=2, alpha=0.8)
        if f"{muscle_name}_std" in emgs:    
            ax.fill_between(emgs['x'], emgs[f"{muscle_name}"] + emgs[f"{muscle_name}_std"], 
                           emgs[f"{muscle_name}"] - emgs[f"{muscle_name}_std"], color='gray', alpha=0.2)

def main():
    # Use GridSpec for custom subplot layout: 2 rows with 7, 7 muscles
    n_muscles = len(PLOT_MUSCLES)
    fig = plt.figure(figsize=(40, 10))

    # Create GridSpec with 2 rows and 7 columns
    gs = fig.add_gridspec(2, 7, hspace=0.4, wspace=0.15)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.87, bottom=0.15)

    axes = []
    muscle_idx = 0

    # Row 0: 7 plots
    for col in range(7):
        if muscle_idx < n_muscles:
            ax = fig.add_subplot(gs[0, col])
            axes.append(ax)
            muscle_idx += 1

    # Row 1: 7 plots
    for col in range(7):
        if muscle_idx < n_muscles:
            ax = fig.add_subplot(gs[1, col])
            axes.append(ax)
            muscle_idx += 1
    
    # Configure each subplot
    for idx in range(len(PLOT_MUSCLES)):
        ax = axes[idx]
        ax.set_xlim(0, 100)
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', labelsize=FONT_SIZE_LABEL)
        ax.set_ylim(-0.05, 1)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_yticks(yticks)
        ax.tick_params(axis='y', labelsize=FONT_SIZE_LABEL)
        ax.set_title(PLOT_MUSCLES_TITLE[idx], fontsize=FONT_SIZE_LABEL)

    # Plot real human data
    plot_real_activation(axes)
    
    # Plot simulation data with mean and std across all checkpoints
    plot_imas_activation(axes)
    
    ax = axes[-1]
    
    # Create legend elements
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
    lg = ax.legend(legend_elements, labels, loc='upper right', fontsize=FONT_SIZE_LABEL,)
    lg.get_frame().set_alpha(0.3)
    
    fig = plt.gcf()
    fig.supxlabel("Gait cycle (%)", fontsize=FONT_SIZE_LABEL + 8)
    fig.supylabel("Activation", fontsize=FONT_SIZE_LABEL + 8)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "plot/figure3.png", dpi=100, transparent=True)

if __name__ == "__main__":
    main()