# Optimize exo parameter (K, Delay) for velocity
# Sampled param is some fixed gait parameter (phase and stride) and varying exo parameters (K and Delay)
import matplotlib.patheffects as path_effects
from python.surrogate.batched_optimizer import Optimizer
from python.surrogate.preprocess import load_cot_processed
from python.plot.util import *
import argparse

################ CONFIG ################
CKPTs = [
    'exo_hei_resist_0807_131406_21000+memo__b21df_LHS60000_no_mod_state_ma15+_on_0821_145627',
]

DIRECT_SAMPLED = "exo_hei_resist_21000_0807_131406+memo_exo_trend101_U1050_no_mod_state_ma15_gems+_on_0831_225143"

# Checkpoint versions to overlay
CKPT_VERSIONS = [
    "nn_v01", 
    "nn_v02"
]

PLOT_CONFIG = {
    "MA_OUTPUT": False,
    "REG_WINDOW": False,
    "REG_1ST_ORDER": True,
    "REG_2ND_ORDER": False,
    "MAX_ITER": 2000,
    # "MAX_ITER": 2,
    "TRIAL_SIZE": 128,
    "REG_BATCH": 1.0,
    "OPT_NUM": 100,
    "LINESTYPES": ['-', '--',],
}
########################################
REGULARIZE = { 
    "K": 0.01,
    "Delay": 0.0,
}

point_size = 10
PROCESSOR = "metabolic"

X_LBL = "Velocity"
OPTIM_TARGET = "cot_ma15"
INPUT_VAR = ["K", "Delay"]
INPUT_FIX = ["Stride", "Phase"]
ranges = {
    "K": (0, 21),
    "Delay": (0, 0.5),
    X_LBL: (2.6, 6),
    # X_LBL: (1, 5),
}
ticks = {
    X_LBL: [3, 4, 5, 6],
    "K": np.arange(ranges["K"][0], ranges["K"][1] + 0.01, 5),
    "Delay": np.arange(ranges["Delay"][0], ranges["Delay"][1] + 0.01, 0.25),
}

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num', type=int, default=PLOT_CONFIG["OPT_NUM"])
args = parser.parse_args()


# Inline style and label constants to reduce function overhead
LINESTYLES = {'K': '-', 'Delay': '--'}
LINEWIDTHS = {'K': 5, 'Delay': 2.5}

# Combined labels with proper LaTeX formatting for each parameter-version combination
COMBINED_LABELS = {
    ('K', 'nn_v01'): r"gain, $\kappa$   $\lambda_{\mathrm{gp}} = 0.1$",
    ('K', 'nn_v02'): r"gain, $\kappa$   $\lambda_{\mathrm{gp}} = 0.01$",
    ('Delay', 'nn_v01'): r"delay, $\Delta t$   $\lambda_{\mathrm{gp}} = 0.1$",
    ('Delay', 'nn_v02'): r"delay, $\Delta t$   $\lambda_{\mathrm{gp}} = 0.01$",
    # Fallback for unknown versions
    ('K', ''): r"gain, $\kappa$",
    ('Delay', ''): r"delay, $\Delta t$"
}

def plot_exo_optimized(ax_k, ax_delay, exp_idx: int, version_idx: int, reg_batch: float):
    exp_name = CKPTs[exp_idx]
    version = CKPT_VERSIONS[version_idx]
    exp_name = f"{exp_name}/{version}"
    
    optimizer = Optimizer(exp_name, reg_input=REGULARIZE, reg_batch=reg_batch)
    
    missing = [col for col in INPUT_FIX + INPUT_VAR if col not in optimizer.model.input_col]
    assert not missing, f"Missing input for the NN model: {missing}"
    assert OPTIM_TARGET in optimizer.model.output_col, f"The target column {OPTIM_TARGET} is not in the NN model."

    real_gait_selection = load_gems_selection()
    interpolated_cadence = np.linspace(min(real_gait_selection["Cadence"]), max(real_gait_selection["Cadence"]), args.num)
    interpolated_step = np.linspace(min(real_gait_selection["Step"]), max(real_gait_selection["Step"]), args.num)
    fixed_fields_template = torch.full((args.num, len(optimizer.model.input_col)), float('nan'), dtype=torch.float32)
    fixed_fields_template[:, optimizer.model.input_col.index("Phase")] = torch.from_numpy(interpolated_cadence / REF_CADENCE)
    fixed_fields_template[:, optimizer.model.input_col.index("Stride")] = torch.from_numpy(interpolated_step / REF_STEP)
    found_param_list = optimizer.run(
        OPTIM_TARGET, fixed_fields_template, max_iter=PLOT_CONFIG["MAX_ITER"], trial_size=PLOT_CONFIG["TRIAL_SIZE"], 
        reg_1st_order=PLOT_CONFIG["REG_1ST_ORDER"], reg_2nd_order=PLOT_CONFIG["REG_2ND_ORDER"], reg_window=PLOT_CONFIG["REG_WINDOW"])
    found_df = Df(found_param_list)
    found_df = found_df.with_columns((pl.col("Stride") * pl.col("Phase") * REF_VELOCITY * 3.6).alias("Velocity"))
    found_df = found_df.sort("Velocity")
    
    # scale K
    found_df = found_df.with_columns((pl.col("K") * 30).alias("K"))
    if PLOT_CONFIG["MA_OUTPUT"]:
        found_df = moving_average(found_df, window_size=int(PLOT_CONFIG["MAX_ITER"] * 0.1))
    
    # Use different colors for different versions
    color_idx = exp_idx * len(CKPT_VERSIONS) + version_idx

    # Get labels with proper LaTeX formatting
    k_label = COMBINED_LABELS.get(('K', version), COMBINED_LABELS[('K', '')])
    delay_label = COMBINED_LABELS.get(('Delay', version), COMBINED_LABELS[('Delay', '')])

    ax_k.plot(found_df["Velocity"], found_df["K"],
              label=k_label,
              linewidth=LINEWIDTHS['K'],
              color=get_color0(color_idx),
              linestyle=LINESTYLES['K'])
    ax_delay.plot(found_df["Velocity"], found_df["Delay"],
                  label=delay_label,
                  linewidth=LINEWIDTHS['Delay'],
                  color=get_color0(color_idx),
                  linestyle=LINESTYLES['Delay'])
    # ax_k.set_title(f'{exp_name} | REG: {reg_batch} | MA: {PLOT_CONFIG["MA_OUTPUT"]}', fontsize=font_size-5)
    return found_df


def plot_direct_sampled(ax_k, ax_delay):
    """Plot optimal parameters from direct sampled data"""
    try:
        print(f"Loading direct sampled data from: {DIRECT_SAMPLED}")
        data = load_cot_processed(DIRECT_SAMPLED, field=[OPTIM_TARGET, "K", "Delay"]).collect()
        
        print(f"Available columns: {data.columns}")
        print(f"Data shape: {data.shape}")
        
        # Check if Phase and Stride columns exist
        if "Phase" not in data.columns or "Stride" not in data.columns:
            print("Phase or Stride columns not found in direct sampled data")
            return None
        
        # Print data ranges for debugging
        phase_range = (data["Phase"].min(), data["Phase"].max())
        stride_range = (data["Stride"].min(), data["Stride"].max())
        print(f"Phase range: {phase_range[0]:.3f} to {phase_range[1]:.3f}")
        print(f"Stride range: {stride_range[0]:.3f} to {stride_range[1]:.3f}")
        
        # Calculate the single velocity from the fixed gait parameters
        fixed_velocity = phase_range[0] * stride_range[0] * REF_VELOCITY * 3.6
        
        # Find optimal parameters (minimum CoT) from the entire dataset
        optimal_idx = data[OPTIM_TARGET].arg_min()
        if optimal_idx is None:
            print("No optimal point found in direct sampled data")
            return None
            
        optimal_row = data[optimal_idx]
        optimal_k = optimal_row["K"][0] * 30  # Scale K to Nm
        optimal_delay = optimal_row["Delay"][0]
        
        # Plot as single points
        # ax_k.scatter([fixed_velocity], [optimal_k], label="", s=70,
        #             color='purple', marker='s', zorder=100,
        #             edgecolor='black', linewidth=1)
        # ax_k.annotate(r"Best $\kappa$ from raw data", xy=(fixed_velocity, optimal_k), xytext=(-3, 13),
        #              textcoords='offset points', fontsize=FONT_SIZE_LABEL,
        #              ha='center', va='bottom', color='purple',)
                    #  path_effects=[path_effects.withStroke(linewidth=1, foreground='black')]
        
        print(f"Successfully plotted direct sampled optimal point: K={optimal_k:.2f} Nm, Delay={optimal_delay:.3f} s at v={fixed_velocity:.1f} km/h")
        
        # Return a simple DataFrame for consistency
        return pl.DataFrame({
            "Velocity": [fixed_velocity],
            "K": [optimal_k], 
            "Delay": [optimal_delay]
        })
            
    except Exception as e:
        print(f"[Error] Direct sampled data {DIRECT_SAMPLED}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    fig, ax_k = plt.subplots(figsize=(5, 5))
    fig.subplots_adjust(
        left=0.15,
        right=0.8, 
        top=0.95,
        bottom=0.15,
    )

    ax_delay = ax_k.twinx()
    found_dfs = []
    
    # Plot direct sampled optimal parameters first
    direct_sampled_df = plot_direct_sampled(ax_k, ax_delay)
    # Note: Don't add direct_sampled_df to found_dfs as it has different column structure
    
    # Loop through both experiments and versions
    for exp_idx in range(len(CKPTs)):
        for version_idx in range(len(CKPT_VERSIONS)):
            try:
                found_df = plot_exo_optimized(ax_k, ax_delay, exp_idx, version_idx, PLOT_CONFIG["REG_BATCH"])
                found_dfs.append(found_df)
                
                # Mark star at x=4.5 with annotation "(b)" for each version
                # x_mark = 4.5
                # if X_LBL in ranges and ranges[X_LBL][0] <= x_mark <= ranges[X_LBL][1]:
                    # Add gray vertical band at x=4.5 (only once, for the first version)
                    # if version_idx == 0:
                        # ax_k.axvspan(x_mark - 0.1, x_mark + 0.1, ymin=0, ymax=12/ranges["K"][1], 
                                #    color='gray', alpha=0.3, zorder=1)
                        # ax_k.annotate('(b)', xy=(x_mark, 6), xytext=(0, 0), 
                                #  textcoords='offset points', fontsize=FONT_SIZE_LABEL-6,
                                #  ha='center', va='center', color='black')
                    
                    # Find the y-value on the K line at x=4.5
                    # closest_idx = int(np.argmin(np.abs(found_df["Velocity"] - x_mark)))
                    # y_k_value = found_df["K"][closest_idx]
                    
                    # Add star marker on K axis
                    # ax_k.plot(x_mark, y_k_value, marker='o', markersize=10, color='red', 
                            #  markeredgecolor='black', markeredgewidth=1, zorder=10)
                    
                    # Add text annotation "(b)" with version info
                    # version = CKPT_VERSIONS[version_idx]
                    # Extract just the lambda part from the combined label for annotation
                    # if version == "nn_v01":
                    #     annotation_text = r"$\lambda_{\mathrm{gp}} = 0.1^*$"
                    # elif version == "nn_v02":
                    #     annotation_text = r"$\lambda_{\mathrm{gp}} = 0.01^*$"
                    # else:
                    #     annotation_text = "$*$"
                    # # Offset annotation vertically for different versions to avoid overlap
                    # ax_k.annotate(annotation_text, xy=(x_mark, y_k_value), xytext=(10, 0),
                    #              textcoords='offset points', fontsize=FONT_SIZE_LABEL-8,
                    #              ha='left', va='center', color='black')
                
                print(f"Successfully plotted {CKPTs[exp_idx]} with {CKPT_VERSIONS[version_idx]}")
            except Exception as e:
                print(f"Error plotting {CKPTs[exp_idx]} with {CKPT_VERSIONS[version_idx]}: {e}")
                continue
    
    if found_dfs:
        found_df = pl.concat(found_dfs, how='vertical')
    else:
        print("No successful plots generated")
        return
    
    ax_k.set_xlabel("Walking speed (km/h)", fontsize=FONT_SIZE_LABEL)
    if X_LBL in ticks.keys():
        ax_k.set_xticks(ticks[X_LBL])
        ax_k.set_xlim([min(ranges[X_LBL]), max(ranges[X_LBL])])
    else:
        ax_k.set_xlim([min(found_df["Velocity"]), max(found_df["Velocity"])])
        
    ax_k.set_ylabel(r"Optimal gain, $\kappa$ (Nm)", labelpad=5, fontsize= FONT_SIZE_LABEL)
    ax_delay.set_ylabel(r"Optimal delay, $\Delta t$ (s)", labelpad=5, fontsize=FONT_SIZE_LABEL)
    # ax_k.set_ylabel(r"$\arg\min_\kappa$ CoT$(\kappa, \Delta t)$ (Nm)", labelpad=5, fontsize= FONT_SIZE_LABEL)
    # ax_delay.set_ylabel(r"$\arg\min_{\Delta t}$ CoT$(\kappa, \Delta t)$ (s)", labelpad=5, fontsize=FONT_SIZE_LABEL, color='blue')
    ax_k.set_yticks(ticks["K"])
    ax_delay.set_yticks(ticks["Delay"])
    ax_k.set_ylim([min(ticks["K"]), max(ticks["K"])])
    ax_delay.set_ylim([min(ticks["Delay"]), max(ticks["Delay"])])
    ax_k.tick_params(axis='both', direction='in', length=8)
    ax_delay.tick_params(axis='both', direction='in', length=8)
    # ax_delay.spines['right'].set_color('blue')
    
    ax_k.legend()
    ax_delay.legend()
    handles_k, labels_k = ax_k.get_legend_handles_labels()
    handles_d, labels_d = ax_delay.get_legend_handles_labels()

    handles = handles_k + handles_d
    labels = labels_k + labels_d
    legend = ax_k.legend(handles, labels, loc='upper center', fontsize=FONT_SIZE_LABEL-6)
    legend.get_frame().set_alpha(0.3)
    ax_delay.legend().remove()
    plt.savefig(PLOT_SAVE_DIR / "figure8_right.png", dpi=100, transparent=True)


if __name__ == "__main__":
    main()
