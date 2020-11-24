# Optimize exo parameter (K, Delay) for velocity
# Sampled param is some fixed gait parameter (phase and stride) and varying exo parameters (K and Delay)
from python.surrogate.batched_optimizer import Optimizer
from python.plot.util import *
import argparse

################ CONFIG ################
CKPTs = [
'exo_hei_resist_21000_0807_131406+memo_b21df_LHS5000_no_mod_state_ma15_gems+_on_0902_122806',
'exo_hei_resist_21000_0807_131406+memo_b21df_LHS10000_no_mod_state_ma15_gems+_on_0901_214322',
'exo_hei_resist_21000_0807_131406+memo_b21df_LHS20000_no_mod_state_ma15_gems+_on_0902_074512',          
'exo_hei_resist_0807_131406_21000+memo__b21df_LHS60000_no_mod_state_ma15+_on_0821_145627',
]

Versions = [
    None,
    None,
    None,
    "nn_v02",
]

TITLES = [
    "Sample size: 5k",
    "Sample size: 10k",
    "Sample size: 20k",
    "Sample size: 60k",
]

PLOT_CONFIG = {
    "MA_OUTPUT": False,
    "REG_WINDOW": False,
    "REG_1ST_ORDER": True,
    "REG_2ND_ORDER": False,
    "MAX_ITER": 2000,
    # "MAX_ITER": 10,
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
    X_LBL: (2, 6),
    # X_LBL: (1, 5),
}
ticks = {
    X_LBL: np.arange(ranges[X_LBL][0], ranges[X_LBL][1] + 0.01, 1.0),
    "K": np.arange(ranges["K"][0], ranges["K"][1] + 0.01, 5),
    "Delay": np.arange(ranges["Delay"][0], ranges["Delay"][1] + 0.01, 0.25),
}

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num', type=int, default=PLOT_CONFIG["OPT_NUM"])
args = parser.parse_args()


def get_linestyle(data_name: str):
    if data_name == 'K':
        return '-'
        # return '--'
    elif data_name == 'Delay':
        return '--'
        # return '-'
    else:
        return '-'

def get_linewidth(data_name: str):
    if data_name == 'K':
        return 5
        # return 1
    elif data_name == 'Delay':
        return 1
        # return 5
    else:
        return 1

def get_label(data_name: str):
    if data_name == 'K':
        return r"gain, $\kappa$"
        # return "Exo gain (dashed)"
    elif data_name == 'Delay':
        return r"delay, $\Delta t$"
        # return "Exo delay (solid)"
    else:
        return ""

def plot_exo_optimized(ax_k, ax_delay, exp_idx: int, reg_batch: float):
    exp_name = CKPTs[exp_idx]
    if Versions[exp_idx] is not None:
        exp_name = exp_name + "/" + Versions[exp_idx]
    else:
        exp_name = opt_ckpt_versioned_path(exp_name)
    
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
    ax_k.plot(found_df["Velocity"], found_df["K"], label=get_label('K'),
              linewidth=get_linewidth('K'), linestyle=get_linestyle('K'))
    ax_delay.plot(found_df["Velocity"], found_df["Delay"], label=get_label('Delay'), 
                  linewidth=get_linewidth('Delay'), linestyle=get_linestyle('Delay'))
    return found_df


def main():
    n_experiments = len(CKPTs)
    assert n_experiments == len(TITLES), "Number of experiments and titles must match"
    if n_experiments == 0:
        print("No experiments to plot")
        return
    
    # Calculate subplot layout (single row, all flat)
    n_rows, n_cols = 1, n_experiments
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_experiments == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    found_dfs = []
    for idx in range(n_experiments):
        ax_k = axes[idx]
        ax_delay = ax_k.twinx()
        
        found_df = plot_exo_optimized(ax_k, ax_delay, idx, PLOT_CONFIG["REG_BATCH"])
        found_dfs.append(found_df)
        
        # Set labels and formatting for each subplot
        if X_LBL in ticks.keys():
            ax_k.set_xticks(ticks[X_LBL])
            ax_k.set_xlim([min(ticks[X_LBL]), max(ticks[X_LBL])])
        else:
            ax_k.set_xlim([min(found_df["Velocity"]), max(found_df["Velocity"])])
        
        if idx == 0:
            ax_k.set_ylabel(r"Optimal gain, $\kappa$ (Nm)", labelpad=5, fontsize=FONT_SIZE_LABEL + 4)
            ax_delay.set_ylabel(r"Optimal delay, $\Delta t$ (s)", labelpad=5, fontsize=FONT_SIZE_LABEL + 4)
        ax_k.set_yticks(ticks["K"])
        ax_delay.set_yticks(ticks["Delay"])
        ax_k.set_ylim([min(ticks["K"]), max(ticks["K"])])
        ax_delay.set_ylim([min(ticks["Delay"]), max(ticks["Delay"])])
        ax_k.tick_params(axis='both', direction='in', length=8)
        ax_delay.tick_params(axis='both', direction='in', length=8)
        # ax_delay.spines['right'].set_color('blue')
        
        # Add subplot title with experiment name
        ax_k.set_title(TITLES[idx], fontsize=FONT_SIZE_LABEL + 4)
        
        # Handle legend for each subplot
        if idx == 0:
            handles_k, labels_k = ax_k.get_legend_handles_labels()
            handles_d, labels_d = ax_delay.get_legend_handles_labels()
            handles = handles_k + handles_d
            labels = labels_k + labels_d
            if handles:
                legend = ax_k.legend(handles, labels, loc='upper right', fontsize=FONT_SIZE_LABEL)
                legend.get_frame().set_alpha(0.3)
        else:
            ax_k.legend().remove()
        ax_delay.legend().remove()
    
    # Hide unused subplots
    for idx in range(n_experiments, len(axes)):
        axes[idx].set_visible(False)
    
    found_df = pl.concat(found_dfs, how='vertical') if found_dfs else None
    fig.supxlabel("Walking speed (km/h)", fontsize=FONT_SIZE_LABEL + 4)
    
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_DIR / "supp_e5.png", dpi=150, transparent=True)


if __name__ == "__main__":
    main()
