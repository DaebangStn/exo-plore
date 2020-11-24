from python.surrogate.preprocess import filter_lastrun
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from python.plot.util import *


################ CONFIG ################
CKPTs = [
    # wo.imas + w.imas

    # 'no_imas_10000_0719_175556+memo_debug_act_on_0720_112953',
    # 'no_imas_3000_0720_175839+memo_debug_act_on_0720_202330',
    # 'no_imas_3000_0721_103026+memo_debug_act_on_0721_162501',
    # 'no_imas_3000_0721_125336+memo_debug_act_on_0721_162514',
    # 'no_imas_3000_0721_174812+memo_debug_act_on_0722_061834', # biased
    'no_imas_10000_0721_174812+memo_debug_act_on_0722_061755',
    # 'no_imas_3000_0721_152027+memo_debug_act_on_0722_061815',
    # 'no_imas_3000_0722_015300+memo_debug_act_on_0722_061852',
    # 'no_imas_5000_0722_015300+memo_debug_act_on_0722_061909',
    # 'no_exo_meta_ma2_20000_0620_135349+memo_debug_act_on_0720_135412',
    # 'no_exo_meta_ma2_10000_0624_110744+memo_debug_act_on_0720_140110',
    # 'no_exo_meta_ma2_10000_0626_124154+memo_debug_act_on_0720_140126',
    # 'no_exo_meta_ma2_20000_0621_184917+memo_debug_act_on_0720_140141',
    'no_exo_meta_ma2_20000_0623_103603+memo_debug_act_on_0720_140156',
]
########################################
CYCLE_AFTER = 13
PLOT_MUSCLES = [
    # "Bicep_Femoris_Longus",
    # "Bicep_Femoris_Short",
    # "Gastrocnemius_Lateral_Head",
    # "Gastrocnemius_Medial_Head",
    "Gluteus_Maximus",
    "Gluteus_Medius",
    # "Rectus_Femoris",
    "Soleus",
    # "Tibialis_Anterior",
    # "Vastus_Lateralis",
    # "Vastus_Medialis",
    # "Peroneus_Brevis",
    # "Peroneus_Longus",
    # "Semitendinosus",
]
fontsize = 12
label_padsize = 9
range_sigma = 2.5
xticks = [0, 50, 100]
yticks = [0, 1]
MA_WINDOW = 10


cmap = plt.get_cmap("viridis")

def plot_imas_activation(axes, exp_idx: int):
    exp_name = CKPTs[exp_idx]
    
    data, _ = data_and_param_path(exp_name, read_file=True)
    param_data = data.select("param_idx").unique().collect()
    assert len(param_data) == 1, f"Expected single parameter index, got {len(param_data)}"
    data = filter_lastrun(data)
    # data = filter_prerun_w_cycle(data, CYCLE_AFTER)
    data = data.filter(pl.col("cycle") == CYCLE_AFTER)
    num_step = len(data.select("step").collect())
    x_data = np.linspace(0, 100, num_step + 1)

    for idx, muscle_name in enumerate(PLOT_MUSCLES):
        ax = axes[idx]
        muscle_data = data.select(pl.col(f"^act_R_{muscle_name}.*$")).collect()
        muscle_data = moving_average(muscle_data, MA_WINDOW)
        print(f"{muscle_name} {muscle_data.shape}")
        for jdx in range(len(muscle_data.columns)):
            norm = matplotlib.colors.Normalize(vmin=0, vmax=len(muscle_data.columns) - 1)
            color = cmap(norm(jdx))
            ax.plot(x_data, muscle_data[:, jdx], linewidth=2.5, color=color)


def plot_real_activation(axes):
    emgs = load_emg(PLOT_MUSCLES)
    for idx, muscle_name in enumerate(PLOT_MUSCLES):
        ax = axes[idx]
        ax.plot(emgs['x'], emgs[muscle_name], color='black', label="Human exp.", linewidth=3, alpha=0.5)
        
def main():
    assert len(CKPTs) == 2, f"Only two checkpoints are supported. But got {len(CKPTs)}"
    axes = set_axes(plt, len(CKPTs) * len(PLOT_MUSCLES), remove_frame=True, num_plot_row=2, figsize=4)
    for idx, ax in enumerate(axes):
        ax.set_xlim(0, 100)
        ax.set_xticks(xticks)
        ax.set_ylim(-0.05, 1)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        ax.set_yticks(yticks)
        if idx < len(PLOT_MUSCLES):
            ax.set_title(PLOT_MUSCLES[idx].replace("_", " "), fontsize=FONT_SIZE_LABEL)
    axes[0].set_ylabel("\n" + r"$\mathrm{without}\ \mathcal{L}_{\mathrm{IMR}}$", fontsize=FONT_SIZE_LABEL)
    axes[3].set_ylabel("\n" + r"$\mathrm{with}\ \mathcal{L}_{\mathrm{IMR}}$", fontsize=FONT_SIZE_LABEL)

    real_line = Line2D([], [], color='gray', linewidth=6, label='Human subject')
    sim_note = mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='(Colored)Simulation')
    axes[0].legend(handles=[real_line, sim_note], fontsize=FONT_SIZE_LABEL-4, loc='upper right')
    
    plot_real_activation(axes[:len(PLOT_MUSCLES)])
    plot_real_activation(axes[len(PLOT_MUSCLES):])
    for exp_idx in range(len(CKPTs)):
        plot_imas_activation(axes[exp_idx * len(PLOT_MUSCLES):(exp_idx + 1) * len(PLOT_MUSCLES)], exp_idx)
    
    fig = plt.gcf()
    fig.supxlabel("Gait cycle (%)", fontsize=FONT_SIZE_LABEL)
    
    plt.savefig(PROJECT_ROOT / "plot/supp_c3_right.png", dpi=150, transparent=True)

if __name__ == "__main__":
    main()
