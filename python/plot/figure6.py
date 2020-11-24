from python.surrogate.preprocess import mean_negpos_field, rms_field
from python.plot.util import *
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
################ CONFIG ################
EXP_NAMEs = [
    [
'exo_hei_resist_21000_0807_131406+memo_selected_k2_d1_no_mod_state_gems_ma15+_on_0808_221826',
'exo_hei_resist_21000_0807_131406+memo_selected_k2_d2_no_mod_state_gems_ma15+_on_0808_221836',
'exo_hei_resist_21000_0807_131406+memo_selected_k2_d3_no_mod_state_gems_ma15+_on_0808_221848',
'exo_hei_resist_21000_0807_131406+memo_selected_k2_d4_no_mod_state_gems_ma15+_on_0808_221859',
    ],
    [
'exo_hei_assist_24000_0809_211111+memo_selected_k2_d1_no_mod_state_ma15_gems+_on_0811_125350',
'exo_hei_assist_24000_0809_211111+memo_selected_k2_d2_no_mod_state_ma15_gems+_on_0811_125402',
'exo_hei_assist_24000_0809_211111+memo_selected_k2_d3_no_mod_state_ma15_gems+_on_0811_125414',
'exo_hei_assist_24000_0809_211111+memo_selected_k2_d4_no_mod_state_ma15_gems+_on_0811_125427',
    ],
    [
'exo_no_hei_18000_0721_102502+memo_selected_k2_d1_no_mod_state_gems_on_0725_093155',
'exo_no_hei_18000_0721_102502+memo_selected_k2_d2_no_mod_state_gems_on_0725_093207',
'exo_no_hei_18000_0721_102502+memo_selected_k2_d3_no_mod_state_gems_on_0725_093217',
'exo_no_hei_18000_0721_102502+memo_selected_k2_d4_no_mod_state_gems_on_0725_093226',
    ],
]
LABELs = [
    r"$r_{HEI}$",
    r"$r_{assist}$",
    r"no $r_{HEI}$",
]

########################################
velocities = np.array([2, 3, 4, 5])
real_delays = np.array([0.05, 0.15, 0.25, 0.35])

momentRange = (0, 4)
pos_powerRange = (0, 8)
neg_powerRange = (0, 4)
moment_yticks = np.arange(momentRange[0], momentRange[1] + 0.1, 1)
pos_power_yticks = np.arange(pos_powerRange[0], pos_powerRange[1] + 0.1, 2)
neg_power_yticks = np.arange(neg_powerRange[0], neg_powerRange[1] + 0.1, 1)
bar_width = 0.2
LINE_WIDTH = 4
GROUP_COLORS = [
    to_01color(253, 231, 37),
    to_01color(33, 145, 140),
    to_01color(68, 1, 84),
]
black_colors =  ['#404040', '#707070', '#afafaf', '#cfcfcf']


def plot_real_data(axes):
    power_real_data = load_power_w_delay_gems()
    moment_real_data = load_moment_w_delay_gems()
    ax_moment = axes[0]
    ax_power_pos = axes[1]
    ax_power_neg = axes[2]
    for idx, delay in enumerate(real_delays):
        pos_power = power_real_data[delay]['positive'][1:1+len(velocities)]
        neg_power = power_real_data[delay]['negative'][1:1+len(velocities)]
        # neg_power = list(map(lambda x: -x, neg_power))
        moment = moment_real_data[delay][1:1+len(velocities)]
        bar_pos = (idx - 1.5) * bar_width
        ax_power_pos.bar(velocities + bar_pos, pos_power, width=bar_width, color=black_colors[idx])
        ax_power_neg.bar(velocities + bar_pos, neg_power, width=bar_width, color=black_colors[idx])
        ax_moment.bar(velocities + bar_pos, moment, width=bar_width, color=black_colors[idx], label='Human subject')
    ax_power_pos.set_title("Mean assist power (W)", fontsize=FONT_SIZE_LABEL + 3)
    ax_power_neg.set_title("Mean resist power (W)", fontsize=FONT_SIZE_LABEL + 3)
    ax_moment.set_title("RMS assist moment (Nm)", fontsize=FONT_SIZE_LABEL + 3)
    fig = plt.gcf()
    fig.supxlabel("Walking speed (km/h)", fontsize=FONT_SIZE_LABEL + 3)
    ax_power_pos.set_xticks(velocities)
    ax_power_neg.set_xticks(velocities)
    ax_moment.set_xticks(velocities)
    ax_power_pos.set_ylim(pos_powerRange[0], pos_powerRange[1])
    ax_power_neg.set_ylim(neg_powerRange[0], neg_powerRange[1])
    ax_moment.set_ylim(momentRange[0], momentRange[1])
    ax_power_pos.set_yticks(pos_power_yticks)
    ax_power_neg.set_yticks(neg_power_yticks)
    ax_moment.set_yticks(moment_yticks)
    ax_power_pos.grid(True, axis='y')
    ax_power_neg.grid(True, axis='y')
    ax_moment.grid(True, axis='y')
    
    axes_center = velocities[0]
    ax_delay = ax_moment.twiny()
    ax_delay.set_xlim(ax_moment.get_xlim())
    ax_delay.set_xticks([2 - 1.5 * bar_width, 2, 2 + 1.5 * bar_width])
    ax_delay.set_xticklabels([0.05, 0.2, 0.35])
    ax_delay.xaxis.set_label_position('top')
    ax_delay.set_xlabel(r"delay ($\Delta t$)", fontsize=FONT_SIZE_LABEL)
    ax_delay.xaxis.set_label_coords(0.15, 0.9)
    ax_delay.spines['top'].set_position(('outward', -50))
    ax_delay.spines['top'].set_bounds(2 - 1.5 * bar_width, 2 + 1.5 * bar_width)
    ax_delay.tick_params(axis='x',
                        direction='in',
                        length=8,
                        labelsize=FONT_SIZE_LABEL-6,
                        pad=4)
    ax_delay.spines[['right', 'bottom', 'left']].set_visible(False)


def regroup_delay_velocity(exp_names: List[str]) -> Tuple[
    Dict[float, Tuple[List[float], List[float], List[float], List[float]]], # velocity -> (delay, moment, power_pos, power_neg)
    ]:
    """
        exp_names: list of experiment names (in the same group)
    """    
    data = {velocity: ([], [], [], []) for velocity in velocities}
    for exp_name in exp_names:
        sim_data, sim_param = data_and_param_path(exp_name, read_file=True)
        _power_data = mean_negpos_field(sim_data, sim_param, 'dev_power_HipR').with_columns(
            (pl.col("Stride") * pl.col("Phase") * REF_CADENCE * REF_STEP * 3.6).round(2).alias("velocity")
        ).sort("velocity").collect()
        _moment_data = rms_field(sim_data, sim_param, "dev_moment_HipR").with_columns(
            (pl.col("Stride") * pl.col("Phase") * REF_CADENCE * REF_STEP * 3.6).round(2).alias("velocity")
        ).sort("velocity").collect()
        
        delay_from_data = np.unique(_power_data['Delay'])
        assert len(delay_from_data) == 1, f"There are multiple delays in the data: {delay_from_data} | {exp_name}"
        delay_from_data = delay_from_data.item()
        for velocity in velocities:
            _delay, _moment, _power_pos, _power_neg = data[velocity]
            _delay.append(delay_from_data)
            _moment.append(_moment_data.filter(pl.col("velocity") == velocity)["moment"].item())
            _power_pos.append(_power_data.filter(pl.col("velocity") == velocity)["power_pos"].item())
            _power_neg.append(-_power_data.filter(pl.col("velocity") == velocity)["power_neg"].item())
            data[velocity] = (_delay, _moment, _power_pos, _power_neg)
            
    return data

def plot_delay_power_moment(axes, group_idx: int):
    exp_names = EXP_NAMEs[group_idx]
    data = regroup_delay_velocity(exp_names)
    group_color = GROUP_COLORS[group_idx]
    
    power_real_data_relist = {}
    power_real_data = load_power_w_delay_gems()
    for v in velocities:
        power_velocity_data = []
        for delay in power_real_data.keys():
            power_velocity_data.append(power_real_data[delay]['negative'][v - 1])
        power_real_data_relist[v] = power_velocity_data
    
    mean_neg_power_ratio = []
    
    ax_moment = axes[0]
    ax_power_pos = axes[1]
    ax_power_neg = axes[2]
    for velocity in velocities:
        _delay, _moment, _power_pos, _power_neg = data[velocity]
        _delay = [(d - 0.2) * bar_width / 0.1 + velocity for d in _delay]
        ax_moment.plot(_delay, _moment, color=group_color, linewidth=LINE_WIDTH, zorder=(10-group_idx))
        ax_power_pos.plot(_delay, _power_pos, color=group_color, linewidth=LINE_WIDTH, zorder=(10-group_idx))
        ax_power_neg.plot(_delay, _power_neg, color=group_color, linewidth=LINE_WIDTH, zorder=(10-group_idx), label=LABELs[group_idx] if velocity == velocities[0] else None)
        
        real_neg_power = power_real_data_relist[velocity]
        for i in range(len(_power_neg)):
            mean_neg_power_ratio.append(_power_neg[i] / real_neg_power[i])
              
    print(LABELs[group_idx])
    print(np.mean(mean_neg_power_ratio).item())
    # _, _moment4, _power_pos4, _power_neg4 = data[4]
    # print('moment', _moment4)
    # print('power_pos', _power_pos4)
    
def main():
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.15, wspace=0.1)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    axes = [ax0, ax1, ax2]
    
    for ax in axes:
        ax.tick_params(axis='both', direction='in', length=8, pad=4)
    
    assert len(GROUP_COLORS) == len(EXP_NAMEs), f"The number of group colors and experiments must be the same. Found {len(GROUP_COLORS)} group colors and {len(EXP_NAMEs)} experiments."
    assert len(EXP_NAMEs) == len(LABELs), f"The number of experiments and labels must be the same. Found {len(EXP_NAMEs)} experiments and {len(LABELs)} labels."
    plot_real_data(axes)
    for group_idx in range(len(EXP_NAMEs)):
        plot_delay_power_moment(axes, group_idx)

    fig = plt.gcf()
    ref_patch = mpatches.Patch(facecolor='gray', edgecolor='gray')
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(ref_patch)
    labels.append('Human exp.')
    lg = ax2.legend(handles, labels, loc='upper left', title="HEI rewards", fontsize=FONT_SIZE_LABEL-3, title_fontsize=FONT_SIZE_LABEL)
    lg.get_frame().set_alpha(0.3)
    plt.savefig(f"plot/figure6.png", dpi=175, transparent=True)

if __name__ == "__main__":
    main()
