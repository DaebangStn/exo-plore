import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from python.plot.util import *
from python.analysis.comparison import compare_time_series, print_comparison_table, print_latex_table
import polars as pl

################ CONFIG ################
EXP_NAMEs = [
'exo_hei_resist_21000_0807_131406+memo_selected_k2_d3_no_mod_state_gems_ma15+_on_0808_221848',
]
used_vel = np.arange(2, 5.1, 1)
used_cycle = 8
moment_yticks = np.arange(-6, 6.1, 2)
MA_WINDOW = 15
GEMS_DATA_ROOT = "experiment_data/real/gems"

# Create viridis colormap for velocity values
velocity_colormap = cm.viridis
velocity_norm = plt.Normalize(vmin=min(used_vel), vmax=max(used_vel))

yticks = {
    'angle': np.arange(-60, 20.1, 20),
    'vel': np.arange(-6, 6.1, 4),
    'moment': np.arange(-6, 6.1, 6),
    'power': np.arange(-5, 25.1, 5),
}


def load_real_data():
    return {
        'angle': load_angle_gems(),
        'vel': load_vel_gems(),
        'moment': load_moment_gems(),
        'power': load_power_gems(),
    }


def plot_real_data(axes, real_data):
    ax_ang, ax_vel, ax_moment, ax_power = axes
    angle_data = real_data['angle']
    vel_data = real_data['vel']
    moment_data = real_data['moment']
    power_data = real_data['power']

    for v in used_vel:
        color = velocity_colormap(velocity_norm(v))
        ax_ang.plot(angle_data[v]["gait_cycle"], angle_data[v]["value"], color=color, linewidth=2)
        ax_vel.plot(vel_data[v]["gait_cycle"], vel_data[v]["value"], color=color, linewidth=2)
        ax_moment.plot(moment_data[v]["gait_cycle"], moment_data[v]["value"], color=color, linewidth=2)
        ax_power.plot(power_data[v]["gait_cycle"], power_data[v]["value"], color=color, linewidth=2)
    ax_ang.set_title("Hip flexion angle (°)", fontsize=FONT_SIZE_LABEL, pad=16)
    ax_vel.set_title("Hip flexion velocity (°/s)", fontsize=FONT_SIZE_LABEL, pad=16)
    ax_moment.set_title("Exo assist moment (Nm)", fontsize=FONT_SIZE_LABEL, pad=16)
    ax_power.set_title("Exo assist power (W)", fontsize=FONT_SIZE_LABEL, pad=16)
    ax_ang.set_ylim(min(yticks['angle']), max(yticks['angle']))
    ax_ang.set_yticks(yticks['angle'])
    ax_vel.set_ylim(min(yticks['vel']), max(yticks['vel']))
    ax_vel.set_yticks(yticks['vel'])
    ax_moment.set_ylim(min(yticks['moment']), max(yticks['moment']))
    ax_moment.set_yticks(yticks['moment'])
    ax_power.set_ylim(min(yticks['power']), max(yticks['power']))
    ax_power.set_yticks(yticks['power'])


def load_sim_data():
    assert len(EXP_NAMEs) == 1, f"Only one experiment is supported, but got {len(EXP_NAMEs)}"
    exp_name = EXP_NAMEs[0]
    sim_data, sim_param = data_and_param_path(exp_name, read_file=True)
    sim_param = sim_param.with_columns((pl.col("Stride") * pl.col("Phase") * REF_VELOCITY * 3.6).alias("velocity")).collect()
    velocities = sim_param["velocity"]
    assert np.allclose(velocities, used_vel, atol=1e-2), f"The velocities are not the same, got {velocities} and {used_vel}"

    sim_param = sim_param.sort("velocity")
    param_indices = sim_param["param_idx"]

    data_dict = {}
    for i in range(len(used_vel)):
        param_data = sim_data.filter(pl.col("param_idx") == param_indices[i]).filter(pl.col("cycle") == used_cycle).collect()
        data_dict[used_vel[i]] = {
            'angle': moving_average(param_data["angle_HipR"], MA_WINDOW),
            'vel': moving_average(param_data["velocity_HipR"] * np.pi / 180, MA_WINDOW),
            'moment': moving_average(param_data["dev_moment_HipR"], MA_WINDOW),
            'power': moving_average(param_data["dev_power_HipR"], MA_WINDOW),
        }
    return data_dict


def plot_sim_data(axes, sim_data_dict):
    ax_ang, ax_vel, ax_moment, ax_power = axes

    for i, v in enumerate(used_vel):
        sim_data = sim_data_dict[v]
        angle = sim_data['angle']
        vel = sim_data['vel']
        moment = sim_data['moment']
        power = sim_data['power']

        xdata = np.linspace(0, 100, len(angle), endpoint=True)
        color = velocity_colormap(velocity_norm(used_vel[i]))
        ax_ang.plot(xdata, angle, color=color, linewidth=2)
        ax_vel.plot(xdata, vel, color=color, linewidth=2)
        ax_moment.plot(xdata, moment, color=color, linewidth=2)
        ax_power.plot(xdata, power, color=color, linewidth=2)

    ax_ang.set_ylim(min(yticks['angle']), max(yticks['angle']))
    ax_ang.set_yticks(yticks['angle'])
    ax_vel.set_ylim(min(yticks['vel']), max(yticks['vel']))
    ax_vel.set_yticks(yticks['vel'])
    ax_moment.set_ylim(min(yticks['moment']), max(yticks['moment']))
    ax_moment.set_yticks(yticks['moment'])
    ax_power.set_ylim(min(yticks['power']), max(yticks['power']))
    ax_power.set_yticks(yticks['power'])


def main():
    fig = plt.figure(figsize=(15, 5))
    fig.subplots_adjust(left=0.03, right=1.08, top=0.9, bottom=0.15, wspace=0.2, hspace=0.2)
    gs = gridspec.GridSpec(2, 5, height_ratios=[1, 1], width_ratios=[1, 2, 2, 2, 2])
    ax_fig1 = plt.subplot(gs[0, 0])
    ax_fig2 = plt.subplot(gs[1, 0])
    ax_real1 = plt.subplot(gs[0, 1])
    ax_real2 = plt.subplot(gs[0, 2])
    ax_real3 = plt.subplot(gs[0, 3])
    ax_real4 = plt.subplot(gs[0, 4])
    ax_sim1 = plt.subplot(gs[1, 1])
    ax_sim2 = plt.subplot(gs[1, 2])
    ax_sim3 = plt.subplot(gs[1, 3])
    ax_sim4 = plt.subplot(gs[1, 4])
    axes_sim = [ax_sim1, ax_sim2, ax_sim3, ax_sim4]
    axes_real = [ax_real1, ax_real2, ax_real3, ax_real4]
    for ax in fig.get_axes():
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
    for ax in axes_real + axes_sim:
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    fig.supxlabel("Gait cycle (%)", fontsize=FONT_SIZE_LABEL)
    insert_image(ax_fig1, "data/figs/human_subject.png", "Human subject", fontsize=FONT_SIZE_LABEL)
    insert_image(ax_fig2, "data/figs/simulation.png", "Simulation", fontsize=FONT_SIZE_LABEL)

    real_data = load_real_data()
    sim_data = load_sim_data()

    plot_real_data(axes_real, real_data)
    plot_sim_data(axes_sim, sim_data)

    # --- Comparison ---
    comparison_results = []
    data_types = ['angle', 'vel', 'moment', 'power']
    for v in used_vel:
        for data_type in data_types:
            x1 = real_data[data_type][v]["gait_cycle"].to_numpy()
            y1 = real_data[data_type][v]["value"].to_numpy()
            y2 = sim_data[v][data_type]
            x2 = np.linspace(0, 100, len(y2), endpoint=True)
            results = compare_time_series(x1, y1, x2, y2, name=f"{data_type}_v{v}")
            comparison_results.append(results)

    print("--- Time Series Comparison ---")
    print_comparison_table(comparison_results)
    print("------------------------------")
    print_latex_table(comparison_results)
    # --------------------

    # Add colorbar for velocity values
    sm = plt.cm.ScalarMappable(cmap=velocity_colormap, norm=velocity_norm)
    sm.set_array([])
    from matplotlib.ticker import MaxNLocator
    cbar = fig.colorbar(sm, ax=axes_real + axes_sim, location='right', shrink=0.8, pad=0.02)
    cbar.locator = MaxNLocator(nbins=3)
    cbar.update_ticks()
    cbar.set_label('Walking speed (km/h)', fontsize=FONT_SIZE_LABEL-4)
    plt.savefig(f"plot/figure5.png", dpi=100, transparent=True)


if __name__ == "__main__":
    main()
