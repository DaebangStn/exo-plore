import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from python.plot.util import data_and_param_path
import polars as pl
from scipy import interpolate
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple


EXP_NAMES = [
'no_exo_meta_ma15_10000_0801_111350+memo_b20_ma15_10000_0801_111350_pf_param_grf+_on_1114_160356',
'no_exo_meta_ma15_10000_0801_111421+memo_b20_ma15_10000_0801_111421_pf_param_grf+_on_1114_160404',
'no_exo_meta_ma15_10000_0801_205540+memo_b20_ma15_10000_0801_205540_pf_param_grf+_on_1114_160413',
'no_exo_meta_ma15_10000_0802_064456+memo_b20_ma15_10000_0802_064456_pf_param_grf+_on_1114_160424',
'no_exo_meta_ma15_10000_0802_163808+memo_b20_ma15_10000_0802_163808_pf_param_grf+_on_1114_160434',
'no_exo_meta_ma15_10000_0805_215352+memo_b20_ma15_10000_0805_215352_pf_param_grf+_on_1114_160445',
'no_exo_meta_ma15_10000_0805_215354+memo_b20_ma15_10000_0805_215354_pf_param_grf+_on_1114_160455',
'no_exo_meta_ma15_10000_0805_215359+memo_b20_ma15_10000_0805_215359_pf_param_grf+_on_1114_160504',
'no_exo_meta_ma15_10000_0806_131413+memo_b20_ma15_10000_0806_131413_pf_param_grf+_on_1114_160514',
'no_exo_meta_ma15_10000_0806_131416+memo_b20_ma15_10000_0806_131416_pf_param_grf+_on_1114_160525',
'no_exo_meta_ma15_10000_0806_131419+memo_b20_ma15_10000_0806_131419_pf_param_grf+_on_1114_160536',
]

PLOT_CONFIG = {
    'AP': {'ylim': (-0.3, 0.5), 'title': 'Anterior-Posterior'},
    'ML': {'ylim': (-0.1, 0.1), 'title': 'Medial-Lateral'},
    'V': {'ylim': (0, 1.6), 'title': 'Vertical'},
    'common': {
        'xlabel': 'Stance (%)',
        'ylabel': 'BW (%)',
        'title_fontsize': 24,
        'label_fontsize': 20,
        'legend_fontsize': 20,
        'tick_fontsize': 15
    },
    'moving_average_window': 20,
    'zero_out_points': {
        'grf_r_x': 7,
        'grf_r_y': 1,
        'grf_r_z': 1
    },
    'FIGSIZE': (5, 5),
}

def process_sim_data(grf_cols: list):
    moving_average_window = PLOT_CONFIG['moving_average_window']
    target_gait_cycle = np.linspace(0, 100, 1000)  # 1000 intervals
    all_normalized_trajectories = {col: [] for col in grf_cols}

    for exp_name in EXP_NAMES:
        sim_data_lf, sim_param_lf = data_and_param_path(exp_name, read_file=True)

        # Filter out prerun and last cycle (following preprocessing patterns)
        sim_data_lf = (sim_data_lf
                       .filter(pl.col("cycle") >= 5)  # Remove first 5 cycles
                       .filter(pl.col("cycle") < pl.max("cycle")))  # Remove last cycle

        sim_data_lf = sim_data_lf.filter(pl.col("contact_R") == 1)

        # Get unique parameter indices
        param_indices = sim_param_lf.collect()["param_idx"].unique().sort()

        # Create progress bar for all parameters across all experiments
        with tqdm(total=len(param_indices), desc=f"Processing {exp_name}", ncols=100) as pbar:
            for param_idx in param_indices:
                pbar.update(1)
                param_data = (sim_data_lf
                             .filter(pl.col("param_idx") == param_idx)
                             .sort(["cycle", "time"])
                             .collect())

                if param_data.is_empty():
                    continue

                cycles = param_data["cycle"].unique().sort()
                for cycle_num in cycles:
                    cycle_data = param_data.filter(pl.col("cycle") == cycle_num)

                    if len(cycle_data) < 10:  # Skip cycles with too few data points
                        print(f"Warning: Skipping cycle {cycle_num} with too few data points")
                        continue

                    for col in grf_cols:
                        if col not in cycle_data.columns:
                            continue

                        grf_vals = cycle_data[col].to_numpy().copy()

                        # 1. Zero out
                        num_zero_out = PLOT_CONFIG['zero_out_points'].get(col, 0)
                        if num_zero_out > 0:
                            grf_vals[:num_zero_out] = 0

                        # 2. Smooth data with moving average
                        if len(grf_vals) >= moving_average_window:
                            grf_vals = pd.Series(grf_vals).rolling(window=moving_average_window, min_periods=1).mean().to_numpy()


                        original_time = np.linspace(0, 100, len(grf_vals))

                        if len(grf_vals) > 1:
                            try:
                                # 3. Interpolate
                                f = interpolate.interp1d(original_time, grf_vals,
                                                       kind='linear', bounds_error=False,
                                                       fill_value='extrapolate')
                                normalized_grf = f(target_gait_cycle)
                                all_normalized_trajectories[col].append(normalized_grf)
                            except Exception as e:
                                print(f"Warning: Interpolation failed for {exp_name}, param {param_idx}, cycle {cycle_num}, col {col}: {e}")
                                continue

    processed_data = {}
    for col in grf_cols:
        if all_normalized_trajectories[col]:
            # 4. Average and compute std
            trajectory_array = np.array(all_normalized_trajectories[col])
            mean_grf = np.mean(trajectory_array, axis=0)
            std_grf = np.std(trajectory_array, axis=0)
            processed_data[col] = {"mean": mean_grf, "std": std_grf, "time": target_gait_cycle}
        else:
            print(f"Warning: No valid trajectories found for {col}")

    return processed_data


def plot_grf(show=True):
    # Group files by measurement type
    file_groups = {
        'V': ['GRF_F_V_PRO_left.csv', 'GRF_F_V_PRO_right.csv'],
        'ML': ['GRF_F_ML_PRO_left.csv', 'GRF_F_ML_PRO_right.csv'],
        'AP': ['GRF_F_AP_PRO_left.csv', 'GRF_F_AP_PRO_right.csv'],
    }
    
    sim_grf_cols_map = {'grf_r_z': 'AP', 'grf_r_x': 'ML', 'grf_r_y': 'V'}
    sim_grf_cols = list(sim_grf_cols_map.keys())

    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(file_groups), figsize=(PLOT_CONFIG['FIGSIZE'][0] * len(file_groups), PLOT_CONFIG['FIGSIZE'][1]), sharex=True, sharey=False)
    if len(file_groups) == 1:
        axes = [axes]
    fig.tight_layout(pad=5.0)
    
    axes_map = {name: axes[i] for i, name in enumerate(file_groups.keys())}

    # Plot real data
    base_path = Path('experiment_data/real/gutenberg')
    common_config = PLOT_CONFIG.get('common', {}) # Get common config here
    for i, (measurement, files) in enumerate(file_groups.items()):
        ax = axes[i]
        all_data = []
        for file_name in files:
            file_path = base_path / file_name
            if not file_path.exists():
                print(f"File not found: {file_path}")
                continue
            df = pd.read_csv(file_path)
            all_data.append(df)

        if not all_data:
            continue

        combined_df = pd.concat(all_data, ignore_index=True)
        data_columns = combined_df.columns[4:]
        mean_data = combined_df[data_columns].mean(axis=0)
        std_data = combined_df[data_columns].std(axis=0)
        time_points = np.arange(1, len(mean_data) + 1)
        
        ax.plot(time_points, mean_data, color='gray')
        ax.fill_between(time_points, mean_data - std_data, mean_data + std_data, alpha=0.2, color='gray')
        
        config = PLOT_CONFIG.get(measurement, {})
        ax.set_title(config.get('title', f'GRF {measurement}'), fontsize=common_config.get('title_fontsize'))
        
        if 'ylim' in config:
            ax.set_ylim(config['ylim'])
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis='x', labelsize=common_config.get('tick_fontsize'))
        ax.tick_params(axis='y', labelsize=common_config.get('tick_fontsize'))

    # Process and plot simulation data
    processed_sim_data = process_sim_data(sim_grf_cols)

    for sim_col, measurement in sim_grf_cols_map.items():
        ax = axes_map[measurement]
        if sim_col in processed_sim_data:
            data = processed_sim_data[sim_col]
            ax.plot(data["time"], data["mean"], color='red')
            ax.fill_between(data["time"], data["mean"] - data["std"], data["mean"] + data["std"], alpha=0.2, color='red')

    # Shared labels and legend
    if common_config.get('xlabel'):
        fig.supxlabel(common_config.get('xlabel'), fontsize=common_config.get('label_fontsize'))
    if common_config.get('ylabel'):
        fig.supylabel(common_config.get('ylabel'), fontsize=common_config.get('label_fontsize'))

    axes[0].set_xticks([0, 50, 100])

    sim_line = mlines.Line2D([], [], color='red', linewidth=2)
    sim_patch = Patch(facecolor='red', alpha=0.2)
    human_line = mlines.Line2D([], [], color='gray', linewidth=2)
    human_patch = Patch(facecolor='gray', alpha=0.2)

    axes[-1].legend(handles=[(human_line, human_patch), (sim_line, sim_patch)],
                   labels=['Human exp.', 'Simulation'],
                #    handler_map={tuple: HandlerTuple(ndivide=None)},
                   loc='upper left',
                   fontsize=common_config.get('legend_fontsize'))

    # Save the plot
    plt.savefig('plot/supp_m9.png')
    print("Plot saved to plot/supp_m9.png")


def main():
    plot_grf(show=True)

if __name__ == "__main__":
    main()
