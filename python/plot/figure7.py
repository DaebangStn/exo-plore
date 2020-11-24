# Comparison between the model prediction and the ground truth
from python.plot.util import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from collections import defaultdict
import statistics


# Data structure: Dict[label: str] where string is comma-separated "ckpt_name, mrr, velocity" format
Data = {
 r'$r_{HEI}$': """
exo_hei_resist_0807_131406_21000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0907_204001, 20.2, 5.3
exo_hei_resist_0807_131406_17000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0810_100841, 22.4, 6.4
exo_hei_resist_0807_131406_18000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0810_100846, 22.9, 6.4
exo_hei_resist_0807_131406_19000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0810_100854, 14.7, 5.4
exo_hei_resist_0807_131406_20000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0810_100900, 25.4, 4.8
exo_hei_resist_0807_131406_22000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0812_235811, 25.5, 4.6
exo_hei_resist_0807_131406_23000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0812_235825, 23.3, 4.6

exo_hei_resist_0807_131416_17000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0810_100908, 15.8, 5.8
exo_hei_resist_0807_131416_18000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0810_100914, 28.5, 7.8
exo_hei_resist_0807_131416_19000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0810_100923, 19.7, 1.4
exo_hei_resist_0807_131416_20000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0810_100930, 20.3, 3.4
exo_hei_resist_0807_131416_22000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0812_235839, 25.8, 5.2
exo_hei_resist_0807_131416_23000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0812_235852, 20.2, 5.0

exo_hei_resist_0807_131424_17000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0810_100938, 21.7, 7.6
exo_hei_resist_0807_131424_18000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0810_100942, 16.7, 3.8
exo_hei_resist_0807_131424_19000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0810_100952, 16.0, 3.8
exo_hei_resist_0807_131424_20000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0810_100955, 26.0, 7.6
exo_hei_resist_0807_131424_22000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0812_235908, 26.7, 4.2
exo_hei_resist_0807_131424_23000+memo__b20df_U40960_v2_no_mod_state_ma15+_on_0812_235922, 10.0, 7.8
""",
r"$r_{assist}$": """
exo_hei_assist_0809_211111_16000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_011035, 17.2, 6.7, 7.644
exo_hei_assist_0809_211111_18000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_164415, 9.1, 7.4, 8.093
exo_hei_assist_0809_211111_20000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_164425, 10.3, 6.5, 7.228
exo_hei_assist_0809_211111_21000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_011048, 10.5, 7.0, 7.017
exo_hei_assist_0809_211111_23000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_164429, 14.0, 7.3, 7.135
exo_hei_assist_0809_211111_24000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_011101, 9.4, 5.5, 6.761

exo_hei_assist_0923_162723_16000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_223817, 8.9, 4.3, 7.903
exo_hei_assist_0923_162723_18000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_223831, 9.1, 2.0, 7.770
exo_hei_assist_0923_162723_20000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_223843, 13.6, 2.4, 8.594
exo_hei_assist_0923_162723_21000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_223856, 14.7, 1.2, 11.619
exo_hei_assist_0923_162723_23000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_223909, 13.0, 1.5, 14.419
exo_hei_assist_0923_162723_24000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_223922, 19.0, 1.1, 16.861

exo_hei_assist_0923_180438_16000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_154523, 13.8, 1.7, 8.443
exo_hei_assist_0923_180438_18000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_154523, 13.1, 1.9, 9.755
exo_hei_assist_0923_180438_20000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_154523, 8.2, 7.2, 5.577
exo_hei_assist_0923_180438_21000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_154523, 6.5, 2.1, 4.216
exo_hei_assist_0923_180438_23000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_154523, 9.7, 1.2, 6.569
exo_hei_assist_0923_180438_24000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_154523, 8.2, 2.4, 4.976

exo_hei_assist_0923_192733_16000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_223934, 20.6, 1.6, 17.516
exo_hei_assist_0923_192733_18000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_223947, 14.8, 1.5, 18.484
exo_hei_assist_0923_192733_20000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_224000, 18.3, 2.1, 18.558
exo_hei_assist_0923_192733_21000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_224013, 13.5, 1.6, 15.271
exo_hei_assist_0923_192733_23000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_224025, 18.7, 1.2, 20.930
exo_hei_assist_0923_192733_24000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_224039, 18.3, 1.4, 20.324

""",
r"no $r_{HEI}$": """
exo_no_hei2_0903_111108_16000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_002600, 11.2, 1.3, 7.839
exo_no_hei2_0903_111108_18000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_024712, 7.9, 6.6, 4.520
exo_no_hei2_0903_111108_20000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_024725, 12.3, 1.5, 8.389
exo_no_hei2_0903_111108_21000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_002613, 13.6, 4.9, 4.966
exo_no_hei2_0903_111108_23000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_024739, 11.2, 4.5, 7.114
exo_no_hei2_0903_111108_24000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_002627, 9.8, 4.4, 4.950

exo_no_hei2_0904_093129_16000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_002640, 8.2, 2.2, 6.059
exo_no_hei2_0904_093129_18000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_024751, 14.3, 2.5, 6.299
exo_no_hei2_0904_093129_20000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_024804, 11.0, 2.4, 6.007
exo_no_hei2_0904_093129_21000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_002654, 8.0, 2.6, 5.248
exo_no_hei2_0904_093129_23000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_024819, 7.4, 2.2, 3.918
exo_no_hei2_0904_093129_24000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_002707, 9.6, 2.1, 6.346

exo_no_hei2_0905_081123_16000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_114304, 9.9, 6.9, 7.017
exo_no_hei2_0905_081123_18000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_024832, 10.5, 7.5, 6.940
exo_no_hei2_0905_081123_20000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_024844, 12.4, 1.2, 7.616
exo_no_hei2_0905_081123_21000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_114320, 8.1, 1.9, 5.607
exo_no_hei2_0905_081123_23000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0924_024857, 9.0, 8.1, 8.099
exo_no_hei2_0905_081123_24000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_114334, 7.7, 5.0, 4.867

exo_hei_resist_err_0805_215340_16000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_164241, 9.5, 5.9, 6.915
exo_hei_resist_err_0805_215340_18000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_164256, 16.9, 1.8, 7.376
exo_hei_resist_err_0805_215340_21000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_164311, 10.6, 7.0, 7.132
exo_hei_resist_err_0805_215340_24000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_164325, 9.1, 6.2, 6.042

exo_no_hei2_dn1_2_err_0806_120509_16000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_164340, 10.0, 7.7, 7.103
exo_no_hei2_dn1_2_err_0806_120509_18000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_164354, 11.4, 1.1, 7.571
exo_no_hei2_dn1_2_err_0806_120509_21000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_164402, 9.3, 7.2, 6.164
exo_no_hei2_dn1_2_err_0806_120509_24000+memo__exo_no_hei2_U58500_no_mod_state_ma15+_on_0923_164410, 17.1, 1.3, 9.605
""",
# "": """
# """,
# "": """
# """,
# "": """
# """,
}

# Human reference data
Human = {
    'mean': 19.8,
    'std': 2.9,
}


def parse_csv_data(csv_string):
    """Parse comma-separated string data into list of tuples."""
    parsed_data = []
    lines = csv_string.strip().split('\n')

    for line in lines:
        if line.strip():  # Skip empty lines
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 3:
                ckpt_name = parts[0]
                mrr = float(parts[1])
                velocity = float(parts[2])
                parsed_data.append((ckpt_name, mrr, velocity))

    return parsed_data


def calculate_statistics(data_dict):
    """Calculate mean and standard deviation for each label in the data."""
    stats = {}

    for label, data_string in data_dict.items():
        if not data_string:
            continue

        # Parse the comma-separated string data
        parsed_values = parse_csv_data(data_string)

        if not parsed_values:
            continue

        # Extract MRR values (index 1 in tuple)
        mrr_values = [item[1] for item in parsed_values]
        velocities = [item[2] for item in parsed_values]

        stats[label] = {
            'mrr_mean': statistics.mean(mrr_values),
            'mrr_std': statistics.stdev(mrr_values) if len(mrr_values) > 1 else 0.0,
            'velocity_mean': statistics.mean(velocities),
            'velocity_std': statistics.stdev(velocities) if len(velocities) > 1 else 0.0,
            'count': len(mrr_values),
            'raw_mrr': mrr_values,
            'raw_velocity': velocities
        }

    return stats


def plot_mrr_comparison(data_dict, human_ref, save_path=None, title="MRR Comparison"):
    """
    Plot bar chart comparing MRR values across different models with error bars.

    Args:
        data_dict: Dictionary with model labels and their data points
        human_ref: Dictionary with human reference mean and std
        save_path: Optional path to save the plot
        title: Plot title
    """



def main():
    """Main function to demonstrate the MRR bar plot."""
    print("Generating MRR comparison bar plot...")

    # Calculate statistics for all models
    stats = calculate_statistics(Data)

    if not stats:
        print("No data available for plotting")
        return

    # Prepare data for plotting
    labels = list(stats.keys())
    mrr_means = [stats[label]['mrr_mean'] for label in labels]
    mrr_stds = [stats[label]['mrr_std'] for label in labels]
    counts = [stats[label]['count'] for label in labels]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6.5))
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.85)

    # Create bar positions
    x_pos = np.arange(len(labels))
    bar_width = 0.6

    # Plot model bars with error bars using viridis colormap
    viridis = plt.colormaps['viridis']
    colors = [viridis((len(labels) - i - 1) / (len(labels) - 1)) for i in range(len(labels))]
    bars = ax.bar(x_pos, mrr_means, bar_width,
                  yerr=mrr_stds, capsize=5,
                  color=colors,
                  alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add human reference line
    human_mean = Human['mean']
    human_std = Human['std']

    # Plot human reference as horizontal line with shaded error region
    ax.axhline(human_mean, color='gray', linestyle='--', linewidth=3,
               label=f'Human exp.', zorder=10)
    ax.axhspan(human_mean - human_std, human_mean + human_std,
               color='gray', alpha=0.2, zorder=1)

    # Customize the plot
    ax.set_title("Maximum\nmetabolic reduction rate (%)", fontsize=FONT_SIZE_LABEL+5, pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 30)

    # Add grid for better readability
    ax.grid(True, axis='y')
    ax.set_axisbelow(True)

    # Create custom legend with Line2D for human reference and Rectangle for colored model results
    human_line = Line2D([], [], color='gray', linestyle='--', linewidth=3, label='Human exp.')
    # model_bars = mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='(Colored) Simulation')
    lg = ax.legend(handles=[human_line], fontsize=FONT_SIZE_LABEL, loc='upper center')
    lg.get_frame().set_alpha(0.6)
    plt.savefig(PLOT_SAVE_DIR / "figure7.png", dpi=100, transparent=True)

    # Display plot
    
    

if __name__ == "__main__":
    main()
