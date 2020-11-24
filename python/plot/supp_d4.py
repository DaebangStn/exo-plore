from python.plot.util import *
from scipy.stats import bootstrap


LOG_SCALE = True
MARK_CHOSEN = True
# --- YOUR RAW DATA HERE ---
# This dictionary must contain all (mass, act) combinations you want to plot.
# Each value is a space-separated string of Z_Values for that specific (mass, act) condition.

LABEL = 'pws_10k'
data_inputs_strings = {
    (0, 1.25): "1.932	1.912",
    (0.5, 1.25): "1.621	1.861",
    (2, 1.25): "1.826",
    (1, 1.25): "1.891	1.579",

    (0, 1):   "2.023	1.852	1.932	2.052	1.986",
    (0, 1.5): "1.211	1.157	1.49	1.132	1.293",
    (0, 2):   "0.819	0.814	1.149	1.124	0.523	1.231	1.596",
    (0, 3):   "0.774	1.595	0.657	0.75	0.891",
    (0.5, 1): "1.824	1.739	1.973	2.061	2.079",
    (0.5, 1.5): "0.98	1.755	1.435	1.095	1.008",
    (0.5, 2): "1.062	1.392	 	0.912	1.539	0.972	1.456	0.967	1.035",
    (0.5, 3): "0.98	0.592	0.77	0.833	1.055",
    (2, 1):   "1.979	 	1.631	 	1.816",
    (2, 1.5): "1.754	1.643	1.866	1.686	1.689",
    (2, 2):   "0.829	0.952	0.87	0.781	0.857	0.976",
    (2, 3):   "0.567	0.871	0.734	0.545	0.657",
    (1, 1):   "1.818	2.112	2.02	2.039	1.89	1.844	1.919	2.011	1.809",
    (1, 1.5): "1.476	1.096	1.023	1.122	1.201",
    (1, 2):   "1.01	0.909	1.128	1.06	0.982	0.871	0.944	0.845	1.173",
    (1, 3):   "0.874	0.761	1.184	0.684	0.737	0.72	0.719	0.83",
}

# LABEL = ''
# data_inputs_strings = {
#     (0, 1):   "",
#     (0, 1.5):   "",
#     (0, 2):   "",
#     (0, 3):   "",
#     (0.5, 1): "",
#     (0.5, 1.5): "",
#     (0.5, 2): "",
#     (0.5, 3): "",
#     (2, 1):   "",
#     (2, 1.5):   "",
#     (2, 2):   "",
#     (2, 3):   "",
#     (1, 1):   "",
#     (1, 1.5):   "",
#     (1, 2):   "",
#     (1, 3):   "",
# }

# LABEL = 'pws_12k'
# data_inputs_strings = {
#     (0, 1): "1.86	1.95	1.89",
#     (0, 2): "0.71	1.23	0.94	1.12	0.9	1.11	1.69",
#     (0, 3): "0.6	1.31	1.07",
#     (0.5, 1): "1.92	1.74	1.91",
#     (0.5, 2): "0.94	1.06	1.55	0.9	1.7	0.97	1.23",
#     (0.5, 3): "0.87	0.53	1.02	0.8	0.93",
#     (1, 1): "1.85	2.03	2.03	1.78",
#     (1, 2): "0.87	0.97	1.11	1.31	0.83	0.94	0.9	0.76",
#     (1, 3): "0.83	0.82	1.19	0.57	1.18	0.73	0.67	0.84",
#     (2, 1): "1.93	2.04	1.88",
#     (2, 2): "0.81	0.85	0.98	0.79	0.82	0.97",
#     (2, 3): "0.41	1.36	0.8	0.52",
# }

# --- CUSTOM SIGNIFICANCE ANNOTATIONS ---
# Define the comparisons you want to highlight manually.
# Each dictionary specifies a comparison.
# For 'vertical' comparisons (between 'act' levels at a specific 'mass'):
#   - 'type': 'vertical'
#   - 'm_val_at_comparison': The 'mass' level for the comparison.
#   - 'act_val1', 'act_val2': The 'act' levels to compare.
# For 'horizontal' comparisons (between 'mass' levels at a specific 'act'):
#   - 'type': 'horizontal'
#   - 'act_val_at_comparison': The 'act' level for the comparison.
#   - 'm_val1', 'm_val2': The 'mass' levels to compare.
sig_annotations = [
    # Example of a vertical comparison
    # {
    #     'type': 'vertical',
    #     'm_val_at_comparison': 0.5,
    #     'act_val1': 1,
    #     'act_val2': 2,
    #     'symbol': '*',
    #     'y_offset_factor': 0.05
    # },
    # {
        # 'type': 'vertical',
        # 'm_val_at_comparison': 0.5,
        # 'act_val1': 2,
        # 'act_val2': 3,
        # 'symbol': '**',
        # 'y_offset_factor': 0.05
    # },
    # Example of a horizontal comparison
    # {
    #     'type': 'horizontal',
    #     'act_val_at_comparison': 1,
    #     'm_val1': 0,
    #     'm_val2': 1,
    #     'symbol': '#',
    #     'y_offset_factor': 0.1
    # },
    # {
    #     'type': 'horizontal',
    #     'act_val_at_comparison': 2,
    #     'm_val1': 0,
    #     'm_val2': 1,
    #     'symbol': '*',
    #     'y_offset_factor': 0.0
    # },
    # {
    #     'type': 'horizontal',
    #     'act_val_at_comparison': 2,
    #     'm_val1': 0.5,
    #     'm_val2': 1,
    #     'symbol': 'ns',
    #     'y_offset_factor': -0.7
    # },
    # {
    #     'type': 'horizontal',
    #     'act_val_at_comparison': 2,
    #     'm_val1': 2,
    #     'm_val2': 1,
    #     'symbol': '**',
    #     'y_offset_factor': 0.1
    # },
]
font_size = 14


x_ticks = [0, 0.5, 1, 1.25, 1.5, 2]
y_common = 0.005
# --- 1. Data Parsing Function ---
def parse_data_strings_to_dataframe(data_inputs_strings: Dict[Tuple[float, int], str]) -> pl.DataFrame:
    """
    Parses a dictionary of string data for Z_Value into a Pandas DataFrame.

    Args:
        data_inputs_strings: A dictionary where keys are (m_value, a_value) tuples
                             and values are space-separated strings of Z_Values.

    Returns:
        A pandas DataFrame with 'm', 'a', and 'Z_Value' columns.
    """
    raw_data_list = []
    for (m_val, a_val), data_str in data_inputs_strings.items():
        z_values = [float(x) for x in data_str.split()]
        for z in z_values:
            raw_data_list.append({'m': m_val, 'a': a_val, 'Z_Value': z})
    return pl.DataFrame(raw_data_list)

# # --- 2. Statistical Calculation Function ---
# def calculate_group_statistics(df_raw: pl.DataFrame) -> pl.DataFrame:
#     """
#     Calculates median, Q1, and Q3 for 'Z_Value' grouped by 'm' and 'a'.

#     Args:
#         df_raw: The raw data DataFrame.

#     Returns:
#         A DataFrame with 'm', 'a', 'median', 'q1', and 'q3' columns.
#     """
#     grouped_stats = df_raw.group_by(['m', 'a']).agg([
#         pl.col('Z_Value').median().alias('median'),
#         pl.col('Z_Value').quantile(0.25).alias('q1'),
#         pl.col('Z_Value').quantile(0.75).alias('q3')
#     ])
#     return grouped_stats

def calculate_group_statistics(df_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates median with 95% confidence intervals using bootstrap for 'Z_Value' grouped by 'm' and 'a'.
    
    Args:
        df_raw: The raw data DataFrame.
        
    Returns:
        A DataFrame with 'm', 'a', 'median', 'ci_lower', and 'ci_upper' columns.
    """
    def bootstrap_median_ci(values):
        """Calculate median and 95% CI using bootstrap"""
        if len(values) == 0:
            return np.nan, np.nan, np.nan
        
        # Convert to numpy array
        data = np.array(values)
        
        # If only one observation, return the value itself with NaN for CI
        if len(data) == 1:
            return data[0], np.nan, np.nan
        
        # Define statistic function (median)
        def median_stat(x):
            return np.median(x)
        
        # Perform bootstrap (requires at least 2 observations)
        try:
            rng = np.random.default_rng(42)  # Set seed for reproducibility
            res = bootstrap((data,), median_stat, n_resamples=1000, 
                           confidence_level=0.9, random_state=rng)
            
            # Return median, lower CI, upper CI
            median_val = np.median(data)
            return median_val, res.confidence_interval.low, res.confidence_interval.high
        except ValueError as e:
            # Fallback for any bootstrap errors
            median_val = np.median(data)
            return median_val, np.nan, np.nan
    
    # Group by 'm' and 'a' and apply bootstrap function
    grouped_stats = []
    
    for group_key, group_df in df_raw.group_by(['m', 'a']):
        m_val = group_key[0]  # First element of the group key tuple
        a_val = group_key[1]  # Second element of the group key tuple
        z_values = group_df.select('Z_Value').to_numpy().flatten()
        
        median_val, ci_lower, ci_upper = bootstrap_median_ci(z_values)
        
        grouped_stats.append({
            'm': m_val,
            'a': a_val,
            'median': median_val,
            'q1': ci_lower,
            'q3': ci_upper
        })
    
    return pl.DataFrame(grouped_stats)

# --- 3. Plotting Function ---
def plot_interaction_with_scatter(
    df_raw: pl.DataFrame,
    grouped_stats: pl.DataFrame,
    m_levels: List[float],
    a_levels: List[int],
    m_to_xpos: Dict[float, int],
    sig_annotations: List[Dict[str, Any]]
) -> None:
    """
    Generates an interaction plot with individual data points, median lines, IQR error bars,
    and custom significance annotations using matplotlib.

    Args:
        df_raw: Raw data DataFrame.
        grouped_stats: DataFrame with calculated medians, Q1, and Q3.
        m_levels: Sorted list of unique 'm' values.
        a_levels: Sorted list of unique 'a' values.
        m_to_xpos: Mapping from 'm' values to their x-axis plot positions.
        sig_annotations: List of dictionaries defining custom significance annotations.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.tick_params(axis='both', direction='in', which='major', length=8)
    ax.tick_params(axis='both', direction='in', which='minor', length=4)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.15)

    # Define colors for 'a' levels using a colormap for distinctness
    a_color_map = {
        1: to_01color(33, 145, 140),
        1.25: to_01color(23, 170, 100),
        1.5: to_01color(23, 231, 37),
        2: to_01color(253, 231, 37),
        3: to_01color(68, 1, 84)
    }

    jitter_width = 0.15 # Max horizontal displacement for jittered scatter points

    # --- Plot individual data points (Jittered Scatter Plot) ---
    for row in df_raw.iter_rows(named=True):
        m_val = row['m']
        a_val = row['a']
        z_val = row['Z_Value']

        x_pos = m_to_xpos[m_val]
        # Apply random jitter to x-position to avoid overplotting
        jitter = np.random.uniform(-jitter_width, jitter_width) 

        ax.scatter(x_pos + 0.1, z_val,
                   color=a_color_map[a_val],
                   alpha=0.6, # Make points slightly transparent
                   s=40,      # Marker size
                   edgecolors='gray', linewidths=0.5,
                   zorder=1) # Ensure scatter points are drawn below lines

    # --- Plot median lines and IQR error bars ---
    for a_val in a_levels:
        # Filter stats for the current 'a' level and sort by 'm'
        a_data = grouped_stats.filter(pl.col('a') == a_val).sort('m')

        # Get x-coordinates corresponding to m_levels
        x_coords = [m_to_xpos[m] for m in a_data['m']]
        y_medians = a_data['median']
        
        # Calculate asymmetric error bar lengths for IQR
        y_lower_err = a_data['median'] - a_data['q1']
        y_upper_err = a_data['q3'] - a_data['median']
        
        formula_label = r"$\mathrm{MEE} \propto a^{" + str(a_val) + r"}$"
        # Plot line connecting medians
        ax.plot(x_coords, y_medians,
                color=a_color_map[a_val],
                linewidth=2,
                marker='o', markersize=8,
                label=formula_label,
                zorder=2) # Ensure lines are drawn above scatter points

        # Plot IQR error bars
        ax.errorbar(
            x_coords, y_medians,
            yerr=[y_lower_err, y_upper_err],  # Asymmetric error bars
            fmt='none',
            capsize=5,  # Increase cap size for larger errorbar ends
            elinewidth=1.5,  # Make the errorbar lines thicker
            capthick=1.5,    # Make the cap lines thicker
            color=a_color_map[a_val],
            linewidth=1.5,
            zorder=2
        )
    
    _data = grouped_stats.filter((pl.col('a') == 1.5) & (pl.col('m') == 1.0))
    if MARK_CHOSEN:
        ax.plot(
            2.0, 
            _data['median'].item(), 
            marker='o',
            markerfacecolor='none',          # 속 비우고
            markeredgecolor='gray',         # 테두리 초록색
            markeredgewidth=1.5,               # 적당한 두께
            markersize=20,                   # 크기 조정
            linestyle='None'                 # 선 없이 marker만 찍음
        )
        ax.annotate(
            r"Chosen $(\alpha, \beta)$",
            (2.0, _data['median'].item()),
            textcoords="offset points",
            xytext=(25, 0),
            ha='left',
            va='center',
            fontweight='medium',
            fontsize=FONT_SIZE_LABEL
        )
    error_bar_width = 1
    # --- Add manual significance annotations ---
    for annotation in sig_annotations:
        y_offset_factor = annotation['y_offset_factor']
        symbol = annotation['symbol']

        if annotation.get('type') == 'vertical' or 'm_val_at_comparison' in annotation: # Fallback for old format
            m_val = annotation['m_val_at_comparison']
            act1 = annotation['act_val1']
            act2 = annotation['act_val2']

            stats_act1 = grouped_stats.filter((pl.col('m') == m_val) & (pl.col('a') == act1))
            stats_act2 = grouped_stats.filter((pl.col('m') == m_val) & (pl.col('a') == act2))

            if stats_act1.empty or stats_act2.empty:
                continue

            median1 = stats_act1['median'].iloc[0]
            median2 = stats_act2['median'].iloc[0]
            q3_1 = stats_act1['q3'].iloc[0]
            q3_2 = stats_act2['q3'].iloc[0]

            y_bar_base = max(q3_1, q3_2)
            y_bar_pos = y_bar_base + y_bar_base * y_offset_factor

            x_pos_on_plot = m_to_xpos[m_val]

            horizontal_offset = jitter_width * 0.7
            x_bar_start = x_pos_on_plot - horizontal_offset
            x_bar_end = x_pos_on_plot + horizontal_offset

            ax.plot([x_bar_start, x_bar_start], [median1, y_bar_pos], color='black', linewidth=error_bar_width, linestyle='-')
            ax.plot([x_bar_end, x_bar_end], [median2, y_bar_pos], color='black', linewidth=error_bar_width, linestyle='-')
            ax.plot([x_bar_start, x_bar_end], [y_bar_pos, y_bar_pos], color='black', linewidth=error_bar_width, linestyle='-')
            ax.text(x_pos_on_plot, y_bar_pos + y_bar_pos * 0.02, symbol, ha='center', va='bottom', fontsize=font_size * 1.4, color='black', weight='bold')

        elif annotation.get('type') == 'horizontal':
            act_val = annotation['act_val_at_comparison']
            m1 = annotation['m_val1']
            m2 = annotation['m_val2']

            stats_m1 = grouped_stats.filter((pl.col('a') == act_val) & (pl.col('m') == m1))
            stats_m2 = grouped_stats.filter((pl.col('a') == act_val) & (pl.col('m') == m2))

            if stats_m1.height == 0 or stats_m2.height == 0:
                continue

            median1 = stats_m1['median'].item()
            median2 = stats_m2['median'].item()
            q3_1 = stats_m1['q3'].item()
            q3_2 = stats_m2['q3'].item()

            y_bar_base = max(q3_1, q3_2)
            y_bar_pos = y_bar_base + y_bar_base * y_offset_factor

            x_pos1 = m_to_xpos[m1]
            x_pos2 = m_to_xpos[m2]

            ax.plot([x_pos1, x_pos1], [median1, y_bar_pos], color='black', linewidth=error_bar_width, linestyle='-')
            ax.plot([x_pos2, x_pos2], [median2, y_bar_pos], color='black', linewidth=error_bar_width, linestyle='-')
            ax.plot([x_pos1, x_pos2], [y_bar_pos, y_bar_pos], color='black', linewidth=error_bar_width, linestyle='-')
            ax.text((x_pos1 + x_pos2) / 2, y_bar_pos + y_bar_pos * 0.02, symbol, ha='center', va='bottom', fontsize=font_size, color='black')

    # --- Axis Labels, Title, and Legend ---
    ax.set_xticks(list(m_to_xpos.values())) # Set tick positions to numerical indices
    ax.set_xlabel(r'$\beta$ (Muscle mass exponent)', fontsize=FONT_SIZE_LABEL)
    ax.set_xticklabels([
        r"$\mathbf{mEE} \propto m^{0.0}$", r"$\mathbf{mEE} \propto m^{0.5}$", 
        r"$\mathbf{mEE} \propto m^{1.0}$", r"$\mathbf{mEE} \propto m^{2.0}$", 
        ], fontsize=FONT_SIZE_LEGEND)
    ax.set_ylabel(r'Preferred walking speed ($v_{sim}$) (m/s)', fontsize=FONT_SIZE_LABEL) # Using specific Z_Value name
    ax.tick_params(axis='y', labelsize=FONT_SIZE_LEGEND + 4)
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0) # Add grid lines in background
    
    # ns label across x axis
    # ax.plot([min(x_ticks), max(x_ticks)], [y_common]*2, color='black')
    # ax.text(np.mean(x_ticks), y_common + 0.01, 'ns', ha='center', va='bottom', fontsize=12)
    ax.set_ylim(0.3, 2.8)
    ax.set_yticks(np.arange(0.5, 2.5, 0.5))
    
    # Add horizontal line at y=1.25 with annotation
    ax.axhline(y=1.25, color='gray', linestyle='--', linewidth=2, alpha=0.8, zorder=3)
    ax.annotate(r'$v_{real}$', xy=(max(m_to_xpos.values()) - 0.15, 1.26), xytext=(5, 0), 
                textcoords='offset points', va='bottom', ha='center', 
                fontsize=FONT_SIZE_LABEL, color='gray', weight='heavy')
    
    # Adjust legend position outside the plot area
    lg = ax.legend(loc='upper center', fontsize=14, title=r'$\alpha$ (Muscle activation exponent)', title_fontsize =16, ncol=3)
    lg.get_frame().set_alpha(0.3)

    # Adjust layout to prevent legend from overlapping with the plot
    plt.savefig(PROJECT_ROOT / "plot/supp_d4.png", dpi=150, transparent=True)


def main():
    # --- Execute workflow ---
    # 1. Parse raw data
    df_raw = parse_data_strings_to_dataframe(data_inputs_strings)

    # 2. Get sorted unique m and a levels
    m_levels = sorted(df_raw['m'].unique())
    a_levels = sorted(df_raw['a'].unique())

    # 3. Create mapping from 'm' values to their x-axis plot positions (0, 1, 2, ...)
    m_to_xpos = {m: i for i, m in enumerate(m_levels)}

    # 4. Calculate group statistics (medians, Q1, Q3)
    grouped_stats = calculate_group_statistics(df_raw)

    # 5. Plot the data
    plot_interaction_with_scatter(df_raw, grouped_stats, m_levels, a_levels, m_to_xpos, sig_annotations)
    
    
if __name__ == "__main__":
    main()