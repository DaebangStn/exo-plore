# Plot pathology trend from PATHO_MEAN_K configuration
# X-axis: Index of list (0, 1, 2, 3, 4)
# Y-axis: Values from PATHO_MEAN_K
# Each pathology corresponds to a separate line
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from python.plot.util import *

PATHO_EXPERIMENTS = [
    # Equinus
    {'name': 'p21_df_dn1_1_equi_0811_134131_15000', 'pathology': 'Equinus', 'label': 'seed1',
     'x_data': [0.73, 0.75, 0.78, 0.82, 0.85], 'y_data': [4.010, 6.864, 10.326, 14.406, 15.545]},
    {'name': 'p21_df_dn1_1_equi_0813_185633_16000', 'pathology': 'Equinus', 'label': 'seed2',
     'x_data': [0.73, 0.75, 0.78, 0.82, 0.85], 'y_data': [0.850, 0.698, 4.670, 5.067, 7.891]},
    {'name': 'p21_df_dn1_1_equia_0817_235245_16000', 'pathology': 'Equinus', 'label': 'seed3',
     'x_data': [0.73, 0.75, 0.78, 0.82, 0.85, 0.90, 0.95, 1.00],
     'y_data': [0.454, 0.407, 1.546, 3.853, 2.502, 3.137, 3.360, 5.231]},
    {'name': 'p21_df_dn1_1_equi_0919_135840_16000', 'pathology': 'Equinus', 'label': 'seed4',
     'x_data': [0.73, 0.75, 0.78, 0.82, 0.85, 0.90], 'y_data': [6.480, 7.065, 7.626, 8.879, 8.513, 10.434]},
    # Waddling
    {'name': 'p21_df_dn1_1_wad_0811_140715_15000', 'pathology': 'Waddling', 'label': 'seed1',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [5.910, 3.163, 5.856, 8.132, 8.666]},
    {'name': 'p21_df_dn1_1_wad_0908_084141_16000', 'pathology': 'Waddling', 'label': 'seed2',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [1.326, 2.010, 3.991, 7.656, 7.942]},
    {'name': 'p21_df_dn1_1_wad_0914_055104_16000', 'pathology': 'Waddling', 'label': 'seed3',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [0.354, 2.883, 7.355, 11.869, 14.817]},
    {'name': 'p21_df_dn1_1_wad_0910_212710_16000', 'pathology': 'Waddling', 'label': 'seed4',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [6.384, 5.465, 6.956, 10.373, 11.721]},
    # Crouch
    {'name': 'p21_df_dn1_1_hypa_0826_100856_16000', 'pathology': 'Crouch', 'label': 'seed1',
     'x_data': [0.58, 0.6, 0.625, 0.65, 0.675], 'y_data': [18.278, 15.348, 12.021, 5.406, 5.141]},
    {'name': 'p21_df_dn1_1_hypa_0823_072634_16000', 'pathology': 'Crouch', 'label': 'seed2',
     'x_data': [0.57, 0.58, 0.6, 0.625, 0.65, 0.675, 0.7, 0.75],
     'y_data': [15.358, 13.886, 18.889, 15.117, 11.683, 9.109, 10.281, 9.737]},
    {'name': 'p21_df_dn1_1_hypa_0824_200136_16000', 'pathology': 'Crouch', 'label': 'seed3',
     'x_data': [0.57, 0.58, 0.6, 0.625, 0.65, 0.675, 0.7, 0.75],
     'y_data': [18.825, 18.913, 20.165, 16.656, 17.670, 15.140, 11.920, 10.678]},
    {'name': 'p21_df_dn1_1_hypa_0827_150621_16000', 'pathology': 'Crouch', 'label': 'seed4',
     'x_data': [0.57, 0.58, 0.6, 0.65, 0.7], 'y_data': [20.942, 20.806, 20.636, 15.620, 13.555]},
    # Calcaneus
    {'name': 'p21_df_dn1_1_calca_0816_120417_16000', 'pathology': 'Calcaneus', 'label': 'seed1',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [18.866, 17.341, 13.858, 12.460, 11.780]},
    {'name': 'p21_df_dn1_1_calc_0811_134143_16000', 'pathology': 'Calcaneus', 'label': 'seed2',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [11.771, 10.012, 6.744, 8.135, 7.897]},
    {'name': 'p21_df_dn1_1_calc_0811_142659_16000', 'pathology': 'Calcaneus', 'label': 'seed3',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [10.227, 7.483, 2.761, 2.164, 1.482]},
    {'name': 'p21_df_dn1_1_calca_0821_234541_16000', 'pathology': 'Calcaneus', 'label': 'seed4',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 'y_data': [18.719, 13.021, 9.857, 7.221, 6.983, 6.492]},
    # Foot drop
    {'name': 'p21_df_dn1_1_footc_0829_144123_20000', 'pathology': 'Foot drop', 'label': 'seed1',
     'x_data': [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3],
     'y_data': [13.662, 15.269, 7.686, 17.185, 2.868, 3.508, 3.636, 3.739, 3.703]},
    {'name': 'p21_df_dn1_1_footc_0830_091919_24000', 'pathology': 'Foot drop', 'label': 'seed2',
     'x_data': [0.0, 0.05, 0.1, 0.15, 0.2], 'y_data': [11.301, 10.538, 16.383, 15.494, 16.571]},
    {'name': 'p21_df_dn1_1_footc_0825_165208_16000', 'pathology': 'Foot drop', 'label': 'seed3',
     'x_data': [0.0, 0.05, 0.1, 0.15, 0.2], 'y_data': [11.402, 16.416, 18.132, 18.552, 19.453]},
    {'name': 'p21_df_dn1_1_footc_0828_035710_16000', 'pathology': 'Foot drop', 'label': 'seed4',
     'x_data': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3], 'y_data': [14.772, 18.727, 19.104, 18.979, 14.164, 11.676]},
]
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from collections import defaultdict

YRANGE = (-1, 22)
NORMALIZE_X = True  # Flag to normalize x data to 0-1 range within each pathology
# Plot configuration
FIGURE_SIZE = (5, 5)
########################################

def group_experiments_by_pathology():
    """Group experiments by pathology type, keeping individual experiments separate for plotting"""
    grouped_data = defaultdict(list)
    
    for exp in PATHO_EXPERIMENTS:
        name = exp.get('name', 'UNNAMED')
        pathology = exp.get('pathology', 'UNKNOWN')
        label = exp.get('label', name)  # Use explicit label or fallback to name
        x_data = exp.get('x_data', [])
        y_data = exp.get('y_data', [])
        
        # Validate data lengths
        if len(x_data) != len(y_data):
            print(f"[ERROR] Data length mismatch in '{name}': x_data={len(x_data)}, y_data={len(y_data)}")
            continue
        
        if len(x_data) == 0 or len(y_data) == 0:
            print(f"[WARNING] Empty data in '{name}': x_data={len(x_data)}, y_data={len(y_data)}")
            continue
        
        # Check for missing required fields
        if pathology == 'UNKNOWN':
            print(f"[ERROR] Missing 'pathology' field in '{name}'")
            continue
        
        # Apply normalization if enabled - normalize each experiment independently
        if NORMALIZE_X and x_data:
            min_x = min(x_data)
            max_x = max(x_data)
            if max_x > min_x:  # Avoid division by zero
                normalized_x = [(x - min_x) / (max_x - min_x) for x in x_data]
            else:
                normalized_x = [0.0] * len(x_data)  # All values are the same
        else:
            normalized_x = x_data
        
        # Transform x_data: negate values to reverse the order
        # This makes mild (higher original values) appear on left, severe (lower values) on right
        if normalized_x:
            transformed_x = [-x for x in normalized_x]
        else:
            transformed_x = normalized_x
        
        grouped_data[pathology].append({
            'name': name,
            'label': label,  # Use the explicit label from data structure
            'x_data': transformed_x,
            'y_data': y_data,
            'original_x': x_data,  # Keep original for reference
            'normalized_x': normalized_x if NORMALIZE_X else x_data  # Keep normalized for reference
        })
    
    return dict(grouped_data)

def compute_correlation_and_regression(x_data, y_data, pathology_name="UNKNOWN"):
    """Compute correlation coefficient and linear regression"""
    try:
        x_array = np.array(x_data)
        y_array = np.array(y_data)
        
        # Check for NaN or infinite values
        if np.any(np.isnan(x_array)) or np.any(np.isnan(y_array)):
            print(f"[ERROR] NaN values found in data for pathology '{pathology_name}'")
            return 0.0, 1.0, 0.0, 0.0, np.array([]), np.array([])
        
        if np.any(np.isinf(x_array)) or np.any(np.isinf(y_array)):
            print(f"[ERROR] Infinite values found in data for pathology '{pathology_name}'")
            return 0.0, 1.0, 0.0, 0.0, np.array([]), np.array([])
        
        # Check for sufficient data variance
        if np.var(x_array) == 0 or np.var(y_array) == 0:
            print(f"[WARNING] Zero variance in data for pathology '{pathology_name}'")
            return 0.0, 1.0, 0.0, 0.0, np.array([]), np.array([])
        
        # Compute Pearson correlation coefficient
        corr_coef, p_value = pearsonr(x_array, y_array)
        
        # Fit linear regression
        reg = LinearRegression()
        X = x_array.reshape(-1, 1)
        reg.fit(X, y_array)
        
        # Generate regression line points
        x_line = np.linspace(x_array.min(), x_array.max(), 100)
        y_line = reg.predict(x_line.reshape(-1, 1))
        
        return corr_coef, p_value, reg.coef_[0], reg.intercept_, x_line, y_line
        
    except Exception as e:
        print(f"[ERROR] Failed to compute correlation for pathology '{pathology_name}': {e}")
        return 0.0, 1.0, 0.0, 0.0, np.array([]), np.array([])

def main():
    """Main function to create and display the pathology trend plot with multiple panes"""
    # Group experiments by pathology
    grouped_data = group_experiments_by_pathology()
    
    # Calculate subplot layout
    n_pathologies = len(grouped_data)
    if n_pathologies == 0:
        print("No pathology data to plot")
        return
    
    # Create subplot layout with GridSpec - extra column for legends
    n_cols = 3
    n_rows = (n_pathologies + n_cols - 1) // n_cols
    
    # Create figure and GridSpec with extra column for legends
    fig = plt.figure(figsize=(FIGURE_SIZE[0] * n_cols + 3, FIGURE_SIZE[1] * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1]*n_cols + [0.15], wspace=0.15, hspace=0.2)
    fig.subplots_adjust(
        left=0.04,
        right=0.9, 
        top=0.96,
        bottom=0.04,
    )
    
    # Create subplot axes in the main grid
    axes = []
    for i in range(n_pathologies):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
    
    # Create legend axis in the rightmost column
    legend_ax = fig.add_subplot(gs[:, -1])
    
    # Setup colormap
    cmap = plt.get_cmap("viridis")
    
    # Track r² values for mean calculation
    all_r_squared_values = []
    
    # Plot data for each pathology
    for idx, (pathology, experiments) in enumerate(grouped_data.items()):
        ax = axes[idx]  # axes is now a simple list
        
        # Plot each experiment in this pathology group with individual regression
        for exp_idx, exp in enumerate(experiments):
            x_data = exp['x_data']
            y_data = exp['y_data']
            label = exp['label']
            
            if len(x_data) == 0 or len(y_data) == 0:
                continue
            
            # Use different color for each experiment
            n_experiments = len(experiments)
            exp_norm = mcolors.Normalize(vmin=0, vmax=max(1, n_experiments-1))
            color = cmap(exp_norm(exp_idx))
            
            # Compute individual regression for this seed
            corr_coef, p_value, slope, intercept, x_line, y_line = compute_correlation_and_regression(x_data, y_data, f"{pathology}_{label}")
            r_squared = corr_coef ** 2
            all_r_squared_values.append(r_squared)
            
            # Plot individual data points with r² in label
            ax.scatter(x_data, y_data, 
                      color=color, 
                      s=60, 
                      alpha=0.6,
                      label=f'{label} (r² = {r_squared:.3f})')
            
            # Plot individual regression line for this seed
            if len(x_line) > 0 and len(y_line) > 0:
                ax.plot(x_line, y_line, 
                       color=color, 
                       linewidth=1.5, 
                       linestyle='--',
                       alpha=0.8)
            
            # Print individual correlation statistics
            print(f"{pathology} - {label}:")
            print(f"  Correlation: r² = {r_squared:.3f}, p = {p_value:.3f}")
            print(f"  Regression: y = {slope:.2f}x + {intercept:.2f}")
            print(f"  N data points: {len(x_data)}")
            print()
        
        # Configure subplot with custom x-axis labels
        # Collect all x_data to find min and max for tick placement
        all_x_values = []
        for exp in experiments:
            all_x_values.extend(exp['x_data'])
        
        if all_x_values:
            min_x = min(all_x_values)
            max_x = max(all_x_values)
            interval = max_x - min_x if max_x > min_x else 1.0
            
            # Set only two ticks at the extremes
            # Since x_data is negated, min_x (most negative) corresponds to original max (mild)
            # and max_x (least negative) corresponds to original min (severe)
            ax.set_xticks([min_x, max_x])
            ax.set_xticklabels(['Mild', 'Severe'])  # min_x is mild, max_x is severe
            
            # Set xlim with more spacing
            xlim_min = min_x - 0.1 * interval
            xlim_max = max_x + 0.1 * interval
            ax.set_xlim(xlim_min, xlim_max)  # Normal order since data is already transformed
        
        if idx == 0:
            # ax.set_xlabel('Pathology severity', fontsize=FONT_SIZE_LABEL)        
            ax.set_ylabel('Averaged optimal gain (Nm)', fontsize=FONT_SIZE_LABEL)
        ax.set_title(f'{pathology}', fontsize=FONT_SIZE_LABEL)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_LABEL)
        
        # Set 5 y-axis ticks
        ax.locator_params(axis='y', nbins=5)
        
        # Store legend information for consolidation (don't create individual legends yet)
        pathology_r_squared = [r for r in all_r_squared_values[-len(experiments):]]
        if pathology_r_squared:
            mean_r_squared_pathology = np.mean(pathology_r_squared)
        else:
            mean_r_squared_pathology = 0.0
            
        ax.grid(True, alpha=0.3)
        ax.set_ylim(YRANGE[0], YRANGE[1])
    
    # Create consolidated legends in the legend column with proper spacing
    # First pass: calculate legend sizes
    legend_info = []
    r_squared_accumulator = 0  # Track position in all_r_squared_values
    
    for idx, (pathology, experiments) in enumerate(grouped_data.items()):
        ax = axes[idx]
        handles, labels = ax.get_legend_handles_labels()
        
        if handles:
            # Calculate r² mean for this pathology using accumulator
            pathology_start_idx = r_squared_accumulator
            pathology_end_idx = r_squared_accumulator + len(experiments)
            pathology_r_squared = all_r_squared_values[pathology_start_idx:pathology_end_idx] if pathology_start_idx < len(all_r_squared_values) else []
            
            if pathology_r_squared:
                mean_r_squared_pathology = np.mean(pathology_r_squared)
                if idx == len(grouped_data) - 1:
                    title = f'{pathology}\n(r²: {mean_r_squared_pathology:.3f})'
                else:
                    title = f'{pathology} (r²: {mean_r_squared_pathology:.3f})'
            else:
                title = f'{pathology}'
            
            # Estimate legend height based on number of entries
            # Each legend entry takes ~0.04 units, title takes ~0.06 units, plus padding
            legend_height = 0.037 + len(handles) * 0.026 + 0.01
            legend_info.append({
                'idx': idx,
                'pathology': pathology,
                'handles': handles,
                'labels': labels,
                'title': title,
                'height': legend_height
            })
            
            # Update accumulator for next pathology
            r_squared_accumulator += len(experiments)
    
    # Second pass: place legends with calculated spacing
    current_y = 1.0  # Start near top
    for info in legend_info:
        # Create legend in the legend axis
        legend = legend_ax.legend(info['handles'], info['labels'], 
                                loc='upper left',
                                bbox_to_anchor=(0.02, current_y),
                                fontsize=FONT_SIZE_LEGEND,
                                title=info['title'],
                                title_fontsize=FONT_SIZE_LEGEND,
                                frameon=True,
                                fancybox=True,
                                shadow=True)
        
        # Add the legend to the legend axis manually
        legend_ax.add_artist(legend)
        
        # Move down by the height of this legend for next one
        current_y -= info['height']
    
    # Set legend axis limits and remove ticks
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.spines['top'].set_visible(False)
    legend_ax.spines['right'].set_visible(False)
    legend_ax.spines['bottom'].set_visible(False)
    legend_ax.spines['left'].set_visible(False)
    
    # No tight_layout needed since we're using GridSpec
    
    plt.savefig(PLOT_SAVE_DIR / "figure9.png", dpi=600, transparent=True)
    print(f"Figure saved as: figure9.png")


if __name__ == "__main__":
    main()
