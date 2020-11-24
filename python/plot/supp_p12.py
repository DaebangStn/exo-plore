# Plot pathology trend from PATHO_MEAN_K configuration
# X-axis: Index of list (0, 1, 2, 3, 4)
# Y-axis: Values from PATHO_MEAN_K
# Each pathology corresponds to a separate line
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from python.plot.util import *

PATHO_EXPERIMENTS = [
    # Calcaneus (with r_HEI)
    {'name': 'p21_df_dn1_1_calca_0816_120417_16000', 'pathology': 'Calcaneus', 'label': 'seed1',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [18.866, 17.341, 13.858, 12.460, 11.780]},
    {'name': 'p21_df_dn1_1_calc_0811_134143_16000', 'pathology': 'Calcaneus', 'label': 'seed2',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [11.771, 10.012, 6.744, 8.135, 7.897]},
    {'name': 'p21_df_dn1_1_calc_0811_142659_16000', 'pathology': 'Calcaneus', 'label': 'seed3',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [10.227, 7.483, 2.761, 2.164, 1.482]},
    {'name': 'p21_df_dn1_1_calca_0821_234541_16000', 'pathology': 'Calcaneus', 'label': 'seed4',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 'y_data': [18.719, 13.021, 9.857, 7.221, 6.983, 6.492]},
    # Calcaneus (no r_HEI)
    {'name': 'p21_df_calc_0821_103638_15000', 'pathology': r'Calcaneus (no $r_{HEI}$)', 'label': 'seed1',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [13.036, 13.363, 19.203, 20.855, 20.246]},
    {'name': 'p21_df_calc_0820_013515_16000', 'pathology': r'Calcaneus (no $r_{HEI}$)', 'label': 'seed2',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [13.638, 13.050, 17.300, 17.225, 19.173]},
    {'name': 'p21_df_calc_0915_213451_16000', 'pathology': r'Calcaneus (no $r_{HEI}$)', 'label': 'seed3',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [5.872, 6.232, 10.334, 10.720, 12.759]},
    {'name': 'p21_df_calc_0922_143118_16000', 'pathology': r'Calcaneus (no $r_{HEI}$)', 'label': 'seed4',
     'x_data': [0.0, 0.1, 0.2, 0.3, 0.4], 'y_data': [12.254, 14.966, 14.135, 14.540, 18.209]},
]
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from collections import defaultdict

# FILENAME = "pathology_trend3.png"
FILENAME = "supp_p12.png"
YRANGE = (-1, 22)
NORMALIZE_X = True  # Flag to normalize x data to 0-1 range within each pathology
# Plot configuration
FIGURE_SIZE = (4, 5)
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
    """Main function to create and display the pathology trend plot in single row"""
    # Group experiments by pathology
    grouped_data = group_experiments_by_pathology()

    # Calculate subplot layout
    n_pathologies = len(grouped_data)
    if n_pathologies == 0:
        print("No pathology data to plot")
        return

    # Create single row layout
    n_cols = n_pathologies
    n_rows = 1

    # Create figure for single row layout
    fig = plt.figure(figsize=(FIGURE_SIZE[0] * n_cols, FIGURE_SIZE[1]))
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.3, hspace=0.2)
    fig.subplots_adjust(
        left=0.1,
        right=0.97,
        top=0.88,
        bottom=0.15,
    )

    # Create subplot axes in single row
    axes = []
    for i in range(n_pathologies):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
    
    # Setup colormap
    cmap = plt.get_cmap("viridis")

    # Track r² values for LaTeX table output and mean calculation
    latex_table_data = []
    all_r_squared_values = []
    pathology_r_squared = {}  # Track r² values by pathology for mean calculation
    
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

            # Store data for LaTeX table
            latex_table_data.append({
                'pathology': pathology,
                'seed': label,
                'r_squared': r_squared
            })

            # Track r² values by pathology for mean calculation
            if pathology not in pathology_r_squared:
                pathology_r_squared[pathology] = []
            pathology_r_squared[pathology].append(r_squared)

            # Plot individual data points
            ax.scatter(x_data, y_data,
                      color=color,
                      s=60,
                      alpha=0.6)
            
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
            ax.set_ylabel(r'Averaged optimal gain, $\kappa$ (Nm)', fontsize=FONT_SIZE_LABEL)

        # Calculate mean r² for this pathology
        if pathology in pathology_r_squared and pathology_r_squared[pathology]:
            mean_r_squared = np.mean(pathology_r_squared[pathology])
            title_text = f'{pathology} (r² = {mean_r_squared:.2f})'
        else:
            title_text = f'{pathology}'

        ax.set_title(title_text, fontsize=FONT_SIZE_LABEL)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_LABEL)

        # Set 5 y-axis ticks
        ax.locator_params(axis='y', nbins=5)

        ax.grid(True, alpha=0.3)
        ax.set_ylim(YRANGE[0], YRANGE[1])
    
    # Add explanatory text on the first axis instead of individual legend entries
    # if axes:
    #     first_ax = axes[0]
    #     # Add descriptive text about colors
    #     first_ax.text(0.93, 0.93, 'Each color corresponds to\na different random seed.',
    #                  transform=first_ax.transAxes,
    #                  fontsize=16,
    #                  verticalalignment='top',
    #                  horizontalalignment='right',
    #                  bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'),
    #                  weight='normal')

    # Generate professional LaTeX table output
    print("\n" + "="*80)
    print("Professional LaTeX Table Format:")
    print("="*80)
    stringbuilder = ""
    # stringbuilder += "\\begin{table}[H]\n"
    # stringbuilder += "  \\centering\n"
    # stringbuilder += "  \\caption{R-squared values for pathology trend analysis}\n"
    stringbuilder += "  \\begin{tabular}{l l c}\n"
    stringbuilder += "    \\toprule\n"
    stringbuilder += "    \\textbf{Category} & \\textbf{Seed} & \\textbf{R\\textsuperscript{2}} \\\\"
    stringbuilder += "    \\midrule\n"

    # Group data by pathology for multirow formatting
    pathology_groups = {}
    for data in latex_table_data:
        pathology = data['pathology']
        if pathology not in pathology_groups:
            pathology_groups[pathology] = []
        pathology_groups[pathology].append(data)

    # Generate table rows with multirow formatting
    for pathology_idx, (pathology, pathology_data) in enumerate(pathology_groups.items()):
        num_seeds = len(pathology_data)

        for seed_idx, data in enumerate(pathology_data):
            if seed_idx == 0:  # First row for this pathology
                if num_seeds > 1:
                    stringbuilder += f"    \\multirow{{{num_seeds}}}{{*}}{{\\textbf{{{pathology}}}}}\n"
                else:
                    stringbuilder += f"    \\textbf{{{pathology}}}\n"
            else:  # Subsequent rows for this pathology
                stringbuilder += "    \n"

            stringbuilder += f"        & {data['seed']} & {data['r_squared']:.3f} \\\\"

        # Add separator between pathologies (except for last one)
        if pathology_idx < len(pathology_groups) - 1:
            stringbuilder += "    \\cmidrule{1-3}\n"

    stringbuilder += "    \\bottomrule\n"
    stringbuilder += "  \\end{tabular}\n"
    # stringbuilder += "  \\label{tab:pathology_r_squared}\n"
    # stringbuilder += "\\end{table}\n"
    
    plt.savefig(PLOT_SAVE_DIR / FILENAME, dpi=150, transparent=True)
    print(f"Figure saved as: {FILENAME}")


if __name__ == "__main__":
    main()
