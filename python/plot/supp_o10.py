# Comparison between the model prediction and the ground truth
from matplotlib.ticker import MaxNLocator
from python.surrogate.preprocess import gt_from_exp_name, gt_stat_from_exp_name
from python.plot.util import *
import matplotlib.pyplot as plt
import numpy as np


# Velocity range for plotting
VELOCITY_RANGE_KMH = np.arange(2.0, 6.1, 0.05)

# Outlier removal configuration
MRR_MAX_THRESHOLD = 40.0  # Maximum reasonable MRR percentage (remove values above this)
MRR_MIN_THRESHOLD = -10.0  # Minimum reasonable MRR percentage (remove values below this)
BENEFIT_MAX_THRESHOLD = 100.0  # Maximum reasonable benefit in J/kg/m
BENEFIT_MIN_THRESHOLD = -20.0  # Minimum reasonable benefit in J/kg/m
OUTLIER_REMOVAL_ENABLED = True  # Toggle outlier removal on/off

# Robust colorbar configuration
COLORBAR_ROBUST_MODE = True  # Use robust colorbar limits to prevent outlier compression
COLORBAR_NTH_MAX = 5  # Use 3rd maximum value as colorbar upper limit (prevents outlier compression)
COLORBAR_NTH_MIN = 3  # Use 3rd minimum value as colorbar lower limit (prevents outlier compression)

# Floating point tolerance for K value comparisons
K_TOLERANCE = 1e-6  # Consider K values within this tolerance as "unassisted" (K=0)
fontsize = 22

CKPTs = [
'exo_hei_resist_0807_131406_21000+memo__b21_df_U58500_no_mod_state_ma15+_on_0907_204001',
'exo_hei_assist_0809_211111_21000+memo__b21_df_U58500_no_mod_state_ma15+_on_0923_011048',
'exo_no_hei2_0904_093129_24000+memo__b21_df_U58500_no_mod_state_ma15+_on_0923_002707',
]
TITLES = [r'$r_{HEI}$', r'$r_{assist}$', r'no $r_{HEI}$']
CBAR_RANGE = {
    'cot': [1, 2.5],
    'benefit': [0, 0.3],
    'mrr': [0, 25], 
}

# Scaling factors for display
PHASE_SCALE = REF_CADENCE
STRIDE_SCALE = REF_STEP


OUTPUT_COL = ['cot_ma15']

speed_margin_kmh = 0.2


def remove_outliers(benefit_values, mrr_values, verbose=False):
    """Remove outliers based on configurable thresholds."""
    if not OUTLIER_REMOVAL_ENABLED:
        return benefit_values, mrr_values

    original_count = len(benefit_values)

    # Create boolean masks for valid data points
    benefit_mask = (benefit_values >= BENEFIT_MIN_THRESHOLD) & (benefit_values <= BENEFIT_MAX_THRESHOLD)
    mrr_mask = (mrr_values >= MRR_MIN_THRESHOLD) & (mrr_values <= MRR_MAX_THRESHOLD)

    # Combine masks - keep points that satisfy both conditions
    valid_mask = benefit_mask & mrr_mask

    # Filter the data
    filtered_benefit = benefit_values[valid_mask]
    filtered_mrr = mrr_values[valid_mask]

    removed_count = original_count - len(filtered_benefit)

    if verbose and removed_count > 0:
        print(f"  Outlier removal: {removed_count}/{original_count} points removed ({removed_count/original_count*100:.1f}%)")
        print(f"  Thresholds: Benefit [{BENEFIT_MIN_THRESHOLD}, {BENEFIT_MAX_THRESHOLD}], MRR [{MRR_MIN_THRESHOLD}, {MRR_MAX_THRESHOLD}]%")

    return filtered_benefit, filtered_mrr


def get_robust_colorbar_limits(data_grid, nth_max=None, nth_min=None):
    """Calculate robust colorbar limits using nth maximum/minimum values to prevent outlier compression.

    Args:
        data_grid: 2D numpy array with data values (may contain NaN)
        nth_max: Use nth maximum value as upper limit (default from COLORBAR_NTH_MAX)
        nth_min: Use nth minimum value as lower limit (default from COLORBAR_NTH_MIN)

    Returns:
        tuple: (vmin, vmax) for colorbar limits
    """
    if not COLORBAR_ROBUST_MODE:
        # Use default behavior (full range)
        return None, None

    if nth_max is None:
        nth_max = COLORBAR_NTH_MAX
    if nth_min is None:
        nth_min = COLORBAR_NTH_MIN

    # Get valid (non-NaN) values
    valid_data = data_grid[~np.isnan(data_grid)]

    if len(valid_data) == 0:
        return None, None

    if len(valid_data) < max(nth_max, nth_min):
        # Not enough data points for robust limits, use full range
        return np.nanmin(data_grid), np.nanmax(data_grid)

    # Sort data to find nth maximum and minimum
    sorted_data = np.sort(valid_data)

    # Get nth minimum (nth_min - 1 because of 0-indexing)
    vmin = sorted_data[nth_min - 1] if nth_min <= len(sorted_data) else sorted_data[0]

    # Get nth maximum from the end (nth_max - 1 because of 0-indexing)
    vmax = sorted_data[-(nth_max)] if nth_max <= len(sorted_data) else sorted_data[-1]

    # Ensure vmin < vmax
    if vmin >= vmax:
        vmin = np.nanmin(data_grid)
        vmax = np.nanmax(data_grid)

    return vmin, vmax


def load_data_once(ckpt: str):
    """Load and preprocess data once for efficiency."""
    input_col = ['Phase', 'Stride', 'K', 'Delay']
    data = gt_from_exp_name(ckpt, input_col, OUTPUT_COL, 'metabolic', is_ckpt=False)[0].collect()
    gt_stat = gt_stat_from_exp_name(ckpt, input_col, OUTPUT_COL, 'metabolic', is_ckpt=False)
    data = denormalize(data, gt_stat)

    # Check if velocity column exists, if not compute it from Phase and Stride
    if "velocity" not in data.columns:
        print("  Computing velocity from Phase and Stride...")
        data = data.with_columns(
            (pl.col("Stride") * pl.col("Phase") * REF_VELOCITY).alias("velocity")
        )
        print(f"  Velocity computed: range {data['velocity'].min():.2f} to {data['velocity'].max():.2f} km/h")

    return data


def get_all_unique_parameter_pairs(data):
    """Get ALL possible (Phase, Stride) pairs from Cartesian product of unique values."""
    # Get unique values for each parameter separately
    unique_phases = sorted(data['Phase'].unique())
    unique_strides = sorted(data['Stride'].unique())

    if len(unique_phases) == 0 or len(unique_strides) == 0:
        return [], [], [], {}, {}

    # Create ALL possible combinations (Cartesian product)
    import itertools
    all_combinations = list(itertools.product(unique_phases, unique_strides))

    # Convert to polars DataFrame format
    import polars as pl
    unique_pairs = pl.DataFrame({
        'Phase': [combo[0] for combo in all_combinations],
        'Stride': [combo[1] for combo in all_combinations]
    }).sort(['Stride', 'Phase'])

    # Create mappings for grid positions
    phase_to_idx = {phase: i for i, phase in enumerate(unique_phases)}
    stride_to_idx = {stride: i for i, stride in enumerate(unique_strides)}

    return unique_pairs, unique_phases, unique_strides, phase_to_idx, stride_to_idx


def compute_cot_unassisted_all_parameter_pairs(data, walking_speed_kmh: float = None, verbose: bool = False):
    """Compute CoT for unassisted gait (K=0) for ALL unique (Phase, Stride) pairs across all velocities or specific velocity."""
    if walking_speed_kmh is None:
        # Use ALL velocity data
        velocity_data = data
        if verbose:
            print(f'Using ALL velocity data for CoT unassisted: {len(velocity_data)} points')
            print(f'Velocity range: {velocity_data["velocity"].min()*3.6:.1f} to {velocity_data["velocity"].max()*3.6:.1f} km/h')
    else:
        # Filter data for specific velocity range
        velocity_data = data.filter(pl.col("velocity") >= (walking_speed_kmh - speed_margin_kmh) / 3.6)
        velocity_data = velocity_data.filter(pl.col("velocity") <= (walking_speed_kmh + speed_margin_kmh) / 3.6)

        if verbose:
            print(f'Number of data within [{walking_speed_kmh - speed_margin_kmh}, {walking_speed_kmh + speed_margin_kmh}] km/h for CoT unassisted: {len(velocity_data)}')

    if len(velocity_data) == 0:
        return None, None, None, None, None, None

    # Filter for unassisted data only (K ≈ 0)
    unassisted_data = velocity_data.filter(pl.col("K").abs() <= K_TOLERANCE)

    if len(unassisted_data) == 0:
        return None, None, None, None, None, None

    # Get all unique parameter pairs
    unique_pairs, unique_phases, unique_strides, phase_to_idx, stride_to_idx = get_all_unique_parameter_pairs(unassisted_data)

    if len(unique_pairs) == 0:
        return None, None, None, None, None, None

    # Initialize result grid based on actual unique values
    n_strides = len(unique_strides)
    n_phases = len(unique_phases)
    cot_grid = np.full((n_strides, n_phases), np.nan)

    # Compute CoT for each unique parameter pair
    processed_pairs = 0
    for row in unique_pairs.iter_rows(named=True):
        phase = row['Phase']
        stride = row['Stride']

        # Get exact data for this parameter pair
        subset = unassisted_data.filter(
            (pl.col("Phase") == phase) &
            (pl.col("Stride") == stride)
        )

        if len(subset) == 0:
            continue

        # Calculate mean CoT for this parameter pair
        mean_cot = subset[OUTPUT_COL[0]].mean()

        # Map to grid positions
        stride_idx = stride_to_idx[stride]
        phase_idx = phase_to_idx[phase]

        cot_grid[stride_idx, phase_idx] = mean_cot
        processed_pairs += 1

    if verbose:
        total_pairs = len(unique_pairs)
        print(f'Processed {processed_pairs}/{total_pairs} unique (Phase, Stride) pairs for CoT unassisted')
        print(f'Grid shape: {n_strides} strides × {n_phases} phases')

    return cot_grid, unique_phases, unique_strides, unique_pairs, (phase_to_idx, stride_to_idx), processed_pairs


def compute_benefit_all_parameter_pairs(data, walking_speed_kmh: float = None, verbose: bool = False):
    """Compute benefit and MRR for ALL unique (Phase, Stride) pairs across all velocities or specific velocity."""
    if walking_speed_kmh is None:
        # Use ALL velocity data
        velocity_data = data
        if verbose:
            print(f'Using ALL velocity data: {len(velocity_data)} points')
            print(f'Velocity range: {velocity_data["velocity"].min()*3.6:.1f} to {velocity_data["velocity"].max()*3.6:.1f} km/h')
    else:
        # Filter data for specific velocity range
        velocity_data = data.filter(pl.col("velocity") >= (walking_speed_kmh - speed_margin_kmh) / 3.6)
        velocity_data = velocity_data.filter(pl.col("velocity") <= (walking_speed_kmh + speed_margin_kmh) / 3.6)

        if verbose:
            print(f'Number of data within [{walking_speed_kmh - speed_margin_kmh}, {walking_speed_kmh + speed_margin_kmh}] km/h: {len(velocity_data)}')

    if len(velocity_data) == 0:
        return None, None, None, None, None, None

    # Get all unique parameter pairs
    unique_pairs, unique_phases, unique_strides, phase_to_idx, stride_to_idx = get_all_unique_parameter_pairs(velocity_data)

    if len(unique_pairs) == 0:
        return None, None, None, None, None, None

    # Initialize result grids based on actual unique values
    n_strides = len(unique_strides)
    n_phases = len(unique_phases)
    benefit_grid = np.full((n_strides, n_phases), np.nan)
    mrr_grid = np.full((n_strides, n_phases), np.nan)

    # Compute benefit and MRR for each unique parameter pair
    processed_pairs = 0
    for row in unique_pairs.iter_rows(named=True):
        phase = row['Phase']
        stride = row['Stride']

        # Get exact data for this parameter pair
        subset = velocity_data.filter(
            (pl.col("Phase") == phase) &
            (pl.col("Stride") == stride)
        )

        if len(subset) == 0:
            continue

        data_unassisted = subset.filter(pl.col("K").abs() <= K_TOLERANCE)
        data_assisted = subset.filter(pl.col("K").abs() > K_TOLERANCE)

        if len(data_unassisted) == 0 or len(data_assisted) == 0:
            continue

        # Find minimum CoT for assisted and unassisted conditions
        min_cot_unassisted = data_unassisted[OUTPUT_COL[0]].min()
        min_cot_assisted = data_assisted[OUTPUT_COL[0]].min()

        benefit = min_cot_unassisted - min_cot_assisted
        mrr = benefit / min_cot_unassisted

        # Apply outlier removal before storing in grid
        benefit_filtered, mrr_filtered = remove_outliers(
            np.array([benefit]), np.array([mrr * 100]), verbose=False
        )

        # Only store if not filtered out as outlier
        if len(benefit_filtered) > 0:
            # Map to grid positions
            stride_idx = stride_to_idx[stride]
            phase_idx = phase_to_idx[phase]

            benefit_grid[stride_idx, phase_idx] = benefit_filtered[0]
            mrr_grid[stride_idx, phase_idx] = mrr_filtered[0]  # Already converted to percentage
        processed_pairs += 1

    if verbose:
        total_pairs = len(unique_pairs)
        print(f'Processed {processed_pairs}/{total_pairs} unique (Phase, Stride) pairs')
        print(f'Grid shape: {n_strides} strides × {n_phases} phases')

    return benefit_grid, mrr_grid, unique_phases, unique_strides, unique_pairs, (phase_to_idx, stride_to_idx)


def plot_heatmap_grid(ckpts: list, velocity_kmh: float = None):
    """Plot grid layout: 3 rows (CoT, Benefit, MRR) × (N+1) columns (checkpoints + colorbar column) with normalization."""
    n_ckpts = len(ckpts)

    # Two-stage GridSpec approach for proper spacing control
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(3.5 * (n_ckpts + 1), 4 * 3))
    
    # 1) Outer grid: [Data block | Colorbar block]
    gs_outer = gridspec.GridSpec(
        1, 2,
        width_ratios=[n_ckpts, 0.25],     # Overall ratio
        left=0.07, right=0.95, bottom=0.07, top=0.95,
        wspace=0.0                       # No spacing between data and colorbar blocks
    )
    
    # 2) Inner grids
    gs_data = gridspec.GridSpecFromSubplotSpec(
        3, n_ckpts,
        subplot_spec=gs_outer[0, 0],
        wspace=0.2, hspace=0.23           # Normal spacing only between data columns
    )
    gs_cbar = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=gs_outer[0, 1],
        wspace=0.0, hspace=0.2           # No horizontal spacing for colorbar column
    )
    
    # 3) Create axes array
    axes = np.empty((3, n_ckpts + 1), dtype=object)
    for r in range(3):
        for c in range(n_ckpts):
            axes[r, c] = fig.add_subplot(gs_data[r, c])
        axes[r, n_ckpts] = fig.add_subplot(gs_cbar[r, 0])  # Colorbar column

    results = {}

    for col_idx, ckpt in enumerate(ckpts):
        print(f"\nProcessing checkpoint {col_idx+1}/{n_ckpts}: {ckpt}")

        # Load data once for efficiency
        data = load_data_once(ckpt)

        # Compute heatmap data for all unique parameter pairs
        result = compute_benefit_all_parameter_pairs(data, velocity_kmh, verbose=True)

        if result[0] is None:
            print(f"No data available for {ckpt}")
            continue

        benefit_grid, mrr_grid, unique_phases, unique_strides, unique_pairs, (phase_to_idx, stride_to_idx) = result

        # Compute CoT unassisted heatmap data
        cot_result = compute_cot_unassisted_all_parameter_pairs(data, velocity_kmh, verbose=True)

        if cot_result[0] is None:
            print(f"No CoT unassisted data available for {ckpt}")
            continue

        cot_grid, _, _, _, _, _ = cot_result

        # NORMALIZATION: Normalize CoT and Benefit with respect to minimal CoT value
        min_cot = np.nanmin(cot_grid)
        print(f"Normalizing with minimal CoT value: {min_cot:.2f}")

        # Apply normalization
        cot_grid_normalized = cot_grid / min_cot
        benefit_grid_normalized = benefit_grid / min_cot

        # Define extent for all plots with scaling
        extent = [min(unique_phases) * PHASE_SCALE, max(unique_phases) * PHASE_SCALE, 
                 min(unique_strides) * STRIDE_SCALE, max(unique_strides) * STRIDE_SCALE]

        # Get axes for this checkpoint (column)
        ax_cot = axes[0, col_idx]     # Row 0: CoT Unassisted
        ax_benefit = axes[1, col_idx] # Row 1: Benefit
        ax_mrr = axes[2, col_idx]     # Row 2: MRR

        # Row 0: CoT Unassisted (normalized)
        im_cot = ax_cot.imshow(cot_grid_normalized, origin='lower', aspect='auto', cmap='coolwarm',
                              extent=extent, vmin=CBAR_RANGE['cot'][0], vmax=CBAR_RANGE['cot'][1])

        # Row 1: Benefit (normalized)
        im_benefit = ax_benefit.imshow(benefit_grid_normalized, origin='lower', aspect='auto', cmap='viridis',
                                      extent=extent, vmin=CBAR_RANGE['benefit'][0], vmax=CBAR_RANGE['benefit'][1])

        # Row 2: MRR (not normalized)
        im_mrr = ax_mrr.imshow(mrr_grid, origin='lower', aspect='auto', cmap='plasma',
                              extent=extent, vmin=CBAR_RANGE['mrr'][0], vmax=CBAR_RANGE['mrr'][1])

        # Store images for colorbar creation after processing all checkpoints
        if col_idx == 0:  # Store images from first checkpoint for colorbars
            im_cot_ref = im_cot
            im_benefit_ref = im_benefit
            im_mrr_ref = im_mrr
        
        ax_cot.set_title(TITLES[col_idx], fontsize=fontsize)

        if not np.all(np.isnan(benefit_grid_normalized)):
            max_benefit_idx = np.unravel_index(np.nanargmax(benefit_grid_normalized), benefit_grid_normalized.shape)
            # max_phase_benefit = unique_phases[max_benefit_idx[1]]
            # max_stride_benefit = unique_strides[max_benefit_idx[0]]
            max_benefit_val = benefit_grid_normalized[max_benefit_idx]
            mean_benefit_val = np.nanmean(benefit_grid_normalized)
            # ax_benefit.plot(max_phase_benefit * PHASE_SCALE, max_stride_benefit * STRIDE_SCALE, 'o', markersize=10, markeredgecolor='white', markeredgewidth=1)
            ax_benefit.set_title(f'(max) {max_benefit_val:.2f}, (mean) {mean_benefit_val:.2f}', fontsize=fontsize)

        if not np.all(np.isnan(mrr_grid)):
            max_mrr_idx = np.unravel_index(np.nanargmax(mrr_grid), mrr_grid.shape)
            max_phase_mrr = unique_phases[max_mrr_idx[1]]
            max_stride_mrr = unique_strides[max_mrr_idx[0]]
            max_mrr_val = mrr_grid[max_mrr_idx]
            # ax_mrr.plot(max_phase_mrr * PHASE_SCALE, max_stride_mrr * STRIDE_SCALE, 'o', markersize=10, markeredgecolor='white', markeredgewidth=1)
            ax_mrr.set_title(f'(max) {max_mrr_val:.1f}%', fontsize=fontsize)

        # Set custom tick marks to show actual parameter values (reduced ticks for grid layout)
        n_ticks = min(5, len(unique_phases))  # Fewer ticks for grid layout
        phase_tick_indices = np.linspace(0, len(unique_phases)-1, n_ticks, dtype=int)
        stride_tick_indices = np.linspace(0, len(unique_strides)-1, n_ticks, dtype=int)

        for ax in [ax_cot, ax_benefit, ax_mrr]:
            ax.set_xticks([unique_phases[i] * PHASE_SCALE for i in phase_tick_indices])
            ax.set_yticks([unique_strides[i] * STRIDE_SCALE for i in stride_tick_indices])
            # Format x and y axis tick labels to 2 decimal places
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            # MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

        print(f"Plotted {len(unique_pairs)} unique parameter pairs")
        print(f"Parameter ranges: Phase [{min(unique_phases):.3f}, {max(unique_phases):.3f}], "
              f"Stride [{min(unique_strides):.3f}, {max(unique_strides):.3f}]")

        # Store results for this checkpoint
        mean_benefit_val = np.nanmean(benefit_grid_normalized) if not np.all(np.isnan(benefit_grid_normalized)) else np.nan
        results[ckpt] = (max_mrr_val, (max_phase_mrr, max_stride_mrr), mean_benefit_val)

    # Add colorbars in the dedicated colorbar column (last column)
    cbar_col = n_ckpts  # Last column index for colorbars
    
    # Remove tick marks and labels from colorbar column axes
    for row_idx in range(3):
        axes[row_idx, cbar_col].set_xticks([])
        axes[row_idx, cbar_col].set_yticks([])
        axes[row_idx, cbar_col].axis('off')
    
    # number of ticks for colorbar
    num_ticks = 3
    
    # Add row labels as colorbar titles
    row_labels = [r'CoT$(\kappa=0)$', 'Benefit', 'MRR (%)']
    
    cbar_cot = fig.colorbar(im_cot_ref, ax=axes[0, cbar_col], shrink=1.0)
    cbar_cot.locator = plt.MaxNLocator(nbins=num_ticks)
    cbar_cot.update_ticks()
    cbar_cot.ax.set_ylabel(row_labels[0], fontsize=fontsize)
    cbar_cot.ax.set_xlim(0, 1)
    cbar_cot.ax.yaxis.set_label_coords(-3, 0.5)
    
    cbar_benefit = fig.colorbar(im_benefit_ref, ax=axes[1, cbar_col], shrink=1.0)
    cbar_benefit.locator = plt.MaxNLocator(nbins=num_ticks)
    cbar_benefit.update_ticks()
    cbar_benefit.ax.set_ylabel(row_labels[1], fontsize=fontsize)
    cbar_benefit.ax.set_xlim(0, 1)
    cbar_benefit.ax.yaxis.set_label_coords(-3, 0.5)
    
    cbar_mrr = fig.colorbar(im_mrr_ref, ax=axes[2, cbar_col], shrink=1.0)
    cbar_mrr.locator = plt.MaxNLocator(nbins=num_ticks)
    cbar_mrr.update_ticks()
    cbar_mrr.ax.set_ylabel(row_labels[2], fontsize=fontsize)
    cbar_mrr.ax.set_xlim(0, 1)
    cbar_mrr.ax.yaxis.set_label_coords(-3, 0.5)

    # Add common axis labels using supxlabel and supylabel
    fig.supxlabel('Step frequency (Hz)', fontsize=fontsize)
    fig.supylabel('Step length (m)', fontsize=fontsize)
    
    # Show the plot
    return results


def main():
    print("Starting max_benefit analysis...")

    try:
        # Plot grid heatmap for all checkpoints at once
        print(f"\n=== PLOTTING GRID HEATMAP FOR ALL CHECKPOINTS ===")
        results = plot_heatmap_grid(CKPTs)

        print("Analysis completed successfully!")
        print("="*100)
        print("ckpt, max_mrr, max_mrr_position, mean_benefit:")
        for ckpt, (max_mrr, max_pos, mean_benefit) in results.items():
            print(f"{ckpt}, {max_mrr:.1f}, {max_pos}, {mean_benefit:.3f}")
        plt.savefig(PLOT_SAVE_DIR / "supp_o10.png", dpi=100, bbox_inches='tight', transparent=True)
        print(f"Figure saved as: supp_o10.png")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()