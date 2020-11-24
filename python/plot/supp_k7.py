import numpy as np
import polars as pl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Tuple, Dict, List
from scipy import interpolate
from tqdm import tqdm
import re
from python.plot.util import *
from python.surrogate.preprocess import data_and_param_path

plt.rc('text', usetex=True)

################ CONFIG ################

EXP_NAMES = [
'no_exo_meta_ma15_10000_0801_111350+memo_b20_ma15_10000_0801_111350_pf_param_no_mod_state_ma15_gems+_on_0826_224549',
'no_exo_meta_ma15_10000_0801_111421+memo_b20_ma15_10000_0801_111421_pf_param_no_mod_state_ma15_gems+_on_0826_224557',
'no_exo_meta_ma15_10000_0801_205540+memo_b20_ma15_10000_0801_205540_pf_param_no_mod_state_ma15_gems+_on_0826_224606',
'no_exo_meta_ma15_10000_0802_064456+memo_b20_ma15_10000_0802_064456_pf_param_no_mod_state_ma15_gems+_on_0826_224615',
'no_exo_meta_ma15_10000_0802_163808+memo_b20_ma15_10000_0802_163808_pf_param_no_mod_state_ma15_gems+_on_0826_224624',
'no_exo_meta_ma15_10000_0805_215352+memo_b20_ma15_10000_0805_215352_pf_param_no_mod_state_ma15_gems+_on_0826_224633',
'no_exo_meta_ma15_10000_0805_215354+memo_b20_ma15_10000_0805_215354_pf_param_no_mod_state_ma15_gems+_on_0826_224641',
'no_exo_meta_ma15_10000_0805_215359+memo_b20_ma15_10000_0805_215359_pf_param_no_mod_state_ma15_gems+_on_0826_224649',
'no_exo_meta_ma15_10000_0806_131413+memo_b20_ma15_10000_0806_131413_pf_param_no_mod_state_ma15_gems+_on_0826_224658',
'no_exo_meta_ma15_10000_0806_131416+memo_b20_ma15_10000_0806_131416_pf_param_no_mod_state_ma15_gems+_on_0826_224707',
'no_exo_meta_ma15_10000_0806_131419+memo_b20_ma15_10000_0806_131419_pf_param_no_mod_state_ma15_gems+_on_0826_224716',
]


# Configuration for 3 main pathology groups (bottom section)
PATHOLOGY_GROUPS = {
    'left': {
        'title': r'Ankle dorsiflexion (°)\\{{\Large [Calcaneus / Foot drop / Equinus]}',
        # 'ylabel': 'Dorsiflexion (°)',
        'ylim': (-50, 30),
        'yticks': [-40, -20, 0, 20, 40],
        'csv_paths': [
            'data/figs/gait_data/calcaneous.csv',
            'data/figs/gait_data/footdrop.csv', 
            'data/figs/gait_data/equinus.csv'
        ],
        'reference_csv': None,  # Use simulation reference
        'render_images': [
            # 4x3 grid = 12 images for left column
            'healty_foot_strike.png', 'healty_foot_swing.png', 'healty_foot_off.png',  
            'calc_strike.png', 'calc_swing.png', 'calc_off.png',   
            'foot_strike.png', 'foot_swing.png', 'foot_off.png',  
            'equi_strike.png', 'equi_swing.png', 'equi_off.png',  
        ]
    },
    'center': {
        'title': r'Knee flexion (°) {{\Large [Crouch]}',
        # 'ylabel': 'Flexion (°)',
        'ylim': (0, 80),
        'yticks': [20, 40, 60],
        'csv_paths': ['data/figs/gait_data/crouch.csv'],
        'reference_csv': None,  # Use simulation reference
        'render_images': [
            # 2x2 grid = 4 images for center column
            'healty_sagittal_strike.png', 'healty_sagittal_off.png',
            'crouch_strike.png', 'crouch_off.png',
        ]
    },
    'right': {
        'title': r'Torso sway (cm) {{\Large [Waddling]}',
        # 'ylabel': 'Sway (cm)',
        'ylim': (-2.5, 4),
        'yticks': [-2, 0, 2],
        'multiplier': 100,
        'csv_paths': ['data/figs/gait_data/waddling.csv'],
        'reference_csv': None,  # Use simulation reference
        'render_images': [
            # 2x2 grid = 4 images for right column
            'healtly_frontal_left.png', 'healtly_frontal_right.png',
            'wad_left.png', 'wad_right.png',
        ]
    }
}

# Image root directory
IMAGE_ROOT = 'data/figs/capturev3'

# Click coordinates configuration for blue circles
CLICK_COORDINATES = """
Clicked ax[6]: x=118.3, y=389.1
Clicked ax[7]: x=625.4, y=479.6
Clicked ax[11]: x=377.6, y=456.4
Clicked ax[16]: x=419.6, y=460.1
Clicked ax[22]: x=372.5, y=154.3
"""

CIRCLE_SIZES = [400, 400, 400, -1, -1]
COMMENT_Y_OFFSETS = [150, 100, 100, -180, -150]
COMMENT_TEXTS = [
    "Reduced push-off angle",
    "Toe strike precedes heel",
    "Toe-walking at midstance",
    "Excessive knee flexion",
    "Lateral trunk sway",
]


# Figure margins configuration  
FIGURE_MARGINS = {
    'left': 0.03,
    'right': 0.98, 
    'top': 0.915,
    'bottom': 0.03,
    'wspace': 0.2,
    'hspace': 0.3
}
########################################

# ============ PATHOLOGY KINEMATICS FUNCTIONS (from joint_kinematics4.py) ============

def load_csv_data(csv_path: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Load CSV data and normalize to gait cycle percentage (0-100%)
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        tuple: (joint_name, x_data, y_data) where x_data is 0-100% gait cycle
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read first line to get joint name
    with open(csv_file, 'r') as f:
        first_line = f.readline().strip()
        if not first_line.startswith('#'):
            raise ValueError(f"First line should start with '#' and contain joint name")
        joint_name = first_line[1:].strip()  # Remove '#' and strip whitespace
    
    # Read CSV data
    df = pl.read_csv(csv_file, skip_rows=1)
    
    # Get raw time data and reverse it (multiply by -1 to flip sign)
    x_raw = -df["x"].to_numpy()  # Reverse the x data by negating
    y_data = df["y"].to_numpy()
    
    # Also reverse the order of both arrays to maintain proper time progression
    x_raw = x_raw[::-1]
    # y_data = y_data[::-1]
    
    # Normalize x data to gait cycle percentage (0-100%)
    x_min, x_max = x_raw.min(), x_raw.max()
    x_data = ((x_raw - x_min) / (x_max - x_min)) * 100.0
    
    return joint_name, x_data, y_data


def get_simulation_reference_data(sim_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get normal simulation reference data from EXP_NAMES
    
    Args:
        sim_col: Column name to extract from simulation data
    
    Returns:
        Tuple of (time_axis, mean_trajectory, std_trajectory)
    """
    # Create standard gait cycle time axis (0 to 100%)
    target_gait_cycle = np.linspace(0, 100, 101)  # 0, 1, 2, ..., 100%
    all_normalized_trajectories = []
    
    # Get total parameter count for progress bar (first experiment param count * num experiments)
    total_params = 0
    if EXP_NAMES:
        try:
            _, first_param = data_and_param_path(EXP_NAMES[0], read_file=True)
            first_param_count = len(first_param.collect()["param_idx"].unique())
            total_params = first_param_count * len(EXP_NAMES)
        except:
            total_params = len(EXP_NAMES) * 100  # fallback estimate
    
    # Create progress bar for all parameters across all experiments
    with tqdm(total=total_params, desc=f"Processing normal reference for {sim_col}", ncols=100) as pbar:
        # Iterate through all experiments
        for exp_name in EXP_NAMES:
            try:
                sim_data, sim_param = data_and_param_path(exp_name, read_file=True)
                
                # Check if the column exists in this experiment
                if sim_col not in sim_data.collect_schema().names():
                    print(f"Warning: Column '{sim_col}' not found in experiment {exp_name}")
                    continue
                
                # Filter out prerun and last cycle (following preprocessing patterns)
                sim_data = (sim_data
                           .filter(pl.col("cycle") >= 5)  # Remove first 5 cycles
                           .filter(pl.col("cycle") < pl.max("cycle")))  # Remove last cycle
                
                # Get unique parameter indices
                param_indices = sim_param.collect()["param_idx"].unique().sort()
                
                # Iterate through param/cycle combinations to normalize xdata
                for param_idx in param_indices:
                    # Update progress bar for each parameter
                    pbar.update(1)
                    
                    # Get all cycles for this parameter
                    param_data = (sim_data
                                 .filter(pl.col("param_idx") == param_idx)
                                 .sort(["cycle", "step"])
                                 .collect())
                    
                    if param_data.is_empty():
                        continue
                    
                    # Process each cycle separately
                    cycles = param_data["cycle"].unique().sort()
                    
                    for cycle_num in cycles:
                        cycle_data = param_data.filter(pl.col("cycle") == cycle_num)
                        
                        if len(cycle_data) < 10:  # Skip cycles with too few data points
                            continue
                        
                        # Get angle data for this cycle
                        angles = cycle_data[sim_col].to_numpy()
                        
                        # Create normalized time axis (0 to 100% of gait cycle)
                        original_time = np.linspace(0, 100, len(angles))
                        
                        # Interpolate to standard gait cycle points (0, 1, 2, ..., 100)
                        if len(angles) > 1:  # Need at least 2 points to interpolate
                            try:
                                f = interpolate.interp1d(original_time, angles, 
                                                       kind='linear', bounds_error=False, 
                                                       fill_value='extrapolate')
                                normalized_angles = f(target_gait_cycle)
                                all_normalized_trajectories.append(normalized_angles)
                            except Exception as e:
                                print(f"Warning: Interpolation failed for {exp_name}, param {param_idx}, cycle {cycle_num}: {e}")
                                continue
                
            except Exception as e:
                print(f"Error processing experiment {exp_name}: {e}")
                continue
    
    if not all_normalized_trajectories:
        print(f"Warning: No valid trajectories found for {sim_col}")
        return target_gait_cycle, np.zeros_like(target_gait_cycle), np.zeros_like(target_gait_cycle)
    
    # Convert to numpy array and compute statistics
    all_trajectories = np.array(all_normalized_trajectories)
    mean_trajectory = np.mean(all_trajectories, axis=0)
    std_trajectory = np.std(all_trajectories, axis=0)
    
    print(f"Processed {len(all_normalized_trajectories)} trajectories for {sim_col}")
    
    return target_gait_cycle, mean_trajectory, std_trajectory


def plot_simulation_reference(ax, sim_col: str, multiplier: float = 1.0):
    """
    Plot simulation reference data from EXP_NAMES
    
    Args:
        ax: Matplotlib axis
        sim_col: Column name to extract from simulation data
        multiplier: Multiplier to apply to the data (default: 1.0)
    """
    try:
        time_axis, mean_trajectory, std_trajectory = get_simulation_reference_data(sim_col)
        
        # Apply multiplier
        mean_trajectory *= multiplier
        std_trajectory *= multiplier
        
        # Plot mean line
        ax.plot(time_axis, mean_trajectory, 
               '-', color='red', linewidth=2)
        
        # Add std deviation band
        ax.fill_between(time_axis, mean_trajectory - std_trajectory, mean_trajectory + std_trajectory,
                        color='red', alpha=0.2)
        
    except Exception as e:
        print(f"Warning: Could not load simulation reference data for {sim_col}: {e}")


def get_simulation_column_for_pathology(pathology_config: dict) -> str:
    """
    Map pathology configuration to simulation column name
    
    Args:
        pathology_config: Configuration dictionary for the pathology
        
    Returns:
        Simulation column name
    """
    # Mapping based on pathology title/type
    title = pathology_config.get('title', '')
    
    if 'Ankle' in title or 'ankle' in pathology_config.get('ylabel', '').lower():
        return 'angle_AnkleR'  # Ankle angle column
    elif 'Knee' in title or 'knee' in pathology_config.get('ylabel', '').lower():  
        return 'angle_KneeR'   # Knee angle column
    elif 'Torso' in title or 'sway' in pathology_config.get('ylabel', '').lower():
        return 'sway_Torso_X'      # Torso Y position for sway
    else:
        # Default fallback
        return 'angle_AnkleR'


def plot_pathology_kinematics(ax, pathology_config: dict) -> str:
    """
    Plot kinematics data for a pathology group
    
    Args:
        ax: Matplotlib axis
        pathology_config: Configuration dictionary for the pathology
    
    Returns:
        Joint name from the first CSV file
    """
    csv_paths = pathology_config['csv_paths']
    reference_csv = pathology_config.get('reference_csv')
    
    # Plot reference data first
    if reference_csv is not None:
        # Load CSV reference data (already normalized to 0-100% by load_csv_data)
        try:
            joint_name, ref_x_data, ref_y_data = load_csv_data(reference_csv)
            # Plot reference data with CSV multiplier if needed
            multiplier = pathology_config.get('multiplier', 1.0)
            ax.plot(ref_x_data, ref_y_data * multiplier, '-', color='red', linewidth=2, 
                   label='Simulation (normal body)', alpha=0.9)
        except Exception as e:
            print(f"Warning: Could not load reference CSV data from {reference_csv}: {e}")
    else:
        # Use simulation reference data from EXP_NAMES
        sim_col = get_simulation_column_for_pathology(pathology_config)
        multiplier = pathology_config.get('multiplier', 1.0)
        plot_simulation_reference(ax, sim_col, multiplier)
    
    # Plot multiple CSV files if available
    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=len(csv_paths)-1)
    if len(csv_paths) > 1:
        colors = [cmap(norm(i)) for i in range(len(csv_paths))]
    else:
        colors = [cmap(0)]
    joint_names = []
    
    for i, csv_path in enumerate(csv_paths):
        joint_name, x_data, y_data = load_csv_data(csv_path)
        joint_names.append(joint_name)
        
        # Apply multiplier if specified
        if 'multiplier' in pathology_config:
            y_data = y_data * pathology_config['multiplier']
            
        # Use different colors and get filename for label
        color = colors[i % len(colors)]
        filename = Path(csv_path).stem
        # Handle specific filename mappings
        ax.plot(x_data, y_data, color=color, linewidth=2, 
               label=filename, linestyle='-')
    
    # Configure axis
    # ax.set_ylabel(pathology_config['ylabel'], fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(pathology_config['ylim'])
    ax.set_yticks(pathology_config['yticks'])
    ax.grid(True, alpha=0.3)
    
    # Create legend
    legend_handles = []
    legend_labels = []
    
    # Add simulation data legends
    for i, csv_path in enumerate(csv_paths):
        color = colors[i % len(colors)]
        filename = Path(csv_path).stem
        sim_line = mlines.Line2D([], [], color=color, linewidth=2, alpha=0.8)
        legend_handles.append(sim_line)
        filename_map = {
            'footdrop': 'Foot drop',
            'calcaneous': 'Calcaneous',
            'equinus': 'Equinus',
            'crouch': 'Crouch',
            'waddling': 'Waddling'
        }
        filename = filename_map.get(filename)
        legend_labels.append(filename)

    # Check if reference data exists
    red_lines = [line for line in ax.get_lines() if line.get_color() == 'red' or 
                (hasattr(line.get_color(), '__iter__') and line.get_color()[0] == 1.0)]
    red_fills = [coll for coll in ax.collections if hasattr(coll, 'get_facecolor') and 
                len(coll.get_facecolor()) > 0]
    
    if red_lines or red_fills:
        # Create merged reference legend entry
        ref_line = mlines.Line2D([], [], color='red', linewidth=2, alpha=0.9)
        ref_patch = Patch(facecolor='red', edgecolor='red', alpha=0.2)
        legend_handles.append((ref_line, ref_patch))
        
        # Use appropriate label based on reference type
        if pathology_config.get('reference_csv') is not None:
            legend_labels.append('Simulation (normal body)')
        else:
            legend_labels.append('normal')
            # legend_labels.append('normal (±1SD)')
    
    lg = ax.legend(legend_handles, legend_labels, fontsize=FONT_SIZE_LABEL-6, loc='upper left' if not 'Ankle' in pathology_config['title'] else 'upper right')
    lg.get_frame().set_alpha(0.3)
    ax.set_title(pathology_config['title'], fontsize=FONT_SIZE_LABEL, pad=10)
    
    return joint_names[0] if joint_names else "Multiple"


def place_render_images(axes_list: List, image_filenames: List[str], column_name: str):
    """
    Place render images in the provided axes grid
    
    Args:
        axes_list: List of matplotlib axes for placing images (flattened from grid)
        image_filenames: List of image filenames
        column_name: Name of the column (for labeling)
    """
    # Handle both cases: when we have enough images and when we have fewer images than axes
    num_axes = len(axes_list)
    num_images = len(image_filenames)
    
    for i, ax in enumerate(axes_list):
        if i < num_images:
            # Place actual image
            filename = image_filenames[i]
            image_path = f"{IMAGE_ROOT}/{filename}"
            
            try:
                # Use the insert_image utility function
                insert_image(ax, image_path, label=None, fontsize=10)
            except Exception as e:
                print(f"Warning: Could not load image {image_path}: {e}")
                # Create placeholder with image name
                ax.text(0.5, 0.5, f"Image {i+1}\n{filename}", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=FONT_SIZE_LABEL,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            # Empty axes - create placeholder or hide
            ax.text(0.5, 0.5, f"Empty\n{i+1}", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=FONT_SIZE_LABEL, alpha=0.5,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.3))
            ax.set_xlim(0, 1) 
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])


# ============ CLICK COORDINATE PARSING FUNCTIONS ============

def parse_click_coordinates(coord_text: str) -> Dict[int, Tuple[float, float]]:
    """
    Parse click coordinate input string into axis index and (x, y) coordinates.
    
    Args:
        coord_text: String containing click coordinates in format:
                   "Clicked ax[N]: x=X.X, y=Y.Y"
                   
    Returns:
        Dictionary mapping axis index to (x, y) coordinate tuples
    """
    coordinates = {}
    
    # Pattern to match "Clicked ax[N]: x=X.X, y=Y.Y"
    pattern = r'Clicked ax\[(\d+)\]: x=([\d.]+), y=([\d.]+)'
    
    for line in coord_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.match(pattern, line)
        if match:
            ax_idx = int(match.group(1))
            x_coord = float(match.group(2))
            y_coord = float(match.group(3))
            coordinates[ax_idx] = (x_coord, y_coord)
        else:
            print(f"Warning: Could not parse coordinate line: {line}")
    
    return coordinates


def plot_click_circles(fig, layout: Dict, click_coords: Dict[int, Tuple[float, float]], 
                      circle_color: str = 'blue'):
    """
    Plot blue circles at specified click coordinates on the figure.
    
    Args:
        fig: matplotlib Figure
        layout: Layout dictionary from configure_merged_layout
        click_coords: Dictionary mapping axis index to (x, y) coordinates
        circle_color: Color of the circles (default: 'blue')
        circle_size: Size of the circles (default: 50)
    """
    # Create a flat list of all axes in order they appear in the layout
    assert len(COMMENT_TEXTS) == len(click_coords), f"Number of click coordinates must match number of comment texts: {len(COMMENT_TEXTS)} != {len(click_coords)}"
    assert len(COMMENT_TEXTS) == len(CIRCLE_SIZES), f"Number of comment texts must match number of circle sizes: {len(COMMENT_TEXTS)} != {len(CIRCLE_SIZES)}"
    assert len(COMMENT_TEXTS) == len(COMMENT_Y_OFFSETS), f"Number of comment texts must match number of comment y offsets: {len(COMMENT_TEXTS)} != {len(COMMENT_Y_OFFSETS)}"

    all_axes = []
    
    # Add pathology kinematics axes (indices 5, 6, 7)
    all_axes.append(layout["pathology"]["left"]["kinematics"])
    all_axes.extend(layout["pathology"]["left"]["render"])
    all_axes.append(layout["pathology"]["center"]["kinematics"])
    all_axes.extend(layout["pathology"]["center"]["render"])
    all_axes.append(layout["pathology"]["right"]["kinematics"])
    all_axes.extend(layout["pathology"]["right"]["render"])
    
    # Plot circles at specified coordinates
    for idx, (ax_idx, (x_coord, y_coord)) in enumerate(click_coords.items()):
        if ax_idx < len(all_axes):
            # Set zorder to highest value to ensure visibility
            ax = all_axes[ax_idx]
            ax.set_zorder(1000)
            if CIRCLE_SIZES[idx] > 0:
                ax.scatter(x_coord, y_coord, c='none', s=CIRCLE_SIZES[idx],
                        marker='o', alpha=0.8, edgecolors=circle_color, linewidths=2.5,
                        zorder=100)
                ax.text(x_coord, y_coord + COMMENT_Y_OFFSETS[idx], COMMENT_TEXTS[idx], fontsize=FONT_SIZE_LABEL-4, ha='center', va='top', zorder=100)
                print(f"Added blue ring at ax[{ax_idx}]: ({x_coord}, {y_coord})")
            else:
                ax.text(0.5, 0.02, COMMENT_TEXTS[idx], fontsize=FONT_SIZE_LABEL-4, ha='center', va='bottom', zorder=100, transform=ax.transAxes)
        else:
            print(f"Warning: Axis index {ax_idx} is out of range (max: {len(all_axes)-1})")

    
# ============ MERGED LAYOUT FUNCTIONS ============

def configure_merged_layout(figsize: Tuple[int, int] = (16, 10)) -> Tuple[plt.Figure, Dict]:
    """
    Configure the merged vertical split layout:
    - Top: normal_kinematics (5 horizontal subplots)
    - Bottom: joint_kinematics4 (3-column pathology layout)
    
    Args:
        figsize: Figure size tuple
        
    Returns:
        Tuple of (figure, layout_dict)
    """
    # Create figure with vertical split (top: normal, bottom: pathology)
    fig = plt.figure(figsize=figsize)
    gs_bottom = gridspec.GridSpec(
        1, 3, width_ratios=[1, 1, 1], 
        wspace=FIGURE_MARGINS['wspace'],
        hspace=0.2
    )
    
    # LEFT: 2 rows (Row 1 = kinematics, Row 2 = 4x3 render grid)
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom[0], hspace=0.25, 
        height_ratios=[1, 2]
    )
    ax_left_kinematics = fig.add_subplot(gs_left[0, 0])
    
    # Create 4x3 grid for left render images (3-column layout)
    gs_left_render = gridspec.GridSpecFromSubplotSpec(
        4, 3, subplot_spec=gs_left[1, 0], hspace=0.05, wspace=0.02
    )
    axes_left_render = [[fig.add_subplot(gs_left_render[i, j]) for j in range(3)] for i in range(4)]
    left_row_titles = ["Normal", "Calcaneous", "Foot drop", "Equinus"]
    for i in range(4):
        ax = axes_left_render[i][0]
        ax.text(0.0, 1.0, left_row_titles[i], ha='left', va='top', fontsize=FONT_SIZE_LABEL-6, transform=ax.transAxes, zorder=100, bbox=dict(facecolor='white', edgecolor='none', pad=1))
    left_col_title = ["Initial contact", "Midstance", "Foot off"]
    for i in range(3):
        ax = axes_left_render[0][i]
        ax.text(0.5, 1.075, left_col_title[i], ha='center', va='bottom', fontsize=FONT_SIZE_LABEL-4, transform=ax.transAxes, zorder=100)

    axes_left_render = [ax for row in axes_left_render for ax in row]  # Flatten to 1D list

    # CENTER: 2 rows (Row 1 = kinematics, Row 2 = 2x2 render grid)
    gs_center = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom[1], hspace=0.25, 
        height_ratios=[1, 2]
    )
    ax_center_kinematics = fig.add_subplot(gs_center[0, 0])
    
    # Create 2x2 grid for center render images (2-column layout)
    gs_center_render = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_center[1, 0], hspace=0.05, wspace=0.05
    )
    axes_center_render = [[fig.add_subplot(gs_center_render[i, j]) for j in range(2)] for i in range(2)]
    center_row_titles = ["Normal", "Crouch"]
    for i in range(2):
        ax = axes_center_render[i][0]
        ax.text(-0.05, 0.5, center_row_titles[i], ha='center', va='bottom', fontsize=FONT_SIZE_LABEL-4, rotation=90, transform=ax.transAxes, zorder=100)
    center_col_title = ["Initial contact", "Foot off"]
    for i in range(2):
        ax = axes_center_render[0][i]
        ax.text(0.5, 1.05, center_col_title[i], ha='center', va='bottom', fontsize=FONT_SIZE_LABEL-4, transform=ax.transAxes, zorder=100)
    axes_center_render = [ax for row in axes_center_render for ax in row]  # Flatten to 1D list

    # RIGHT: 2 rows (Row 1 = kinematics, Row 2 = 2x2 render grid)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_bottom[2], hspace=0.25, 
        height_ratios=[1, 2]
    )
    ax_right_kinematics = fig.add_subplot(gs_right[0, 0])
    
    # Create 2x2 grid for right render images (2-column layout)
    gs_right_render = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_right[1, 0], hspace=0.05, wspace=0.05
    )
    axes_right_render = [[fig.add_subplot(gs_right_render[i, j]) for j in range(2)] for i in range(2)]
    right_row_titles = ["Normal", "Waddling"]
    for i in range(2):
        ax = axes_right_render[i][0]
        ax.text(-0.05, 0.5, right_row_titles[i], ha='center', va='bottom', fontsize=FONT_SIZE_LABEL-4, rotation=90, transform=ax.transAxes, zorder=100)
    right_col_title = ["Right midstance", "Left midstance"]
    for i in range(2):
        ax = axes_right_render[0][i]
        ax.text(0.5, 1.05, right_col_title[i], ha='center', va='bottom', fontsize=FONT_SIZE_LABEL-4, transform=ax.transAxes, zorder=100)
    axes_right_render = [ax for row in axes_right_render for ax in row]  # Flatten to 1D list

    # Configure pathology kinematics axes
    pathology_kin_axes = [ax_left_kinematics, ax_center_kinematics, ax_right_kinematics]
    for ax in pathology_kin_axes:
        ax.set_xticks([0, 50, 100])

    # Add x-axis labels
    # Bottom section x-label
    # bottom_bbox = [ax.get_position() for ax in pathology_kin_axes]
    # bottom_x0 = min(b.x0 for b in bottom_bbox)
    # bottom_x1 = max(b.x1 for b in bottom_bbox)
    # bottom_y0 = min(b.y0 for b in bottom_bbox)
    # fig.text((bottom_x0 + bottom_x1)/2, bottom_y0 - 0.04, "Gait Cycle (%)", ha="center", va="top", fontsize=12)

    # Return dictionary for easy access
    layout = {
        "pathology": {
            "left": {"kinematics": ax_left_kinematics, "render": axes_left_render},
            "center": {"kinematics": ax_center_kinematics, "render": axes_center_render}, 
            "right": {"kinematics": ax_right_kinematics, "render": axes_right_render},
        }
    }
    
    return fig, layout


def create_merged_analysis():
    """
    Create a comprehensive merged analysis plot with vertical split layout
    """
    # Configure merged layout
    fig, layout = configure_merged_layout()
    
    # TOP SECTION: Plot normal kinematics
    layout["pathology"]["center"]["kinematics"].set_xlabel(r"Gait Cycle (\%)", fontsize=FONT_SIZE_LABEL)

    
    # BOTTOM SECTION: Plot pathology kinematics
    for column_name, pathology_config in PATHOLOGY_GROUPS.items():
        # Plot kinematics
        kinematics_ax = layout["pathology"][column_name]["kinematics"]
        plot_pathology_kinematics(kinematics_ax, pathology_config)
        
        # Place render images
        render_axes = layout["pathology"][column_name]["render"]
        render_images = pathology_config.get('render_images', [])
        place_render_images(render_axes, render_images, column_name)
    
    # Parse and plot click coordinates as blue circles
    try:
        click_coords = parse_click_coordinates(CLICK_COORDINATES)
        if click_coords:
            plot_click_circles(fig, layout, click_coords)
            print(f"Plotted {len(click_coords)} blue circles from click coordinates")
    except Exception as e:
        print(f"Warning: Failed to plot click coordinates: {e}")
    
    # Add overall titles
    # fig.suptitle('Kinematics and visualization of simulated gait',
            # fontsize=18)

    # Apply figure margins
    fig.subplots_adjust(
        left=FIGURE_MARGINS['left'],
        right=FIGURE_MARGINS['right'], 
        top=FIGURE_MARGINS['top'],
        bottom=FIGURE_MARGINS['bottom'],
    )
    
    filename = f"plot/supp_k7.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight', transparent=True)
    print(f"Figure saved as: {filename}")

    return fig, layout


def main():
    """Main function to create merged kinematics analysis plot"""
    # Validate that all CSV files exist for pathology analysis
    for column_name, pathology_config in PATHOLOGY_GROUPS.items():
        csv_paths = pathology_config.get('csv_paths', [])
        for csv_path in csv_paths:
            if not Path(csv_path).exists():
                print(f"Error: CSV file not found for {column_name}: {csv_path}")
                return
        
        # Validate reference CSV if specified
        reference_csv = pathology_config.get('reference_csv')
        if reference_csv is not None and not Path(reference_csv).exists():
            print(f"Error: Reference CSV file not found for {column_name}: {reference_csv}")
            return
    
    # Create the merged plot
    create_merged_analysis()


if __name__ == "__main__":
    main()