import re
import os
import os.path as osp
import sys
import json
import pickle
import shutil
import curses
import subprocess
import resource
from glob import glob
from io import BytesIO
from random import randint
from time import time, sleep
from functools import partial
from datetime import datetime
from math import ceil, floor, sqrt, isclose
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Type, Sequence

import yaml
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from pathlib import Path

import gc
import polars as pl
from polars import LazyFrame as Lf
from polars import DataFrame as Df

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # Points to project-exo-plore/
PLOT_SAVE_DIR = PROJECT_ROOT / "plot"
PLOT_SAVE_DIR.mkdir(exist_ok=True)  # Ensure plot directory exists
REF_STEP = 0.67
REF_CADENCE = 2 / 1.1
REF_CADENCE_RPM = REF_CADENCE * 60
REF_VELOCITY = REF_STEP * REF_CADENCE


def timestamp() -> str:
    return datetime.now().strftime("%m%d_%H%M%S")


def limit_memory(max_gb):
    max_bytes = max_gb * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))


def ensure_param_idx_type(df: Union[Lf, Df]) -> Union[Lf, Df]:
    """
    Ensures that param_idx column has consistent i64 type for join operations.
    This prevents SchemaError when joining DataFrames with different param_idx types.

    :param df: LazyFrame or DataFrame to check and fix
    :return: DataFrame/LazyFrame with param_idx as i64 type
    """
    # Check if param_idx column exists
    if isinstance(df, pl.LazyFrame):
        schema = df.collect_schema()
        if "param_idx" in schema and schema["param_idx"] != pl.Int64:
            print(f"[Type Fix] Converting param_idx from {schema['param_idx']} to Int64")
            df = df.with_columns(pl.col("param_idx").cast(pl.Int64))
    else:  # pl.DataFrame
        if "param_idx" in df.columns and df.schema["param_idx"] != pl.Int64:
            print(f"[Type Fix] Converting param_idx from {df.schema['param_idx']} to Int64")
            df = df.with_columns(pl.col("param_idx").cast(pl.Int64))

    return df


def normalize(df: Lf, operand_lbl: List[str], data_path: Path) -> Tuple[Lf, Df]:
    """
    Normalizes the data using native Polars operations
    :param df: LazyFrame to normalize
    :param operand_lbl: the columns to normalize
    :param data_path: the path for the given data
    :return: the DataFrame statistics
    """
    print(f"[Normalize] Start normalizing {data_path}")
    statistics_path = str(data_path).replace(".gt.parquet", ".stat.csv")
    statistics_path = Path(statistics_path)

    # Ensure param_idx has consistent type before any operations
    # df = ensure_param_idx_type(df)

    # Filter out param_idx from normalization columns as it should not be normalized
    operand_lbl_filtered = [col for col in operand_lbl if col != "param_idx"]

    # Compute means and stds directly using LazyFrame aggregation, avoiding collect
    agg_exprs = []
    for col in operand_lbl_filtered:
        agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
        agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))
    stats_row = df.select(agg_exprs).collect(streaming=True).to_dicts()[0]

    # Compose clean statistics DataFrame
    mean_row = {"describe": "mean"}
    std_row = {"describe": "std"}
    for col in operand_lbl_filtered:
        mean_row[col] = stats_row[f"{col}_mean"]
        std_row[col] = stats_row[f"{col}_std"]
    statistics = Df([mean_row, std_row])

    # Apply normalization to LazyFrame
    normalize_exprs = []
    for col in operand_lbl_filtered:
        mean_val = mean_row[col]
        std_val = std_row[col]

        if std_val is None or isclose(std_val, 0.0):
            print(f"[Warning] {col} has std=0. normalize to 0.")
            normalize_exprs.append(pl.lit(0.0).alias(col))
        else:
            normalize_exprs.append(((pl.col(col) - mean_val) / std_val).alias(col))

    df = df.with_columns(normalize_exprs)

    # Save to CSV
    statistics.write_csv(statistics_path)
    print(f"[Normalize] Normalized finished for {data_path}")
    gc.collect()
    return df, statistics


def normalize_from_stat(df: Union[Lf, Df], stat_df: Df) -> Union[Lf, Df]:
    """
    Normalizes the all columns of data using the given statistics
    :param df:
    :param stat_df:
    :return:
    """
    # Ensure param_idx has consistent type before any operations
    # df = ensure_param_idx_type(df)

    # Check if input is LazyFrame or DataFrame and get column names accordingly
    if isinstance(df, pl.LazyFrame):
        columns = df.collect_schema().names()
    else:  # pl.DataFrame
        columns = df.columns

    # Create normalization expressions (excluding param_idx)
    normalize_exprs = []

    for col in columns:
        if col == "param_idx":
            # Skip param_idx normalization as it's an identifier
            continue

        if col not in stat_df.columns:
            print(f"[Warning] {col} not found in statistics, skipping normalization for this column")
            continue

        # Get mean and std for this column
        mean_val = stat_df.filter(pl.col("describe") == "mean")[col].item()
        std_val = stat_df.filter(pl.col("describe") == "std")[col].item()

        if isclose(std_val, 0.0):
            print(f"[Warning] {col} has std=0. normalize to 0.")
            normalize_exprs.append(pl.lit(0.0).alias(col))
        else:
            normalize_exprs.append(((pl.col(col) - mean_val) / std_val).alias(col))

    return df.with_columns(normalize_exprs)


def normalize_from_stat_tensor(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the all columns of data using the given statistics
    :param data:
    :param mean:
    :param std:
    :return:
    """
    prev_device = data.device
    data = data.to(mean.device)
    std_zero_mask = torch.isclose(std, torch.tensor(0.0))
    if std_zero_mask.any():
        zero_std_indices = torch.where(std_zero_mask)[0]
        print(f"[Warning] Columns {zero_std_indices.tolist()} have std=0. Normalizing to 0 for these columns.")
    normalized_data = torch.where(std_zero_mask, torch.tensor(0.0), (data - mean) / std)
    normalized_data = normalized_data.to(prev_device)
    return normalized_data


def denormalize(df: Union[Lf, List[Lf], Df, List[Df]], stat_df: Df, operand_lbl: List[str] = None) -> Union[Lf, List[Lf], Df, List[Df]]:
    """
    Denormalizes the data
    :param df: to denormalize
    :param stat_df: the statistics (mean, std) path to use for de-normalization
    :param operand_lbl: the columns to denormalize. If None, all columns are denormalized
    :return:
    """
    # Check if input is a single item or a list
    is_single_item = not isinstance(df, list)

    # Convert single item to list for processing
    if is_single_item:
        df_list = [df]
    else:
        df_list = df

    result = []

    for d in df_list:
        # Check if it's LazyFrame or DataFrame and get column names accordingly
        if isinstance(d, pl.LazyFrame):
            columns = d.collect_schema().names()
        else:  # pl.DataFrame
            columns = d.columns

        # Determine columns to denormalize
        if operand_lbl is None:
            col_denorm = columns
        else:
            col_denorm = operand_lbl

        # Create denormalization expressions
        denorm_exprs = []

        for col in col_denorm:
            if col == "param_idx":
                continue
            if col not in stat_df.columns:
                # Keep the column as is if not in statistics
                denorm_exprs.append(pl.col(col))
                continue

            # Get mean and std for this column
            # Extract mean and std values for the column from stat_df DataFrame
            mean_val = stat_df.filter(pl.col("describe") == "mean")[col].item()
            std_val = stat_df.filter(pl.col("describe") == "std")[col].item()

            # Denormalize: value * std + mean
            if std_val is None or std_val == 0.0 or np.isnan(std_val):
                print(f"[Warning] {col} has std=0. Denormalizing to mean.")
                denorm_exprs.append(pl.lit(mean_val).alias(col))
            else:
                denorm_exprs.append((pl.col(col) * std_val + mean_val).alias(col))

        # Apply denormalization
        if denorm_exprs:
            result.append(d.with_columns(denorm_exprs))
        else:
            result.append(d)

    # Return single item if input was single item, otherwise return list
    if is_single_item:
        return result[0]
    else:
        return result


def denormalize_tensor(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Denormalizes the data tensor.
    :param data: Tensor of data to be denormalized.
    :param mean: Tensor of means for each column.
    :param std: Tensor of standard deviations for each column.
    :return: Denormalized tensor.
    """
    denormalized_data = data.clone()
    return denormalized_data * std + mean


def tensor_to_df(data: torch.Tensor, col_lbl: List[str]) -> pl.DataFrame:
    """
    Converts a tensor to a DataFrame
    :param data: to convert
    :param col_lbl: the column labels
    :return: the DataFrame
    """
    if data.ndim == 2 and data.shape[1] == 1:
        data = data.squeeze(-1)
    if data.ndim > 1:
        data = data.flatten(end_dim=-2)
    return pl.DataFrame(data.cpu().numpy(), schema=col_lbl)


def np_to_df(np_arr: np.ndarray, col_lbl: List[str]) -> pl.DataFrame:
    """
    Converts a numpy array to a DataFrame
    :param np_arr: to convert
    :param col_lbl: the column labels
    :return: the DataFrame
    """
    if np_arr.ndim == 1:
        np_arr = np_arr.reshape(1, -1)
    assert np_arr.ndim == 2, "Only 1D or 2D arrays are supported"
    assert np_arr.shape[1] == len(col_lbl), "Number of columns do not match"
    return pl.DataFrame(np_arr, schema=col_lbl)


def get_existing_tmux() -> List[str]:
    result = subprocess.run(["tmux", "ls"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return []
    return [line.split(":")[0] for line in result.stdout.decode().splitlines() if line]


def run_in_tmux(_commands: List[str], session_name: str= "my_tmux_session"):
    exist = get_existing_tmux()
    while session_name in exist:
        session_name += "a"
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name])

    # Loop through the commands and create a new pane for each
    for i, command in enumerate(_commands):
        if i == 0:
            subprocess.run(["tmux", "send-keys", "-t", f"{session_name}.0", command, "C-m"])
        else:
            subprocess.run(["tmux", "split-window", "-t", session_name, "-v"])
            subprocess.run(["tmux", "send-keys", "-t", f"{session_name}.{i}", command, "C-m"])
            subprocess.run(["tmux", "select-layout", "-t", session_name, "tiled"])

    # Attach to the tmux session
    subprocess.run(["tmux", "attach-session", "-t", session_name])


def data_and_param_path(exp_name: str, read_file: bool = False) -> Union[Tuple[Path, Path], Tuple[Lf, Lf]]:
    """
    :param exp_name: grf_0918_001051
    :param read_file: if True, returns the DataFrame instead of the path
    :return:
    """
    exp_dir = PROJECT_ROOT / "experiment_data/simulation" / exp_name
    if not exp_dir.is_dir():
        exp_dir = exp_dir.parent
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    if (exp_dir / "data.parquet").exists():
        data_path = exp_dir / "data.parquet"
    else:
        raise FileNotFoundError(f"Data file not found in {exp_dir}")
    param_path = exp_dir / "param.csv"
    if not param_path.exists():
        param_path = exp_dir / "param.parquet"
    assert param_path.exists(), f"{param_path} does not exist."

    if read_file:
        return read_lf(data_path), read_lf(param_path)
    else:
        return data_path, param_path


def reverse_nested_dict(
    nested_dict: Union[Dict[Any, Any], List[Any]],
    parent_keys: List[str] = None,
    reversed_dict: Dict[Any, List[Tuple[str, ...]]] = None
) -> Dict[Any, List[Tuple[str, ...]]]:
    """
    Recursively reverses a nested dictionary or list.

    Args:
        nested_dict (dict or list): The dictionary or list to reverse.
        parent_keys (list, optional): Accumulates the keys along the current path.
        reversed_dict (dict, optional): The dictionary to store reversed key-value pairs.

    Returns:
        dict: The reversed dictionary.
    """
    if parent_keys is None:
        parent_keys = []
    if reversed_dict is None:
        reversed_dict = {}

    if isinstance(nested_dict, dict):
        for key, value in nested_dict.items():
            current_keys = parent_keys + [str(key)]
            reverse_nested_dict(value, current_keys, reversed_dict)
    elif isinstance(nested_dict, list):
        for index, item in enumerate(nested_dict):
            # Use index as part of the path if needed, or skip it
            # current_keys = parent_keys + [str(index)]
            # For muscle groups, indices might not be necessary
            reverse_nested_dict(item, parent_keys, reversed_dict)
    else:
        # Assume it's a scalar value
        combined_keys = tuple(parent_keys)
        if nested_dict in reversed_dict:
            reversed_dict[nested_dict].append(combined_keys)
        else:
            reversed_dict[nested_dict] = [combined_keys]

    return reversed_dict


def load_muscle_group(group_file_path: str = "data/muscle_group.yaml") -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Load muscle group from the muscle_group.yaml file
    :return:
        1. flattened muscle groups
        2. muscle membership dictionary. key: muscle name, value: list of muscle group names
            (muscle can be included by multiple groups)
    """
    with open(PROJECT_ROOT / group_file_path, 'r') as file:
        muscle_group = yaml.safe_load(file)
    muscle_membership = reverse_nested_dict(muscle_group)  # Dict[str, List[Tuple[str]]]
    muscle_membership_flatten = {}
    muscle_group_flatten = []
    for muscle, groups in muscle_membership.items():
        muscle_membership_flatten[muscle] = []
        for group in groups:
            label = '_'.join(group)
            muscle_group_flatten.append(label)
            muscle_membership_flatten[muscle].append(label)
    muscle_group_flatten = list(set(muscle_group_flatten))
    return muscle_group_flatten, muscle_membership_flatten


# def read_df(data_path: Path) -> pl.DataFrame:
#     if data_path.suffix == ".parquet":
#         return pl.read_parquet(data_path)
#     else:
#         return pl.read_csv(data_path)

# def read_lf(data_path: Path) -> pl.LazyFrame:
#     if data_path.suffix == ".parquet":
#         return pl.scan_parquet(data_path)
#     else:
#         return pl.scan_csv(data_path)

def read_df(data_path: Path) -> pl.DataFrame:
    if data_path.suffix == ".parquet":
        df = pl.read_parquet(data_path)
    else:
        df = pl.read_csv(data_path)
    # If the column 'param_idx' exists, check its dtype and convert to Int32 if needed
    if "param_idx" in df.columns:
        if df["param_idx"].dtype != pl.Int32:
            df = df.with_columns([pl.col("param_idx").cast(pl.Int32)])
    for col in df.columns:
        if df[col].dtype == pl.Float64 and col != "param_idx":
            df = df.with_columns([pl.col(col).cast(pl.Float32)])
    return df

def read_lf(data_path: Path) -> pl.LazyFrame:
    if data_path.suffix == ".parquet":
        lf = pl.scan_parquet(data_path)
    else:
        lf = pl.scan_csv(data_path)
    # If the column 'param_idx' exists, check its dtype and convert to Int32 if needed
    schema = lf.collect_schema()
    if "param_idx" in schema.names():
        if schema["param_idx"] != pl.Int32:
            lf = lf.with_columns([pl.col("param_idx").cast(pl.Int32)])
    for col in schema.names():
        if schema[col] == pl.Float64 and col != "param_idx":
            lf = lf.with_columns([pl.col(col).cast(pl.Float32)])
    return lf


###############################################################################
# Plot utilities (formerly python/plot/util.py)
###############################################################################

import itertools

try:
    from alphashape import alphashape
    from shapely.geometry import Polygon, MultiPolygon
except ImportError:
    alphashape = None
    Polygon = None
    MultiPolygon = None
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

import matplotlib
if os.environ.get('MPLBACKEND') or not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import TwoSlopeNorm, to_rgba
from matplotlib.collections import LineCollection
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as font_manager
from matplotlib.patches import Rectangle
plt.rcParams["font.family"] = "Times New Roman"
FONT_SIZE_LABEL = 22
FONT_SIZE_LEGEND = 10
DEFAULT_FONT_SIZE = 12
KEY_CMD_MAP = {
    '1': 'xy',
    '2': 'yz',
    '3': 'zx'
}
PIE_COLORS = [
    '#FF5733',  # Red-Orange
    '#33FF57',  # Green
    '#3357FF',  # Blue
    '#FF33A1',  # Pink
    '#FF8C33',  # Orange
    '#33FFF1',  # Aqua
    '#8D33FF',  # Purple
    '#FFD133',  # Yellow
    '#33FF8C',  # Mint Green
    '#FF3333',  # Red
    '#33A1FF',  # Light Blue
    '#A1FF33',  # Lime
]

PIE_COLORS0 = plt.rcParams['axes.prop_cycle'].by_key()['color']

def to_01color(r: int, g: int, b: int) -> Tuple[float, float, float]:
    return tuple(c / 255.0 for c in (r, g, b))

def get_color0(idx: int) -> str:
    return PIE_COLORS0[idx % len(PIE_COLORS0)]

def get_color1(idx: int) -> str:
    return PIE_COLORS[idx % len(PIE_COLORS)]

def get_color2(idx: int) -> str:
    cmap = plt.cm.get_cmap('viridis')
    return cmap(idx / len(PIE_COLORS))


def cot_from_exp_name(exp_name: str) -> str:
    """
    Extract experiment name from a string with pattern: (exp_name)_(datetime)_(else)

    For example,
        String like "b20_a_0626_123854_10000+memo__b20_U500_k0_no_mod_state_bhar+_on_0804_105957"
        1. check the memo (memo__b20_U500_k0_no_mod_state_bhar) -> the last part is cot_type
        2. check the exp_name (b20_a_0626_123854_10000) -> the second part is cot_type
        3. if both are not found, raise an error

    Args:
        exp_name

    Returns:
        str: The extracted experiment name (e.g., "b20_a3")
    """
    COT_TYPES = [
        "a", "a2", "a3",
        "m05a", "m05a2", "m05a3",
        "m2a", "m2a2", "m2a3",
        "ma", "ma2", "ma3",
        "a15", "ma15", "m05a15", "m2a15",
        "bhar", "bharx4", "bharx16",
        "a125", "m05a125", "ma125", "m2a125",
    ]

    # Step 1: Check memo pattern first
    memo_pattern = r'\+memo_(.+?)\+'
    memo_match = re.search(memo_pattern, exp_name)

    cot_type = None
    if memo_match:
        memo_part = memo_match.group(1)
        # Extract the last part after the final underscore as cot_type
        memo_parts = memo_part.split('_')
        for i in range(len(memo_parts)):
            potential_cot = '_'.join(memo_parts[i:])
            if potential_cot in COT_TYPES:
                cot_type = potential_cot

    if cot_type is None:
        # Step 2: Pattern: capture everything before first datetime pattern (DDDD_DDDDDD)
        pattern = r'^(.+?)_(\d+_\d+)'
        match = re.match(pattern, exp_name)

        if not match:
            # If no datetime pattern found, return the original string
            raise ValueError(f"No datetime pattern found in exp_name: {exp_name}")

        base_exp_name = match.group(1)
        cot_type = "_".join(base_exp_name.split("_")[1:])
        if cot_type not in COT_TYPES:
            raise ValueError(f"Invalid cot type: could not extract valid COT type from exp_name: {exp_name}")

    if cot_type in ['bharx4', 'bharx16']:
        cot_type = 'bhar'

    return cot_type

def on_key(event):
    fig = plt.gcf()
    axs = fig.get_axes()  # Get all axes (subplots) from the current figure

    if event.key == 'escape':
        plt.close()
        sys.exit()
    elif event.key == 'm':
        for ax in axs:
            ax.view_init(elev=ax.elev, azim=ax.azim + 180)

    cmd = KEY_CMD_MAP.get(event.key)
    if cmd in ['xy', 'yz', 'zx']:
        for ax in axs:
            if isinstance(ax, Axes3D):
                change_ax_view(ax, cmd)
        fig.canvas.draw()


def on_click(event):
    # 축 밖 클릭은 무시
    if event.inaxes is None or event.xdata is None or event.ydata is None:
        print("Clicked outside any Axes")
        return

    ax = event.inaxes
    x, y = event.xdata, event.ydata

    # 축 라벨이 없으면 인덱스로 표기
    try:
        fig = plt.gcf()
        ax_idx = fig.axes.index(ax)
    except ValueError:
        ax_idx = "?"
    ax_label = ax.get_label() or f"ax[{ax_idx}]"

    # 좌표/축 정보 출력
    print(f"Clicked {ax_label}: x={x:.3f}, y={y:.3f}")


def change_ax_view(ax: Axes, view_init: str):
    if view_init == 'xy':
        ax.view_init(elev=90, azim=270)
    elif view_init == 'yz':
        ax.view_init(elev=0, azim=90)
    elif view_init == 'zx':
        ax.view_init(elev=0, azim=0)


def refind_gait_cycle(data: pl.DataFrame) -> pl.DataFrame:
    assert 'gait_cycle' in data.columns and 'value' in data.columns, f"Columns not found in the dataframe. Found columns: {data.columns}"

    new_gait_cycle = np.arange(0, 100, 1)
    interpolation_function = interp1d(
        data['gait_cycle'].to_numpy(),
        data['value'].to_numpy(),
        kind='quadratic',
        fill_value='extrapolate'
    )
    interpolated_values = interpolation_function(new_gait_cycle)
    interpolated_data = pl.DataFrame({
        'gait_cycle': new_gait_cycle,
        'value': interpolated_values
    })

    min_original = data['gait_cycle'].min()
    max_original = data['gait_cycle'].max()
    extrapolated_gait_cycles = interpolated_data.filter(
        (pl.col('gait_cycle') < min_original) |
        (pl.col('gait_cycle') > max_original)
    )['gait_cycle']
    print("Extrapolated gait_cycle values:")
    print(extrapolated_gait_cycles.to_list())

    return interpolated_data


def translate_gait_data(raw_data: pl.DataFrame, trans_cycle: int = 50) -> pl.DataFrame:
    data = raw_data.clone()
    assert 'gait_cycle' in data.columns and 'value' in data.columns, f"Columns not found in the dataframe. Found columns: {data.columns}"
    data = data.with_columns(
        pl.col('gait_cycle') - trans_cycle
    ).with_columns(
        pl.when(pl.col('gait_cycle') < 0)
        .then(pl.col('gait_cycle') + 100)
        .when(pl.col('gait_cycle') >= 100)
        .then(pl.col('gait_cycle') - 100)
        .otherwise(pl.col('gait_cycle'))
        .alias('gait_cycle')
    ).sort('gait_cycle')
    return data


def delay_in_exp(exp_name: str) -> List[float]:
    _, sim_param = data_and_param_path(exp_name, read_file=True)
    delays = sim_param.select("Delay").unique().collect().to_series().to_list()
    return delays


def load_wpd_dataset(name: str) -> pl.DataFrame:
    data_root = "experiment_data/real"
    filename_dict = {
        "cot": "cot/wpd_datasets.csv",
        "gems_cadence": "gems/cadence_speed_wpd_datasets.csv",
        "gems_negative": "gems/negative_wpd_datasets.csv",
    }
    data_path = osp.join(data_root, filename_dict[name])
    # For complex MultiIndex headers, we'll need to handle this differently
    # This is a simplified version - you may need to adjust based on your specific CSV structure
    if name == "cot":
        with open(PROJECT_ROOT / data_path, "r") as f:
            header1 = f.readline().strip().split(",")
            header2 = f.readline().strip().split(",")
        for i in range(int(len(header1) / 2)):
            header1[2 * i + 1] = header1[2 * i]
        columns = [
            f"{h1.strip()}_{h2.strip()}" if h1.strip() else h2.strip()
            for h1, h2 in zip(header1, header2)
        ]
        real_data = pl.read_csv(PROJECT_ROOT / data_path, skip_rows=1, new_columns=columns)
    else:
        real_data = pl.read_csv(PROJECT_ROOT / data_path)
    return real_data


def load_gems_selection() -> pl.DataFrame:
    return pl.DataFrame({
        "velocity": [0.556, 0.833, 1.111, 1.389, 1.667], # m/s
        "Step": [0.351, 0.488, 0.606, 0.709, 0.8], # m
        "Cadence": [1.583, 1.708, 1.833, 1.958, 2.083] # Hz
    })

def load_p20_selection() -> pl.DataFrame:
    return pl.DataFrame({
        "Step": [0.3, 0.7], # m
        "Cadence": [1.2, 1.9] # Hz
    })


def load_angle_gems() -> dict:
    angle_real_data = {}  # velocity -> data (gait_cycle, value)
    for v in [1, 2, 3, 4, 5]:
        angle_real_data[v] = pl.read_csv(PROJECT_ROOT / f"experiment_data/real/gems/GEMS_graph1_angle_data{v}.csv")
    return angle_real_data


def load_vel_gems() -> dict:
    vel_real_data = {}  # velocity -> data (gait_cycle, value)
    for v in [1, 2, 3, 4, 5]:
        vel_real_data[v] = pl.read_csv(PROJECT_ROOT / f"experiment_data/real/gems/GEMS_graph1_vel_data{v}.csv")
    return vel_real_data


def load_power_gems() -> dict:
    power_real_data = {}  # velocity -> data (gait_cycle, value)
    for v in [1, 2, 3, 4, 5]:
        power_real_data[v] = pl.read_csv(PROJECT_ROOT / f"experiment_data/real/gems/GEMS_graph1_power_data{v}.csv")
    return power_real_data


def load_moment_gems() -> dict:
    moment_real_data = {}  # velocity -> data (gait_cycle, value)
    for v in [1, 2, 3, 4, 5]:
        moment_real_data[v] = pl.read_csv(PROJECT_ROOT / f"experiment_data/real/gems/GEMS_graph1_torque_data{v}.csv")
    return moment_real_data


def load_moment_w_delay_gems() -> Dict[float, List[float]]:
    """
    :return: delay -> data (ordered in velocity)
    """
    DATA_DELAYS = [0.05, 0.15, 0.25, 0.35]
    real_data_vels = 5
    real_moment_df = pl.read_csv(PROJECT_ROOT / "experiment_data/real/gems/GEMS_graph3_RMStorque_data.csv")
    real_data = {}
    for delay in DATA_DELAYS:
        delay_data = []
        for v in range(real_data_vels):
            delay_data.append(float(real_moment_df["value"].item(v * len(DATA_DELAYS) + DATA_DELAYS.index(delay))))
        real_data[delay] = delay_data
    return real_data


def load_emg(
        muscle_names: List[str],
        num_data: int = 200,
        per_muscle_source: Dict[str, str] = None,
        gait120_task: str = "LevelWalking",
        gait120_muscle_alias: Dict[str, str] = None,
) -> pl.DataFrame:
    """
    Load EMG profiles for the requested muscles, with per-muscle source selection.

    Sources:
    - "legacy" (default): CSVs under experiment_data/real/emg using the existing mapping.
    - "gait120": Parquet dataset at experiment_data/real/gait120/emg_data.parquet (columns: task, muscle, time_percent, mean_activation, std_activation, count).

    Args:
        muscle_names: List of target muscle names (can include numeric suffix like "Gluteus_Medius1").
        num_data: Number of samples to resample per profile (returns num_data+1 points inclusive).
        per_muscle_source: Optional mapping muscle_name -> {'legacy'|'gait120'}. If missing, defaults to 'legacy'.
        gait120_task: Task name to filter in gait120 dataset (e.g., 'LevelWalking').
        gait120_muscle_alias: Optional mapping from our muscle names to gait120 'muscle' labels.

    Returns:
        pl.DataFrame with columns: 'x' (0..100) and each requested muscle (mean activation).
        For muscles loaded from gait120, also includes '{muscle}_std' columns interpolated from 'std_activation'.
    """
    if per_muscle_source is None:
        per_muscle_source = {
            "Bicep_Femoris_Longus": "gait120",
            "Bicep_Femoris_Short": "gait120",
            "Gastrocnemius_Lateral_Head": "gait120",
            "Gastrocnemius_Medial_Head": "gait120",
            "Gluteus_Maximus": "legacy",
            "Gluteus_Medius": "legacy",
            "Rectus_Femoris": "gait120",
            "Soleus": "gait120",
            "Tibialis_Anterior": "gait120",
            "Vastus_Lateralis": "gait120",
            "Vastus_Medialis": "gait120",
            "Peroneus_Brevis": "gait120",
            "Peroneus_Longus": "gait120",
            "Semitendinosus": "gait120",
        }

    data_root = "experiment_data/real/emg"
    # Legacy CSV mapping
    legacy_muscle_to_file = {
        "Bicep_Femoris_Longus": "BicepFem.csv",
        "Bicep_Femoris_Short": "BicepFem.csv",
        "Gastrocnemius_Lateral_Head": "GasLat.csv",
        "Gluteus_Maximus": "GltMax.csv",
        "Gluteus_Medius": "GltMed.csv",
        "Rectus_Femoris": "RecFem.csv",
        "Soleus": "Sol.csv",
        "Tibialis_Anterior": "TibAnt.csv",
        "Vastus_Lateralis": "VasLat.csv",
    }

    # Default gait120 alias map (desired name -> dataset label)
    default_g120_alias = {
        "Bicep_Femoris_Longus": "Biceps Femoris",
        "Bicep_Femoris_Short": "Biceps Femoris",
        "Gastrocnemius_Lateral_Head": "Gastrocnemius Lateralis",
        "Gastrocnemius_Medial_Head": "Gastrocnemuis Medialis",
        "Gluteus_Maximus": "Gluteus Maximus",
        "Gluteus_Medius": "Gluteus Medius",
        "Rectus_Femoris": "Rectus Femoris",
        "Soleus": "Soleus Lateralis",
        "Tibialis_Anterior": "Tibialis Anterior",
        "Vastus_Lateralis": "Vastus Lateralis",
        "Vastus_Medialis": "Vastus Medialis",
        "Peroneus_Brevis": "Peroneus Brevis",
        "Peroneus_Longus": "Peroneus Longus",
        "Semitendinosus": "Semitendinosus",
    }
    if gait120_muscle_alias is not None:
        default_g120_alias.update(gait120_muscle_alias)

    # Normalize per-muscle source map
    per_muscle_source = per_muscle_source or {}

    # Prepare output X grid [0, 1]
    x_data = np.linspace(0, 1, num_data + 1)
    return_data = pl.DataFrame({"x": x_data})

    # Lazy-load gait120 parquet if needed
    gait120_df = None
    gait120_available_labels = None

    # Helper to get source per muscle, allowing suffix-less fallback
    def resolve_source(name: str) -> str:
        name_wo_suffix = name[:-1] if len(name) > 0 and name[-1].isdigit() else name
        if name in per_muscle_source:
            return per_muscle_source[name]
        if name_wo_suffix in per_muscle_source:
            return per_muscle_source[name_wo_suffix]
        return "legacy"

    for mname in muscle_names:
        mname_wo_suffix = mname[:-1] if len(mname) > 0 and mname[-1].isdigit() else mname
        source = resolve_source(mname)

        if source.lower() in {"legacy", "csv", "emg"}:
            if mname_wo_suffix not in legacy_muscle_to_file:
                raise KeyError(f"Legacy EMG mapping not found for muscle '{mname_wo_suffix}'. Available: {list(legacy_muscle_to_file.keys())}")
            data_path = osp.join(data_root, legacy_muscle_to_file[mname_wo_suffix])
            df = pl.read_csv(PROJECT_ROOT / data_path)
            xi = df["x"].to_numpy()
            yi = df["EMG"].to_numpy()
            f = interp1d(xi, yi, kind='cubic', fill_value='extrapolate')
            y_data = f(x_data)
            return_data = return_data.with_columns(pl.lit(y_data).alias(mname))

        elif source.lower() in {"gait120", "parquet"}:
            if gait120_df is None:
                gait120_df = pl.read_parquet(PROJECT_ROOT / "experiment_data/real/gait120/emg_data.parquet")
                gait120_available_labels = sorted(gait120_df["muscle"].unique().to_list())
            # Resolve alias
            if mname_wo_suffix in default_g120_alias:
                g120_label = default_g120_alias[mname_wo_suffix]
            else:
                # Fallback to a loose formatting attempt
                g120_label = mname_wo_suffix.replace("_", " ")
            df_m = gait120_df.filter((pl.col("task") == gait120_task) & (pl.col("muscle") == g120_label))
            if len(df_m) == 0:
                raise KeyError(
                    f"gait120 EMG not found for muscle '{mname_wo_suffix}' as '{g120_label}' with task '{gait120_task}'. "
                    f"Available muscles: {gait120_available_labels}"
                )
            # Interpolate mean_activation over [0,1]
            xi = (df_m["time_percent"].to_numpy() / 100.0)
            yi_mean = df_m["mean_activation"].to_numpy()
            f_mean = interp1d(xi, yi_mean, kind='cubic', fill_value='extrapolate')
            y_mean = f_mean(x_data)
            return_data = return_data.with_columns(pl.lit(y_mean).alias(mname))

            yi_std = df_m["std_activation"].to_numpy()
            f_std = interp1d(xi, yi_std, kind='cubic', fill_value='extrapolate')
            y_std = f_std(x_data)
            return_data = return_data.with_columns(pl.lit(y_std).alias(f"{mname}_std"))

        else:
            raise ValueError(f"Unknown EMG source '{source}' for muscle '{mname}'. Use 'legacy' or 'gait120'.")

    # Convert x to [0,100] like the previous implementation
    return_data = return_data.with_columns((pl.col('x') * 100).alias('x'))
    return return_data


def load_power_w_delay_gems() -> Dict[float, Dict[str, List[float]]]:
    """
    :return: delay -> signs -> data (ordered in velocity)
    """
    DATA_DELAYS = [0.05, 0.15, 0.25, 0.35]
    signs = ["positive", "negative"]
    real_data_vels = 5
    real_power_df = pl.read_csv(PROJECT_ROOT / "experiment_data/real/gems/GEMS_graph3_MeanPower_data.csv")
    real_data = {}
    for delay in DATA_DELAYS:
        delay_data = {}
        for s in signs:
            sign_data = []
            for v in range(real_data_vels):
                sign_data.append(float(real_power_df["value"].item(DATA_DELAYS.index(delay) * real_data_vels * 2 +
                                                               signs.index(s) * real_data_vels + v)))
            delay_data[s] = sign_data
        real_data[delay] = delay_data
    return real_data


def get_selection_title_and_filename(
        exp_name: str, plot_name: str, selection_method: str = None, selection_file: str = None,
        selection_field: str = None) -> Tuple[str, str]:
    """
    :param exp_name:
    :param plot_name:
    :param selection_method:
    :param selection_file:
    :param selection_field:
    :return:
        1. Title
        2. Filename
    """
    if selection_method is not None:
        title = f"{exp_name} (Method: {selection_method})"
        filename = f"{plot_name}_{selection_method}_selected"
    elif selection_file is not None:
        selection_file = selection_file.replace('.txt', '')
        title = f"{exp_name} (Selection: {selection_file})"
        filename = f"{plot_name}_{selection_file}"
    else:
        title = exp_name
        filename = plot_nameㅅ
    if selection_field is not None:
        title += f" ({selection_field})"
    return title, filename


def axes_pixel_size_and_aspect(ax: Axes):
    """
    Return (width_px, height_px, aspect) for an Axes as rendered on screen.
    aspect = width_px / height_px
    """
    fig = ax.figure
    # Ensure a renderer exists (needed to get a correct window extent)
    fig.canvas.draw()
    bbox = ax.get_window_extent(fig.canvas.get_renderer())
    w_px, h_px = bbox.width, bbox.height
    print(f"width_px: {float(w_px):.2f}, height_px: {float(h_px):.2f}, aspect: {float(w_px / h_px):.4f}")


def moving_average(
    data: Union[pl.Series, pl.DataFrame, torch.Tensor, np.ndarray],
    window_size: int = 10
) -> Union[pl.Series, pl.DataFrame, torch.Tensor]:

    # ---- 입력 정규화 ----
    is_series = isinstance(data, pl.Series)
    if isinstance(data, pl.DataFrame):
        arr = data.to_numpy()                     # (rows, cols)
        data_tensor = torch.tensor(arr, dtype=torch.float32).T.unsqueeze(0)   # (1, C, L)
        columns = data.columns
        post = lambda y: pl.DataFrame(y.squeeze(0).T.numpy(), schema=columns)

    elif isinstance(data, pl.Series):
        arr = data.to_numpy()                     # (L,)
        data_tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,L)
        post = lambda y: pl.Series(y.squeeze().numpy())

    elif isinstance(data, np.ndarray):
        arr = torch.tensor(data, dtype=torch.float32)
        if arr.ndim == 1:   # (L,)
            data_tensor = arr.unsqueeze(0).unsqueeze(0)   # (1,1,L)
            post = lambda y: y.squeeze()
        elif arr.ndim == 2: # (rows, cols)
            data_tensor = arr.T.unsqueeze(0)              # (1,C,L)
            post = lambda y: y.squeeze(0).T
        else:
            raise ValueError("ndarray must be 1D or 2D")
    elif isinstance(data, torch.Tensor):
        if data.ndim == 1:        # (L,)
            data_tensor = data.unsqueeze(0).unsqueeze(0)  # (1,1,L)
            post = lambda y: y.squeeze()
        elif data.ndim == 2:      # (rows, cols)
            data_tensor = data.T.unsqueeze(0)             # (1,C,L)
            post = lambda y: y.squeeze(0).T
        elif data.ndim == 3:      # (N,C,L)
            data_tensor = data
            post = lambda y: y
        else:
            raise ValueError("Tensor must be 1D, 2D or 3D")
    else:
        raise ValueError("Unsupported input type")

    pad = window_size // 2

    # ---- 이동평균 계산 ----
    x_pad = F.pad(data_tensor, (pad, pad), mode="replicate")
    y = F.avg_pool1d(x_pad, kernel_size=window_size, stride=1)

    # ---- 원래 타입으로 복원 ----
    return post(y)

def set_axes(plot, num: int, use_3d: bool = False, num_plot_row: Optional[int] = None, elev: int = None, azim: int = None,
             num_plot_col: Optional[int] = None, figsize: int = 5, wspace: float = None, hspace: float = None, remove_frame: bool = False,
             wsize: int = None, hsize: int = None, idx3d: List[int] = None
             ) -> NDArray[Axes]:
    if num_plot_row is not None:
        assert num_plot_row > 0, "Number of rows must be greater than 0"
        num_plot_col = ceil(num / num_plot_row)
    if num_plot_col is not None:
        assert num_plot_col > 0, "Number of columns must be greater than 0"
        num_plot_row = ceil(num / num_plot_col)
    if num_plot_row is None and num_plot_col is None:
        num_plot_row = floor(sqrt(num))
        num_plot_col = ceil(num / num_plot_row)
    assert num_plot_row * num_plot_col >= num, "Not enough subplots for the given number of plots."

    if wsize is None:
        wsize = num_plot_col * figsize
    if hsize is None:
        hsize = num_plot_row * figsize

    # If idx3d is specified, we need to create subplots individually with different projections
    if idx3d is not None:
        # Validate idx3d indices
        assert all(0 <= idx < num for idx in idx3d), f"idx3d indices must be within range [0, {num-1}]"

        fig = plot.figure(figsize=(wsize, hsize))
        if not (wspace is not None or hspace is not None):
            fig.set_tight_layout(True)

        axes = []
        for i in range(num):
            # Calculate subplot position (1-indexed for matplotlib)
            subplot_idx = i + 1
            # Determine if this subplot should be 3D
            projection = '3d' if (i in idx3d) else None
            ax = fig.add_subplot(num_plot_row, num_plot_col, subplot_idx, projection=projection)
            axes.append(ax)

        axes = np.array(axes)

        # Apply 3D view settings only to 3D axes
        if elev is not None or azim is not None:
            for i, ax in enumerate(axes):
                if i in idx3d:
                    ax.view_init(elev=elev, azim=azim)
    else:
        # Original logic for uniform projection across all subplots
        fig, axes = plot.subplots(nrows=num_plot_row, ncols=num_plot_col,
                                  figsize=(wsize, hsize),
                                  subplot_kw={'projection': '3d'} if use_3d else None,
                                  constrained_layout=False if wspace is not None or hspace is not None else True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        # Apply 3D view settings to all axes if use_3d is True
        if use_3d and (elev is not None or azim is not None):
            for ax in axes:
                ax.view_init(elev=elev, azim=azim)

    # Apply subplot adjustments if specified
    if wspace is not None and hspace is not None:
        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2, wspace=wspace, hspace=hspace)

    # Apply frame removal to all axes
    if remove_frame:
        lw = 1.5
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(lw)
            ax.spines['bottom'].set_linewidth(lw)
            ax.tick_params(axis='both', direction='in', labelsize=14)

    return axes


def plot_cost_per_exo_params(ax: Axes, cost: pl.DataFrame, cot_name: str):
    XY_LBL = ["K", "Delay"]
    LBL_STANDARD = "K"
    standard_value = cost.filter(pl.col(LBL_STANDARD).abs() < 1e-2)[cot_name].mean()
    min_value = cost[cot_name].min()
    max_value = cost[cot_name].max()
    try:
        norm = TwoSlopeNorm(vmin=min_value, vmax=max_value, vcenter=standard_value)
        label = f'K=0: {standard_value:.1f}, min {min_value:.1f}, max {max_value:.1f}'
        label = label.replace("cot_R_", "")
        plot3d_df(ax, cost, XY_LBL, cot_name, 'surf', label=label, norm=norm, cmap='RdBu_r', disp_z_lbl=False)
    except ValueError as e:
        print(f"[Error] {e}")


def plot3d_df(ax: Axes, df: pl.DataFrame, xy_lbls: List[str], z_lbl: str, shape: str = Union['surf', 'scat'], disp_z_lbl: bool = True,
              **kwargs):
    """
    :param ax:
    :param df:
    :param xy_lbls:
    :param z_lbl:
    :param shape:
    :param kwargs:
        "fontsize": int
        "grid_num": int, if shape is 'surf'
    :return:
    :rtype:
    """
    assert len(xy_lbls) == 2, "xy_lbls must be a list of two labels"
    assert all(lbl in df.columns for lbl in xy_lbls) and z_lbl in df.columns, "Labels not in the dataframe"
    assert shape in ['surf', 'scat'], "Shape must be either 'surf' or 'scat'"
    x_lbl = xy_lbls[0]
    y_lbl = xy_lbls[1]

    if shape == 'scat':
        ax.scatter(df[x_lbl].to_numpy(), df[y_lbl].to_numpy(), df[z_lbl].to_numpy(), **kwargs)
    else:
        x_unique = np.sort(df[x_lbl].unique().to_numpy())
        y_unique = np.sort(df[y_lbl].unique().to_numpy())
        X, Y = np.meshgrid(x_unique, y_unique)
        Z = np.full(X.shape, np.nan)

        for row in df.iter_rows(named=True):
            x_idx = np.where(x_unique == row[x_lbl])[0][0]
            y_idx = np.where(y_unique == row[y_lbl])[0][0]
            val = row[z_lbl]
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            Z[y_idx, x_idx] = val
        ax.plot_surface(X, Y, Z, **kwargs)

        # ax.contour(X, Y, Z, levels=10, cmap='viridis')
        # for i in range(0, num_points_y, 20):
        #     X_const = X[i, 0] * np.ones_like(Y[i, :])
        #     ax.plot(X_const, Y[i, :], Z[i, :], c='y')
        # for i in range(0, num_points_x, 20):
        #     Y_const = Y[0, i] * np.ones_like(X[:, i])
        #     ax.plot(X[:, i], Y_const, Z[:, i], c='g')

    ax.set_xlabel(x_lbl, fontsize=kwargs.get('fontsize', DEFAULT_FONT_SIZE))
    ax.set_ylabel(y_lbl, fontsize=kwargs.get('fontsize', DEFAULT_FONT_SIZE))
    if disp_z_lbl:
        ax.set_zlabel(z_lbl, fontsize=kwargs.get('fontsize', DEFAULT_FONT_SIZE))
    # ax.legend()


def sort_and_plot2d(ax: Axes, x_col: pl.Series, y_col: pl.Series, **kwargs):
    sorted_idx = np.argsort(x_col.to_numpy())
    x_arr = x_col.to_numpy()
    y_arr = y_col.to_numpy()
    ax.plot(x_arr[sorted_idx], y_arr[sorted_idx], **kwargs)


def plot2d_df(ax: Axes, df: pl.DataFrame, x_lbl: str, y_lbl: str, shape: str = 'scat', legend: bool = False, **kwargs):
    """
    Plot a 2D scatter or line plot using the data from the DataFrame.

    :param ax: The Axes object where the plot will be drawn.
    :param df: DataFrame containing the data.
    :param x_lbl: Column label for the x-axis.
    :param y_lbl: Column label for the y-axis.
    :param shape: Type of plot, either 'scat' for scatter or 'line' for line plot.
    :param kwargs: Additional keyword arguments for styling (e.g., fontsize, color, etc.)
    """
    # Check if the provided labels exist in the DataFrame
    assert x_lbl in df.columns and y_lbl in df.columns, "Labels not in the dataframe"
    assert shape in ['scat', 'line', 'bar'], "Shape must be either 'scat' or 'line' or 'bar'"

    # Scatter plot
    if shape == 'scat':
        ax.scatter(df[x_lbl].to_numpy(), df[y_lbl].to_numpy(), **kwargs)

    # Line plot
    elif shape == 'line':
        ax.plot(df[x_lbl].to_numpy(), df[y_lbl].to_numpy(), **kwargs)

    # Bar plot
    elif shape == 'bar':
        ax.bar(df[x_lbl].to_numpy(), df[y_lbl].to_numpy(), **kwargs)

    # Set axis labels
    ax.set_xlabel(x_lbl, fontsize=kwargs.get('fontsize', DEFAULT_FONT_SIZE))
    ax.set_ylabel(y_lbl, fontsize=kwargs.get('fontsize', DEFAULT_FONT_SIZE))

    if legend:
        ax.legend()

def plot_gradient_line(ax: Axes, x: List[float], y: List[float], start_color: str, end_color: str, num_color_sample: int = 20, **kwargs):
    x = np.array(x)
    y = np.array(y)
    seg_len = np.diff(x)
    # seg_len = np.hypot(np.diff(x), np.diff(y))
    cum_len = np.concatenate(([0], np.cumsum(seg_len)))
    total_len = cum_len[-1]

    x_new = np.linspace(x[0], x[-1], num_color_sample * len(x) - 1)
    y_new = np.interp(x_new, x, y)

    points = np.column_stack([x_new, y_new]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    start_rgba = np.array(to_rgba(start_color))
    end_rgba   = np.array(to_rgba(end_color))
    t = np.linspace(0, 1, len(segments))[:, None]
    colors = start_rgba + (end_rgba - start_rgba) * t

    lc = LineCollection(segments, colors=colors, **kwargs)
    ax.add_collection(lc)
    return lc

def lerp_colors(start_color: str, end_color: str, num_color_sample: int = 20):
    start_rgba = np.array(to_rgba(start_color))
    end_rgba   = np.array(to_rgba(end_color))
    t = np.linspace(0, 1, num_color_sample)[:, None]
    colors = start_rgba + (end_rgba - start_rgba) * t
    return colors

def df_row_pretty_print(df: pl.DataFrame, row_idx: int = 0):
    out = ""
    row = df.row(row_idx, named=True)
    for col in df.columns:
        out += f"{col}({row[col]:.2f}) "
    return out.strip()


def compute_param_grid(data: pl.DataFrame, device: torch.device, in_cols: List[str], param_fixed: Dict[str, float] = None, grid_num: int = 10
                       ) -> torch.Tensor:
    """
    Compute a grid of parameters for the given parameter dataframe.
    :param data: The parameter dataframe.
    :param in_cols: Input column names.
    :param config: The configuration dictionary. Each key is a parameter name and the value is the fixed value.
        If the value is None or there is no key, the parameter is varied across the grid.
    :param device: The device for output tensor.
    :param grid_num: The number of grid points for each parameter.
    :return: A list of parameter grids. Each element is a list of parameters.
    """
    param_grid = []
    print(f"Normalized input range")
    for key in in_cols:
        if key not in data.columns:
            raise ValueError(f"{key} not in data columns")
        if key in param_fixed:
            param_grid.append(torch.tensor([param_fixed[key]], device=device))
        else:
            param_grid.append(torch.linspace(data[key].min(), data[key].max(), grid_num, device=device))
    meshgrid = torch.meshgrid(param_grid, indexing='ij')
    return torch.stack(meshgrid, dim=-1).squeeze()

def compute_grid_param_tensor(param: pl.DataFrame, device: torch.device, grid_num: int = 10) -> Dict[str, torch.Tensor]:
    param_ranges = {
        col: torch.linspace(param[col].min(), param[col].max(), grid_num, device=device)
        for col in param.columns
    }
    meshgrid = torch.meshgrid(*param_ranges.values(), indexing='ij')
    return {col: meshgrid[i].flatten() for i, col in enumerate(param_ranges.keys())}


def compute_grid_param_list(param: pl.DataFrame, grid_num: int = 10) -> List[Dict[str, float]]:
    """
    Compute a list of parameters for the given parameter dataframe.
    Parameter is selected within the grid range.
    :param param: The parameter dataframe.
    :param grid_num: The number of grid points for each parameter.
    :return: A list of parameter grids. Each element is a dictionary of parameters.
    """
    param_ranges = {
        col: np.linspace(param[col].min(), param[col].max(), grid_num) for col in param.columns
    }

    keys = list(param_ranges.keys())
    param_combinations = itertools.product(*(param_ranges[key] for key in keys))

    param_list = [dict(zip(keys, combination)) for combination in param_combinations]
    return param_list


def compute_unique_param_list(param: pl.DataFrame) -> List[Dict[str, float]]:
    """
    Compute a list of unique parameters for the given parameter dataframe.
    :param param: The parameter dataframe.
    :return: A list of parameter grids. Each element is a dictionary of parameters.
    """
    unique_param_df = param.unique()
    param_list = unique_param_df.to_dicts()
    return param_list


def param_idx_from_crit(df: pl.DataFrame, crit_field: str, crit_value: List[float], crit_tol: float, return_num: int = 3,
                        selection_from_path: Path = None, selection_from_list: List[int] = None) -> List[Tuple[List[int], float]]:
    """ Return the row indices that satisfy the given criteria.

    :param df:
    :param crit_field:
    :param crit_value:
    :param crit_tol:
    :param return_num:
        If set to 1, raises error if there are more than one satisfying rows.
        else, adjust the criteria tolerance to return more rows. (If crit_tol * 3 and not satisfied, raise error.)

    Selection filters retrieved param_idx. Only one of the two can be set.
        :param selection_from_path: If not None, load the filtering selection file from the given path.
        :param selection_from_list: If not None, use the given list of indices for filtering.
    :return:
        One element per crit_value, param_idx, crit_value = Element
    """

    selected_param_idx = None
    exp_selection_set = False
    list_selection_set = False
    if selection_from_path is not None:
        if selection_from_path.exists():
            print(f"Loading selection from file: {selection_from_path}")
            selected_param_idx = []
            with open(selection_from_path, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    selected_param_idx.append(int(line.replace(',', '').strip()))
            exp_selection_set = True
        else:
            print(f"Selection file not found: {selection_from_path}")
    if selection_from_list is not None and len(selection_from_list) > 0:
        print(f"Using selection from list: {selection_from_list}")
        selected_param_idx = selection_from_list
        list_selection_set = True

    assert not (exp_selection_set and list_selection_set), "Only one of the two selection methods can be set."
    if selected_param_idx is None or len(selected_param_idx) == 0:
        print("Selected param_idx is empty. Using all param_idx.")
    df_selected = df.filter(pl.col("param_idx").is_in(selected_param_idx)) if selected_param_idx is not None else df
    return_buffer = []  # List[List[int]]  # crit_field -> param_idx

    for crit_v in crit_value:
        actual_crit_tol = crit_tol
        mask_df = df_selected.filter((pl.col(crit_field) - crit_v).abs() < crit_tol)
        while len(mask_df) < return_num:
            actual_crit_tol += 0.01
            mask_df = df_selected.filter((pl.col(crit_field) - crit_v).abs() < actual_crit_tol)
            if actual_crit_tol > crit_tol * 3:
                raise ValueError(f"Cannot find a cluster for field {crit_field} with center {crit_v:.2f} with range {actual_crit_tol:.2f}")
        print(f"Found {len(mask_df)} members for field {crit_field} with center {crit_v:.2f} with ±{crit_tol:.2f}")

        if selected_param_idx is not None:
            param_idx_df = mask_df.filter(pl.col("param_idx").is_in(selected_param_idx))
            if len(param_idx_df) == 0:
                print(f"Selected param_idx is empty. Using all param_idx. Selection: {selected_param_idx}")
                param_idx_df = mask_df
        else:
            param_idx_df = mask_df
        if return_num == 1 and len(param_idx_df) > 1:
            print(f"[Warning] More than one(got {len(param_idx_df)}) satisfying rows for field {crit_field} with center "
                  f"{crit_v:.2f} with range {crit_tol:.2f}. Selecting the minimum.")
            min_crit_idx = (param_idx_df[crit_field] - crit_v).abs().arg_min()
            param_idx_df = param_idx_df.slice(min_crit_idx, 1)
        return_buffer.append((param_idx_df["param_idx"].to_list(), actual_crit_tol))

    return return_buffer


def ckpt_from_ckpt_name(ckpt_name_wo_ep: str, epochs: List[int] = None, sample_time_prefix: str = "10", memo: str = "") -> List[str]:
    """ Load the experiments for the given checkpoint name.
    :param ckpt_name_wo_ep: trained checkpoint name without epoch number. e.g. hip.ext10.exo_1017_224925
    :param epochs: If None, load all epochs.
    :param sample_time_prefix:
    :param memo: Additional memo for the checkpoint name. Folder has string like {ckpt_name_wo_ep}_{epoch}+memo_{memo}+_on_{sample_time}
    :return:
    """
    exp_root = PROJECT_ROOT / "experiment_data/simulation"

    exps = []
    epochs = epochs if epochs else []
    if len(epochs) == 0:
        # Use pattern that matches exact epoch number boundaries
        exps = glob(f"{exp_root}/{ckpt_name_wo_ep}_[0-9]*[!0-9]_on_{sample_time_prefix}*")
    else:
        for epoch in epochs:
            # Use pattern that matches exact epoch number boundaries
            ckpt_path = glob(f"{exp_root}/{ckpt_name_wo_ep}_{epoch}[!0-9]*_on_{sample_time_prefix}*")
            if len(ckpt_path) == 0:
                all_ckpt_path = glob(f"{exp_root}/{ckpt_name_wo_ep}*_on_{sample_time_prefix}*")
                raise ValueError(f"No checkpoints found for epoch={epoch} among {all_ckpt_path}. Search string: {exp_root}/{ckpt_name_wo_ep}*_on_{sample_time_prefix}*")
            else:
                exps.extend(ckpt_path)

    # Filter out the memo
    if len(memo) > 0:
        exps = [exp for exp in exps if memo in exp]
    else:
        exps = [exp for exp in exps if "+memo_" not in exp]
    assert len(exps) > 0, f"No experiments found for {ckpt_name_wo_ep}"
    exps = sorted(exps, key=lambda x: int(x.split("/")[-1].split("+")[0].split("_")[-1]))
    print(f"Found {len(exps)} experiments for {ckpt_name_wo_ep}")

    return exps


def filter_df(data: pl.DataFrame, query: Dict[str, float]) -> Tuple[pl.DataFrame, Dict[str, float]]:
    fixed_params = {}
    for key, value in query.items():
        data_param = data[key].unique().to_numpy()
        nearest_value = data_param[np.abs(data_param - value).argmin()]
        fixed_params[key] = float(nearest_value)

    filter_conditions = []
    for f_param in query.keys():
        filter_conditions.append((pl.col(f_param) - fixed_params[f_param]).abs() < 1e-2)

    filtered_data = data
    for condition in filter_conditions:
        filtered_data = filtered_data.filter(condition)

    return filtered_data.clone(), fixed_params


def sorted_legend(ax: Axes, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_handles_labels)
    if 'exclude' in kwargs:
        exclude_labels = kwargs.pop('exclude')
        # Create a filtered list of (label, handle) pairs
        filtered_pairs = [(label, handle) for label, handle in zip(sorted_labels, sorted_handles)
                          if label not in exclude_labels]
        # Unzip the filtered pairs
        if filtered_pairs:
            sorted_labels, sorted_handles = zip(*filtered_pairs)
        else:
            sorted_labels, sorted_handles = [], []
    ax.legend(sorted_handles, sorted_labels, **kwargs)


def remove_first_axis_tick(ax: Axes, direction: str = "xy"):
    if "x" in direction:
        xtickslabels = ax.get_xticklabels()
        xtickslabels[0] = ""
        ax.set_xticklabels(xtickslabels)
    if "y" in direction:
        ytickslabels = ax.get_yticklabels()
        ytickslabels[0] = ""
        ax.set_yticklabels(ytickslabels)


def insert_image(ax: Axes, path: str, label: str = None, label_on_y: bool = True, fontsize: int = 12):
    img = mpimg.imread(PROJECT_ROOT / path)
    img_height, img_width = img.shape[:2]
    # Set the new position for the axes
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.imshow(img)
    ax.set_aspect('auto')
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_xticklabels([])
    if label is not None:
        if label_on_y:
            ax.set_ylabel(label, fontsize=fontsize)
            ax.set_yticks([])
            ax.set_yticklabels([])
        else:
            ax.set_xlabel(label, fontsize=fontsize)
            ax.set_xticks([])
            ax.set_xticklabels([])
    else:
        ax.yaxis.set_visible(False)


def mean_profile_by_cycle(data_list: List[np.array], profile_size: int = 400) -> pl.DataFrame:
    """
    Compute the mean profile by gait cycle for the given data list.
    :param data_list: List of data series.
    :param profile_size: Number of points in the profile.
    :return: DataFrame containing the mean and std profile by gait cycle.
    """
    accumulated_profiles = np.zeros((len(data_list), profile_size))

    for i, data in enumerate(data_list):
        original_indices = np.linspace(0, 1, num=len(data))
        target_indices = np.linspace(0, 1, num=profile_size)
        interpolated_data = np.interp(target_indices, original_indices, data.flatten())
        accumulated_profiles[i] = interpolated_data
    mean_profile = np.mean(accumulated_profiles, axis=0)
    std_profile = np.std(accumulated_profiles, axis=0)
    return pl.DataFrame({'gait_cycle': np.linspace(0, 100, profile_size), 'mean': mean_profile, 'std': std_profile})


def change_axis_to_3d(ax: plt.Axes) -> plt.Axes:
    """
    Change a 2D axis to 3D projection after creation.

    :param ax: The axis to change
    :return: The new 3D axis
    """
    # Get the figure and position from the axis
    fig = ax.figure
    position = ax.get_position()

    # Remove the old axis
    ax.remove()

    # Create a new axis with 3D projection in the same position
    new_ax = fig.add_axes(position, projection='3d')

    return new_ax

def change_axis_to_2d(ax: plt.Axes) -> plt.Axes:
    """
    Change a 3D axis to 2D projection after creation.

    :param ax: The axis to change
    :return: The new 2D axis
    """
    # Get the figure and position from the axis
    fig = ax.figure
    position = ax.get_position()

    # Remove the old axis
    ax.remove()

    # Create a new axis with 2D projection in the same position
    new_ax = fig.add_axes(position)

    return new_ax


def opt_ckpt_versioned_path(ckpt_path: str, latest: bool = True, memo: str = None):
    """
    ckpt_path is exp_name/v[00-99] or exp_name,
        return exp_name/v[00-99]
    If latest is False, return exp_name/v01 else exp_name/v<latest>

    If memo is not empty, return exp_name/v[00-99]_[memo]
    """
    num_nested = ckpt_path.count("/")
    assert num_nested == 1 or num_nested == 0, f"Only exp_name/v01 or exp_name is allowed: {ckpt_path}"
    ckpt_name = ckpt_path.split("/")[0]
    ckpt_base = PROJECT_ROOT / "experiment_data/simulation" / ckpt_name
    assert ckpt_base.exists(), f"ckpt_base does not exist: {ckpt_base}"
    path_with_versions = [p for p in ckpt_base.glob("*_v*") if p.is_dir()]
    # Sort by version number
    path_with_versions.sort(key=lambda x: int(x.name.split('_')[-1].replace('v', '')))
    if latest:
        version = path_with_versions[-1].name
    else:
        version = path_with_versions[0].name
    versioned_path = f"{ckpt_name}/{version}"
    return versioned_path
