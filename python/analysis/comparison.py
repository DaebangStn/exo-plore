from typing import List
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import polars as pl
from scipy.interpolate import interp1d

def compare_time_series(x1, y1, x2, y2, name: str = ""):
    """
    Compares two time series and returns a dictionary of metrics.

    Args:
        x1: X-axis data for the first time series (1D numpy array).
        y1: Y-axis data for the first time series (1D numpy array).
        x2: X-axis data for the second time series (1D numpy array).
        y2: Y-axis data for the second time series (1D numpy array).
        name: Name of the data being compared.

    Returns:
        A dictionary containing comparison metrics.
    """
    # Interpolate the second series to the x-axis of the first series
    f = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value='extrapolate')
    y2_interp = f(x1)

    exp = y1
    sim = y2_interp

    # 1) RMSE
    rmse = np.sqrt(np.mean((exp - sim)**2))

    # 2) NRMSE (Normalized RMSE)
    nrmse = rmse / (exp.max() - exp.min()) if (exp.max() - exp.min()) != 0 else 0

    # 3) Pearson Correlation
    corr = np.corrcoef(exp, sim)[0, 1]

    # 4) Peak timing difference
    exp_peak = np.argmax(exp)
    sim_peak = np.argmax(sim)
    peak_timing_error = sim_peak - exp_peak  # sample index difference

    # 5) Dynamic Time Warping distance
    exp_reshaped = exp.reshape(-1, 1)
    sim_reshaped = sim.reshape(-1, 1)
    dtw_dist, path = fastdtw(exp_reshaped, sim_reshaped, dist=euclidean)

    # 6) Combined Normalized DTW
    exp_range = exp.max() - exp.min()
    if exp_range == 0:
        dtw_norm = np.inf  # Avoid division by zero
    else:
        dtw_norm = dtw_dist / (len(path) * exp_range)

    return {
        "name": name,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "Correlation": corr,
        "PeakTimingError_samples": peak_timing_error,
        "DTW_distance": dtw_dist,
        "DTW_distance_norm": dtw_norm,
    }

def print_comparison_table(results: List[dict]):
    """
    Prints a list of comparison results in a table.

    Args:
        results: A list of dictionaries, where each dictionary is the output of compare_time_series.
    
    Metrics Explanation:
    - RMSE (Root Mean Square Error): The square root of the average of the squared differences between the two series.
      It represents the standard deviation of the residuals (prediction errors). Lower is better.
    - NRMSE (Normalized RMSE): RMSE divided by the range of the experimental data. This normalizes the error,
      making it easier to compare across different data scales. Lower is better.
    - Correlation (Pearson): Measures the linear relationship between the two series. A value of 1 means a perfect
      positive linear relationship, -1 a perfect negative linear relationship, and 0 no linear relationship. Higher is better.
    - PeakTimingError_samples: The difference in the sample index of the peak value between the two series.
      A positive value means the simulation peak occurs after the experimental peak.
    - DTW_distance (Dynamic Time Warping): A measure of similarity between two temporal sequences which may vary in time or speed.
      It finds the optimal alignment between the two series. Lower is better.
    - DTW_distance_norm (Normalized DTW): The DTW distance normalized by both the path length and the range of the
      experimental data. This provides a scale-invariant measure of similarity. Lower is better.
    """
    if not results:
        print("No comparison results to display.")
        return
    
    with pl.Config(tbl_rows=-1):
        df = pl.DataFrame(results)
        df = df.sort("name")
        print(df)

def print_latex_table(results: List[dict]):
    """
    Prints the comparison results in a single wide LaTeX table format.
    """
    if not results:
        print("No results to format for LaTeX.")
        return

    df = pl.DataFrame(results)
    
    # Extract velocity and data type from 'name'
    df = df.with_columns([
        pl.col('name').str.extract(r"([a-zA-Z_]+)_v").alias("type"),
        pl.col('name').str.extract(r"_v(\d+\.\d+)").cast(pl.Float64).alias("velocity")
    ])

    metrics = ["NRMSE", "Correlation", "DTW_distance_norm"]
    velocities = sorted(df['velocity'].unique().to_list())
    
    # Pivot the table to a wide format
    pivot_df = df.pivot(index="type", columns="velocity", values=metrics)
    pivot_df = pivot_df.sort("type")

    # --- LaTeX Table Generation ---
    print("--- Combined LaTeX Table ---")
    
    num_vels = len(velocities)
    num_metrics = len(metrics)
    
    # Define column specifications
    col_spec = "l" + "".join(["c" * num_vels] * num_metrics)
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\toprule")
    
    # Header Row 1: Metrics
    header1 = "Type & " + " & ".join([f"\\multicolumn{{{num_vels}}}{{c}}{{{metric.replace('_', ' ')}}}" for metric in metrics]) + r" \\"
    print(header1)
    
    # Header Separator
    cmidrules = " & ".join([f"\\cmidrule(lr){{{2 + i*num_vels}-{1 + (i+1)*num_vels}}}" for i in range(num_metrics)])
    print(cmidrules)

    # Header Row 2: Velocities
    vel_header_part = " & ".join([f"{v:.1f}" for v in velocities])
    header2 = " & " + " & ".join([vel_header_part] * num_metrics) + r" \\"
    print(header2)
    print(r"\midrule")
    
    # Table Body
    # Construct the order of columns as they appear in the pivoted DataFrame
    ordered_cols = [f"{m}_{v}" for m in metrics for v in velocities]
    
    for row in pivot_df.to_dicts():
        row_str = row['type'].replace('_', ' ').title()
        for col_name in ordered_cols:
            row_str += f" & {row[col_name]:.3f}"
        row_str += r" \\"
        print(row_str)
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print("\n")

