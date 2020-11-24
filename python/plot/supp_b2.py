from tbparse import SummaryReader
import matplotlib.pyplot as plt
import matplotlib.ticker
import os
import pandas as pd # Import pandas for DataFrame operations
from python.plot.util import *


def main():
    # Configuration for the plot
    config = {
        "tfevents_path": "experiment_data/tfevents/exo_hei_resist_0807_131302.tfevents",
        "layout": {"rows": 2, "cols": 4},
        "metrics": [
            {"name": "ray/tune/custom_metrics/reward/total", "title": r"$r_{total}$", "y_range": (0, 3), "yticks": [0, 2]},
            {"name": "ray/tune/custom_metrics/reward/meta", "title": r"$r_{energy}$", "y_range": (0, 1), "yticks": [0, 1]},
            {"name": "ray/tune/custom_metrics/reward/dev", "title": r"$r_{HEI}$", "y_range": (-1.0, 0.5), "yticks": [-1.0, 0.0]},
            {"name": "ray/tune/custom_metrics/reward/imitation", "title": r"$r_{arm}$", "y_range": (0, 1), "yticks": [0, 1]},
            {"name": "ray/tune/custom_metrics/reward/gait", "title": r"$r_{gait}$", "y_range": (0, 1), "yticks": [0, 1]},
            {"name": "ray/tune/custom_metrics/reward/loco", "title": r"$r_{step} \cdot r_{vel}$", "y_range": (0, 1), "yticks": [0, 1]},
            {"name": "ray/tune/custom_metrics/reward/head", "title": r"$r_{head}$", "y_range": (0, 1), "yticks": [0, 1]},
            {"name": "ray/tune/custom_metrics/reward/sway", "title": r"$r_{sway}$", "y_range": (0, 1), "yticks": [0, 1]},
        ],
        "x_axis": {"max": 2e8, "ticks": [0, 2e8], "labels": ['0', '200M']},
        "y_axis": {"ticks": 3}
    }

    tfevents_path = config["tfevents_path"]
    metrics_to_plot = config["metrics"]
    layout = config["layout"]

    if not os.path.exists(tfevents_path):
        print(f"Error: The file '{tfevents_path}' was not found.")
        exit()

    try:
        # Initialize SummaryReader from tbparse
        reader = SummaryReader(tfevents_path)
        df_scalars = reader.scalars  # Get all scalar data as a pandas DataFrame

        fig, axs = plt.subplots(layout["rows"], layout["cols"], figsize=(10, 4))
        axs = axs.flatten()  # Flatten the 2D array of axes for easy iteration

        for i, metric_info in enumerate(metrics_to_plot):
            ax = axs[i]
            metric_name = metric_info["name"]
            metric_title = metric_info["title"]

            # Filter the DataFrame for the current metric
            metric_df = df_scalars[df_scalars["tag"] == metric_name].sort_values("step")

            if not metric_df.empty:
                values = metric_df["value"].to_numpy()
                steps = metric_df["step"].to_numpy()

                ax.plot(steps, values)
                ax.set_title(metric_title)

                # Simplify axes for paper
                ax.set_xlim(0, config["x_axis"]["max"])
                ax.set_xticks(config["x_axis"]["ticks"])
                ax.set_xticklabels(config["x_axis"]["labels"])
                ax.set_ylim(metric_info["y_range"])
                ax.set_yticks(metric_info["yticks"])
                ax.tick_params(axis='x', direction='in')
                ax.tick_params(axis='y', direction='in')

            else:
                print(f"Metric '{metric_name}' not found in the tfevents file.")
                ax.set_title(f"{metric_title} (Not Found)")
                ax.text(0.5, 0.5, 'Metric not found', horizontalalignment='center', verticalalignment='center')

        # Hide unused subplots if any
        for i in range(len(metrics_to_plot), len(axs)):
            axs[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"plot/supp_b2.png", dpi=100, transparent=True)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

