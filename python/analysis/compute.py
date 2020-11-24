from python.analysis.slice_interval import compute_bound as slice_interval
from python.analysis.fixed_centerX_minY import compute_bound as fixed_centerX_minY
from python.surrogate.preprocess import load_cot_processed
from python.plot.util import *
import polars as pl

################ CONFIG ################
EXP_NAME = "jaedD_60_2_1001_141120"


def compute_boundary(exp_name: str, method: str = "slice", metabolic_type: str = "cot_jaed", save: bool = True,
                     data_for_boundary: pl.DataFrame = None, **kwargs) -> Union[List[int], List[List[int]]]:
    x_lbl = "velocity"
    # IN_COLS = ["Phase", "Stride"]
    ########################################
    path_prefix = PROJECT_ROOT / "experiment_data/simulation" / exp_name
    if data_for_boundary is None:
        data_for_boundary = load_cot_processed(exp_name, field=[metabolic_type])
    # Collect data once for all operations
    data = data_for_boundary.collect()
    
    if method == "slice":
        param_indices = slice_interval(data[x_lbl].to_numpy(),
                                     data[metabolic_type].to_numpy(),
                                     data['param_idx'].to_numpy(), **kwargs)
    # elif method == "kmeansX_minY":
    #     indices = kmeansX_minY(data_for_boundary[x_lbl].to_numpy(), data_for_boundary[metabolic_type].to_numpy(), **kwargs)
    elif method == "fixed_centerX_minY":
        param_indices = fixed_centerX_minY(data[x_lbl].to_numpy(),
                                         data[metabolic_type].to_numpy(),
                                         param_indices=data['param_idx'].to_numpy(), **kwargs)
    else:
        raise ValueError(f"Invalid method {method}")

    if save:
        path = path_prefix / f"{method}_selection_{metabolic_type}.txt"
        with open(path, "w") as f:
            f.write("# kwargs used for the selection\n")
            for k, v in kwargs.items():
                f.write(f"# {k}: {v}\n")
            f.write("# Selected indices\n")
            for idx in param_indices:
                f.write(f"{idx}\n")
        print(f"Exported boundary indices to {path}")
    return param_indices


def load_param_selection(exp_name: str, method: str, selection_metabolic: str, refresh_selection: bool = False,
                         **kwargs):
    sim_prefix = PROJECT_ROOT / "experiment_data/simulation" / exp_name
    boundary_selection_path = sim_prefix / f"{method}_selection_{selection_metabolic}.txt"
    if refresh_selection or not boundary_selection_path.exists():
        boundary_selection = compute_boundary(exp_name, method=method, metabolic_type=selection_metabolic, **kwargs)
    else:
        with open(boundary_selection_path, "r") as f:
            boundary_selection = [int(line.strip()) for line in f.readlines() if line.strip().isdigit()]
    return boundary_selection


# TODO, you must refactor this method until Nov. if you don't want to mess up the code
def load_param_idx_from_velocity_crit(
        exp_name: str, crit_values: List[float], param_data: pl.DataFrame = None, crit_tol: float = (0.2 / 3.6), selection_method: str = None,
        selection_file: str = None, refresh_boundary: bool = False, minimum_member_num: int = 3, cot_margin: float = None
) -> List[Tuple[List[int], float]]:
    """
    :param exp_name:
    :param param_data:
    :param crit_values:
    :param crit_tol:
    :param selection_method:
    :param selection_file:
    :param refresh_boundary:
    :param minimum_member_num:
    :param cot_margin:
    :return:
    """
    sim_prefix = PROJECT_ROOT / "experiment_data/simulation" / exp_name
    if selection_method is not None:
        boundary_selection_path = sim_prefix / f"{selection_method}_selection.txt"
        if refresh_boundary or not boundary_selection_path.exists() or param_data is not None:
            kwargs = {
                "method": selection_method,
                'save': False,
                'data_for_boundary': param_data,
            }
            if selection_method == "fixed_centerX_minY":
                kwargs["members_per_cluster"] = minimum_member_num
                kwargs["range_init"] = crit_tol
                if cot_margin is not None:
                    kwargs["margin"] = cot_margin

            boundary_selection = compute_boundary(exp_name, **kwargs)
            param_indices = param_idx_from_crit(param_data, "velocity", crit_values, crit_tol, selection_from_list=boundary_selection)
        else:
            param_data = pd.read_csv(sim_prefix / f"metabolic.cot_jaed.csv")
            param_indices = param_idx_from_crit(param_data, "velocity", crit_values, crit_tol, selection_from_path=boundary_selection_path)
    elif selection_file is not None:
        boundary_selection_path = sim_prefix / selection_file
        param_indices = param_idx_from_crit(param_data, "velocity", crit_values, crit_tol, selection_from_path=boundary_selection_path)
    else:  # select by given velocities
        param_indices = param_idx_from_crit(param_data, "velocity", crit_values, crit_tol)
    return param_indices


if __name__ == "__main__":
    compute_boundary(EXP_NAME)
