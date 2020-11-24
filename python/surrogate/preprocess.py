from python.analysis.slice_interval import compute_bound as slice_interval
from python.surrogate.util import *

# ----------------------------------------------------------------------
# 0. Utility
def _ensure_cols(data: Lf, param: Lf, required: Sequence[str], ctx: str = "") -> None:
    """Lightweight utility to check for required columns"""
    data_columns = data.collect_schema().names()
    param_columns = param.collect_schema().names()
    missing = [c for c in required if c not in data_columns]
    missing = [c for c in missing if c not in param_columns]
    assert not missing, f"{ctx}: Missing columns {missing} / Available columns {data_columns + param_columns}"


# ----------------------------------------------------------------------
# 1. Preprocessing filters

def filter_prerun_w_step(data: Lf, step_until: int = 1500) -> Lf:
    """
    Drop everything before the row index *step_until*.
    (Original logic used a cycle boundary; here we keep things lazy by row index.)
    """
    return (
        data.with_row_index("row_idx")
            .filter(pl.col("row_idx") >= step_until)
            .drop("row_idx")
    )

def filter_prerun_w_cycle(data: Lf, cycle_from: int = 5) -> Lf:
    """Keep rows whose cycle is >= *cycle_from* (lazy)."""
    # debug_data = data.select("cycle").collect().to_numpy()
    # print(np.unique(debug_data))
    
    return data.filter(pl.col("cycle") >= cycle_from)

def filter_lastrun(data: Lf) -> Lf:
    """
    Drop rows belonging to the last‑recorded cycle—all done with
    a window expression so we stay lazy (no collect()).
    """
    last_cycle_expr = pl.max("cycle")
    return data.filter(pl.col("cycle") < last_cycle_expr)


# ----------------------------------------------------------------------
# 2. Parameter‑ratio → absolute
def cvt_ratio_to_absolute(param: Lf) -> Lf:
    """Convert Phase/Stride ratios into absolute cadence / step length."""
    return (
        param.with_columns(
            (pl.col("Phase")  * pl.lit(REF_CADENCE)).alias("param_cadence"),
            (pl.col("Stride") * pl.lit(REF_STEP)).alias("param_step_length"),
        )
    )

# ----------------------------------------------------------------------
# 3. Core computations
def compute_meta(data: Lf,
                 muscle_mass: Dict[str, float],
                 act_cols: List[str]) -> Lf:
    """Derive metabolic columns from activations – lazy expressions only."""
    
    print("===============> [Warning] using jaed formula for the metabolic computation")
    exprs: List[pl.Expr] = [
        ((pl.col(c) ** 3) * (muscle_mass[c.removeprefix("act_")] ** 0.5))
            .alias(f"meta_{c.removeprefix('act_')}")
        for c in sorted(act_cols)
    ]
    return data.with_columns(*exprs).drop(act_cols)

def compute_cot(data: Lf, meta_cols: List[str]) -> Lf:
    """Per‑param_idx Cost‑of‑Transport plus raw meta sums."""
    cot_cols = [m.replace("meta_", "cot_") for m in meta_cols]
    return (
        data.group_by("param_idx")
            .agg(
                *[pl.col(m).sum().alias(m) for m in meta_cols],
                (pl.col("com_z").last()  - pl.col("com_z").first()).alias("whole_dist"),
                (pl.col("time").last()   - pl.col("time").first()).alias("whole_dur"),
            )
            .with_columns(
                (pl.col("whole_dist") / pl.col("whole_dur")).alias("velocity"),
                *[(pl.col(m) / pl.col("whole_dist")).alias(cot)
                  for m, cot in zip(meta_cols, cot_cols)]
            )
            .select(["param_idx", "velocity", *meta_cols, *cot_cols])
    )

def compute_cadence_and_step(data: Lf) -> Lf:
    """Cadence (steps/s) and step length per param_idx."""
    # Preserve velocity if it exists in the input data
    cols_to_select = ["param_idx", "cadence", "step_length", "velocity"]
    return (
        data.group_by("param_idx")
            .agg(
                pl.col("cycle").n_unique().alias("n_cycle"),
                (pl.col("time").last() - pl.col("time").first()).alias("elapsed"),
                (pl.col("foot_R").last() - pl.col("foot_R").first()).alias("stride_sum"),
                pl.col("velocity").first().alias("velocity"),
            )
            .with_columns(
                (2 * pl.col("n_cycle") / pl.col("elapsed")).alias("cadence"),
                (pl.col("stride_sum") / (2 * pl.col("n_cycle"))).alias("step_length"),
            )
            .select(cols_to_select)
    )

def cycle_length_integrity(data: Lf, vol: float = 0.2) -> Lf:
    """Remove cycles whose length deviates more than ±vol from the mean length."""
    len_by_cycle = data.group_by(["param_idx", "cycle"]).len()
    mean_len = len_by_cycle.group_by("param_idx").agg(
        pl.mean("len").alias("mean_len")
    )
    keep_mask = (
        len_by_cycle.join(mean_len, on="param_idx")
            .with_columns(
                ((pl.col("len") > pl.col("mean_len") * (1 - vol)) &
                 (pl.col("len") < pl.col("mean_len") * (1 + vol))).alias("keep")
            )
            .filter(pl.col("keep"))
            .select(["param_idx", "cycle"])
    )
    return data.join(keep_mask, on=["param_idx", "cycle"], how="semi")

# ----------------------------------------------------------------------
# 4. High‑level pipeline helpers
def mean_per_cycle(data: Lf, param: Lf,
                   input_col: List[str], target_col: List[str]) -> Lf:
    """Per‑subject mean across cycles (fully lazy)."""
    _ensure_cols(data, param, input_col + target_col + ["param_idx", "cycle"], "mean_per_cycle")

    data = filter_prerun_w_cycle(data)
    data = filter_lastrun(data)
    per_param_mean = (
        data.drop("cycle")
            .group_by("param_idx")
            .mean()
    )
    return (
        param.select("param_idx")
             .join(per_param_mean, on="param_idx", how="inner")
             .select(input_col + target_col)
    )

# ---------- CoT from activation ---------------------------------------
def process_act(data: Lf, param: Lf,
                input_col: List[str], out_cols: List[str],
                prerun_filter: int = 1500) -> Lf:
    """Compute CoT using activation signals—no collect inside."""
    act_cols = [c for c in data.collect_schema().names() if c.startswith("act_")]
    _ensure_cols(data, param, input_col + ["param_idx", "cycle", "com_z", "time"],
                 "process_act")

    with open("data/muscle_mass.json") as f:
        muscle_mass = json.load(f)

    data = filter_prerun_w_step(data, prerun_filter)
    data = filter_lastrun(data)
    data = cycle_length_integrity(data)
    data = compute_meta(data, muscle_mass=muscle_mass, act_cols=act_cols)
    data = compute_cot(data, meta_cols=[c.replace("act_", "meta_") for c in act_cols])
    return param.join(data, on="param_idx", how="inner").select(input_col + out_cols)

# ---------- CoT from pre‑computed metabolic fields ---------------------
def process_meta(data: Lf, param: Lf,
                 input_col: List[str], out_cols: List[str],
                 prerun_filter: int = 1500) -> Lf:
    """Compute CoT from existing meta_* or cot_* columns, lazily."""
    meta_cols = [c.replace("cot_", "meta_") if c.startswith("cot_") else c
                 for c in out_cols if c.startswith(("meta_", "cot_"))]
    
    check_cols = input_col + meta_cols + ["param_idx", "cycle", "com_z", "time"]
    check_cols = [col for col in check_cols if col != "velocity"]
    _ensure_cols(data, param, check_cols, "process_meta")

    data = filter_prerun_w_step(data, prerun_filter)
    data = filter_prerun_w_cycle(data)
    data = filter_lastrun(data)
    data = cycle_length_integrity(data)
    data = compute_cot(data, meta_cols=meta_cols)
    param_cols = param.collect_schema().names()
    select_cols = ["param_idx"] + input_col + out_cols
    if "Delay" in param_cols and "Delay" not in select_cols:
        select_cols.append("Delay")
    return param.join(data, on="param_idx", how="inner").select(select_cols)


def process_meta_select(data: Lf, param: Lf,
                        input_col: List[str], out_cols: List[str],
                        selection_x: str, selection_y: str,
                        selection_num: int, selection_margin: float,
                        prerun_filter: int = 1500) -> Lf:
    """
    Same as `process_meta`, but uses partial collect for minimal memory:
    - Collects only ["param_idx", sel_x, sel_y] to minimize memory usage
    - Applies `slice_interval` to select subset
    - Returns full LazyFrame filtered by selected param_idx
    """
    out_cols = list(set(out_cols) | {selection_x, selection_y})
    # Step 1: Run the normal processing pipeline
    out = process_meta(data, param, input_col, out_cols, prerun_filter)
    
    # Step 2: Select only the minimal columns needed for selection
    sel_df = out.select(["param_idx", selection_x, selection_y]).collect()

    # Step 3: Run the numpy-based selection logic
    selected_param_indices = slice_interval(
        sel_df[selection_x].to_numpy(),
        sel_df[selection_y].to_numpy(),
        sel_df["param_idx"].to_numpy(),
        num_interval=selection_num,
        margin=selection_margin,
    )

    print(
        f"[DATA] from {len(sel_df)}, selected {len(selected_param_indices)} with "
        f"slice_interval({selection_x}, {selection_y}, num={selection_num}, margin={selection_margin})"
    )

    # Step 4: Filter the original LazyFrame by selected param_idx
    return out.filter(pl.col("param_idx").is_in(selected_param_indices))


def process_cadence_and_phase(data: Lf, param: Lf,
                              input_col: List[str], out_cols: List[str],
                              prerun_filter: int = 1500) -> Lf:
    """Cadence + step‑length shifts versus reference parameters."""
    # Ensure all required columns are present in the raw input data
    _ensure_cols(data, param, ["param_idx", "cycle", "foot_R", "time", "com_z"], "cadence&phase")

    # Step-by-step preprocessing
    processed_data = filter_prerun_w_step(data, prerun_filter)
    processed_data = filter_prerun_w_cycle(processed_data)
    processed_data = filter_lastrun(processed_data)
    processed_data = cycle_length_integrity(processed_data)
    processed_data = processed_data.group_by("param_idx").agg(
        (pl.col("com_z").last()  - pl.col("com_z").first()).alias("whole_dist"),
        (pl.col("time").last()   - pl.col("time").first()).alias("whole_dur"),
    ).with_columns(
        (pl.col("whole_dist") / pl.col("whole_dur")).alias("velocity"),
    ).select(["param_idx", "velocity"]).join(processed_data, on="param_idx", how="inner")

    measured = compute_cadence_and_step(processed_data)

    # Param to absolute values
    param_abs = cvt_ratio_to_absolute(param)

    # Join measured and param_abs to compute shifts
    return (
        param_abs
        .join(measured, on="param_idx", how="inner")
        .with_columns(
            ((pl.col("param_cadence") - pl.col("cadence")) / pl.col("param_cadence"))
            .alias("cadence_shift"),
            ((pl.col("param_step_length") - pl.col("step_length")) / pl.col("param_step_length"))
            .alias("step_length_shift"),
        )
        .select(
            pl.col('param_idx'),
            pl.col("velocity"),
            pl.col("cadence"),
            pl.col("step_length"),
            pl.col("cadence_shift"),
            pl.col("step_length_shift")
        )
    )


def process_angle_amplitude(data: Lf, param: Lf,
                            input_col: List[str], out_cols: List[str],
                            prerun_filter: int = 1500) -> Lf:
    """Min / max / amplitude per angle_* column (lazy)."""
    data_columns = data.collect_schema().names()
    angle_cols = [c for c in data_columns if c.startswith("angle_")]
    meta_cols  = [c for c in data_columns if c.startswith("meta_")]
    _ensure_cols(data, param, angle_cols + ["param_idx", "cycle", "time"], "angle_amplitude")

    data = filter_prerun_w_step(data, prerun_filter)
    data = filter_lastrun(data)
    data = cycle_length_integrity(data)

    grp = data.group_by("param_idx")

    angle_stats = grp.agg(
        *[pl.min(c).alias(f"{c}_min") for c in angle_cols],
        *[pl.max(c).alias(f"{c}_max") for c in angle_cols],
    ).with_columns(
        *[(pl.col(f"{c}_max") - pl.col(f"{c}_min")).alias(f"{c}_amplitude")
          for c in angle_cols]
    )

    cot_tbl = compute_cot(grp, meta_cols=meta_cols)

    return (
        param.join(angle_stats.join(cot_tbl, on="param_idx"), on="param_idx", how="inner")
             .select(input_col + out_cols)
    )


def mean_negpos_field(data: Lf, param: Lf, field: str,
                      prerun_filter: int = 1500) -> Lf:
    """Positive‑only and negative‑only means of *field*."""
    _ensure_cols(data, param, [field], "mean_negpos_field")

    data = filter_prerun_w_step(data, prerun_filter)
    data = filter_lastrun(data)
    data = cycle_length_integrity(data)

    stats = data.group_by("param_idx").agg(
        pl.col(field).filter(pl.col(field) > 0).mean().alias(f"{field}_pos_mean"),
        pl.col(field).filter(pl.col(field) < 0).mean().alias(f"{field}_neg_mean"),
    )
    return param.join(stats, on="param_idx", how="inner").rename({
        f"{field}_pos_mean": "power_pos",
        f"{field}_neg_mean": "power_neg",
    })


def rms_field(data: Lf, param: Lf, field: str,
              prerun_filter: int = 1500) -> Lf:
    """Root‑mean‑square of *field*."""
    _ensure_cols(data, param, [field], "rms_field")

    data = filter_prerun_w_step(data, prerun_filter)
    data = filter_lastrun(data)
    data = cycle_length_integrity(data)

    stats = data.group_by("param_idx").agg(
        ((pl.col(field) ** 2).mean().sqrt()).alias(f"{field}_rms")
    )
    return param.join(stats, on="param_idx", how="inner").rename({
        f"{field}_rms": "moment",
    })


def gt_path_from_exp_name(exp_name: str, in_cols: List[str], out_cols: List[str], processor: Union[str, dict] = None, is_ckpt: bool = False
                          ) -> Optional[Path]:
    exp_prefix = PROJECT_ROOT / "experiment_data/simulation" / exp_name
    exp_prefix = exp_prefix.parent if is_ckpt else exp_prefix
    assert exp_prefix.is_dir(), f"Experiment {exp_name} does not exist"

    if isinstance(processor, dict):
        processor = processor["func"]

    matched_files = []
    if processor == "activation":
        gt_candid = exp_prefix / "activation.parquet"
        if gt_candid.exists():
            matched_files.append(gt_candid)
    else:
        gt_candid = list(exp_prefix.glob("*.gt.csv")) + list(exp_prefix.glob("*.gt.parquet"))
        required_cols_set = set(in_cols + out_cols)
        # Iterate over each ground truth file to find a match
        for gt_file in gt_candid:
            # Extract the base filename without the '_gt.csv' suffix
            base_name = gt_file.stem  # e.g., "Phase+Stride+K+Delay+cot_jaed"

            if base_name.endswith(".gt"):
                base_name = base_name.split(".")[0]

            # Split the base name by '+' to get individual column names
            file_cols = set(base_name.split('+'))

            # Check if all required columns are present in the file
            if required_cols_set.issubset(file_cols):
                # if isinstance(processor, str) and processor.endswith("selected"):
                    # if "selected" in gt_file.stem:
                        # matched_files.append(gt_file)
                # else:
                    # matched_files.append(gt_file)
                matched_files.append(gt_file)
    if len(matched_files) == 0:
        return None
    elif len(matched_files) > 1:
        matched_filenames = ', '.join([file.name for file in matched_files])
        print(
            f"[Warning] Multiple ground truth files found containing columns {', '.join(required_cols_set)}: {matched_filenames}."
        )
    return matched_files[0]


def gt_stat_from_exp_name(exp_name: str, in_cols: List[str], out_cols: List[str], processor: Union[str, dict] = None, is_ckpt: bool = True
                         ) -> Optional[Df]:
    exp_prefix = PROJECT_ROOT / "experiment_data/simulation" / exp_name
    exp_prefix = exp_prefix.parent if is_ckpt else exp_prefix
    gt_path = gt_path_from_exp_name(exp_name, in_cols, out_cols, processor, is_ckpt)
    
    if isinstance(processor, dict):
        processor_func_name = processor["func"]
    else:
        processor_func_name = processor
    
    # Handle the cases based on the number of matched files
    # if gt_path and not processor_func_name.endswith("selected"):
    if gt_path:
        if processor == "activation":
            stat = None
        else:
            gt_stat_path = Path(str(gt_path).replace(".gt.parquet", ".stat.csv"))
            stat = pl.read_csv(gt_stat_path, has_header=True)
    else:
        if processor is None:
            raise ValueError(f"No ground truth file found in '{exp_prefix}' containing columns: {', '.join(required_cols_set)}. You must provide a processor.")
        _, stat= preprocess(exp_name, processor, in_cols, out_cols)
    return stat


def load_data(exp_name: str) -> Tuple[Lf, Df]:
    exp_prefix = PROJECT_ROOT / "experiment_data/simulation" / exp_name
    return read_lf(exp_prefix / "data.parquet"), read_df(exp_prefix / "param.parquet")


def gt_from_exp_name(exp_name: str, in_cols: List[str], out_cols: List[str], processor: Union[str, dict] = None, is_ckpt: bool = False
                     ) -> Tuple[Lf, Df]:
    """ Load the ground truth or process data to create ground truth data
    :param exp_name:
    :param in_cols:
    :param out_cols:
    :param processor:
    :param is_ckpt: If the experiment is a checkpoint naming format (e.g. gaitnet_exo_unif_1002_231657/v01)
    :return:
        1. The ground truth data
        2. The ground truth statistics
    """
    exp_prefix = PROJECT_ROOT / "experiment_data/simulation" / exp_name
    exp_prefix = exp_prefix.parent if is_ckpt else exp_prefix
    gt_path = gt_path_from_exp_name(exp_name, in_cols, out_cols, processor, is_ckpt)
    
    if isinstance(processor, dict):
        processor_func_name = processor["func"]
    else:
        processor_func_name = processor
    
    # Handle the cases based on the number of matched files
    # print(f"[DEBUG] re-compuring the ground truth data for {exp_name}")
    if gt_path:
    # if False:
        print(f"[Data] Preprocessed data found at {gt_path}")
        gt = read_lf(gt_path)
        gt_stat_path = Path(str(gt_path).replace(".gt.parquet", ".stat.csv"))
        stat = pl.read_csv(gt_stat_path, has_header=True)
    else:
        print(f"[Data] Preprocessing the ground truth data for {exp_name}")
        if processor is None:
            raise ValueError(f"No ground truth file found in '{exp_prefix}' containing columns: {', '.join(required_cols_set)}. You must provide a processor.")
        gt, stat= preprocess(exp_name, processor, in_cols, out_cols)
    return gt, stat


def save_lazyframe_in_batches(lf: pl.LazyFrame, saved_path: str, batch_size: int = 60000):
    # Get the total number of rows in the LazyFrame
    total_rows = lf.select(pl.count()).collect()[0, 0]
    
    if total_rows <= batch_size:
        lf.collect().write_parquet(saved_path)
        return
    
    # Temporary folder to store individual batch parquet files
    tmp_dir = saved_path + "_tmp_parts"
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=False)

    print(f"[INFO] Total rows: {total_rows}, Batch size: {batch_size}")
    part_paths = []

    # Iterate through the LazyFrame in batches and save each as a separate Parquet file
    for i in tqdm(range(0, total_rows, batch_size), desc="Saving parquet in batches"):
        batch_df = lf.slice(i, batch_size).collect()
        batch_path = os.path.join(tmp_dir, f"part_{i//batch_size:04d}.parquet")
        batch_df.write_parquet(batch_path)
        part_paths.append(batch_path)

    # Optional: merge all batch files into one final parquet file
    print(f"[INFO] Merging {len(part_paths)} parts into {saved_path}")
    df_all = pl.concat([pl.read_parquet(p) for p in part_paths])
    df_all.write_parquet(saved_path)

    # Cleanup: remove temporary batch files and folder
    for p in part_paths:
        os.remove(p)
    os.rmdir(tmp_dir)


def preprocess(exp_name: str, processor: Union[str, dict], in_cols: List[str], out_cols: List[str]) -> Tuple[Lf, Optional[Df]]:
    if isinstance(processor, dict):
        processor_func_name = processor["func"]
        processor_params = processor["params"]
    else:
        processor_func_name = processor
        processor_params = {}
    exp_dir = PROJECT_ROOT / "experiment_data/simulation" / exp_name
    if processor_func_name == "metabolic":
        if (exp_dir / "data_ma2cot.parquet").exists():
            # raise ValueError(f"[DEBUG] do not use ma2cot data for this period")
            print(f"[WARNING] use ma2cot data for {exp_name}")
            data, stat = process_data_ma2cot(exp_dir)
            data = data.select(in_cols + out_cols)
            return data, stat
        if (exp_dir / "data_ma15cot.parquet").exists():
            print(f"[WARNING] use ma15cot data for {exp_name}")
            data, stat = process_data_ma15cot(exp_dir)
            data = data.select(in_cols + out_cols)
            return data, stat
    
    data, param = data_and_param_path(exp_name, read_file=True)
    processor_func = processor_map[processor_func_name]
    processed = processor_func(data, param, in_cols, out_cols, **processor_params, prerun_filter=1500)
    exp_prefix = PROJECT_ROOT / "experiment_data/simulation" / exp_name
    if processor == "activation":
        saved_path = exp_prefix / "activation.parquet"
        stat = None
        print("[DATA] Activation data has no statistics.")
    else:
        col_tot = sorted(set(in_cols + out_cols))
        if processor_func_name.endswith("selected") and False:
            files = list(exp_prefix.glob("*.selected.gt.parquet"))
            if len(files) > 0:
                print(f"[DATA] Removing all *.selected.gt.parquet files in {exp_prefix}")
                for f in files:
                    print(f"[DATA] Removing {f}")
                    os.remove(f)
            saved_path = exp_prefix / f"{'+'.join(col_tot)}.selected.gt.parquet"
        else:
            saved_path = exp_prefix / f"{'+'.join(col_tot)}.gt.parquet"
        processed, stat = normalize(processed, in_cols + out_cols, saved_path)
    processed.collect().write_parquet(saved_path)
    # save_lazyframe_in_batches(processed, saved_path)
    print(f"[DATA] Preprocessed data saved at {saved_path}")
    return processed, stat


def process_data_ma2cot(exp_dir: Path) -> Tuple[Lf, Optional[Df]]:
    if not exp_dir.is_dir():
        exp_dir = exp_dir.parent
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    data = pl.scan_parquet(exp_dir / "data_ma2cot.parquet")
    
    # how to see the mass.unique
    # print(data.select("Phase").unique().collect())
    # return
    
    # data = data.filter(pl.col("cot_ma2") <= 140)
    # data = data.filter(pl.col("K") <= 0.5)
    # data = data.filter((pl.col("Mass") > 1.1) & (pl.col("Mass") < 1.3))
    # data = data.filter((pl.col("Mass") > 0.9) & (pl.col("Mass") < 1.1))
    # data = data.filter(pl.col("Phase") > 1.0)
    # data = data.filter(pl.col("Stride") > 1.0)
    columns = list(sorted(data.collect_schema().names()))
    saved_path = exp_dir / f"{'+'.join(columns)}.gt.parquet"
    data, stat = normalize(data, columns, saved_path)
    data.sink_parquet(saved_path)
    return data, stat


def process_data_ma15cot(exp_dir: Path) -> Tuple[Lf, Optional[Df]]:
    if not exp_dir.is_dir():
        exp_dir = exp_dir.parent
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    data = pl.scan_parquet(exp_dir / "data_ma15cot.parquet")
    
    # how to see the mass.unique
    # print(data.select("Phase").unique().collect())
    # return
    
    # data = data.filter(pl.col("cot_ma2") <= 140)
    # data = data.filter(pl.col("K") <= 0.5)
    # data = data.filter((pl.col("Mass") > 1.1) & (pl.col("Mass") < 1.3))
    # data = data.filter((pl.col("Mass") > 0.9) & (pl.col("Mass") < 1.1))
    # data = data.filter(pl.col("Phase") > 1.0)
    # data = data.filter(pl.col("Stride") > 1.0)
    columns = list(sorted(data.collect_schema().names()))
    saved_path = exp_dir / f"{'+'.join(columns)}.gt.parquet"
    data, stat = normalize(data, columns, saved_path)
    data.sink_parquet(saved_path)
    return data, stat

def load_cot_processed(exp_name: str, use_gt: bool = False, field: List[str] = None) -> Optional[pl.DataFrame]:
    exp_dir = PROJECT_ROOT / "experiment_data/simulation" / exp_name
    if (exp_dir / "data_ma2cot.parquet").exists():
        print(f"[INFO] Use ma2cot data for {exp_name}")
        data, _ = process_data_ma2cot(exp_dir)
        return data
    if (exp_dir / "data_ma15cot.parquet").exists():
        print(f"[INFO] Use ma15cot data for {exp_name}")
        data, _ = process_data_ma15cot(exp_dir)
        return data
    
    try:
        if field is None:
            field = ["cot_ma2"]
        field = sorted(list(set(field)))

        sim_prefix = PROJECT_ROOT / "experiment_data/simulation" / exp_name
        if not sim_prefix.is_dir():
            sim_prefix = sim_prefix.parent
            if not sim_prefix.is_dir():
                raise FileNotFoundError(f"Experiment directory not found: {sim_prefix}")

        if use_gt:
            gt_data, gt_stat = gt_from_exp_name(exp_name, ["Phase", "Stride"], field, "metabolic")
            denormalize([gt_data], gt_stat, operand_lbl=["cot_jaed", "Phase", "Stride"])
            return gt_data

        if len(field) == 1:
            metabolic_path = sim_prefix / f"metabolic.{field[0]}.csv"
        else:
            metabolic_path = sim_prefix / f"metabolic.{'+'.join(field)}.csv"
        # if metabolic_path.exists():
            # print(f"Loading cot data from {metabolic_path}")
            # cot_data = pl.read_csv(metabolic_path)
            # return cot_data

        sim_data, sim_param = data_and_param_path(exp_name, read_file=True)
        processed = processor_map["metabolic"](sim_data, sim_param, ["velocity", "Phase", "Stride"], field, prerun_filter=1500)
        processed.collect().write_csv(metabolic_path)
        return processed
    except Exception as e:
        print(f"[ERROR] {e}")
        raise e


def load_shift_processed(exp_name: str) -> Optional[pl.DataFrame]:
    try:
        sim_prefix = PROJECT_ROOT / "experiment_data/simulation" / exp_name
        if not sim_prefix.is_dir():
            sim_prefix = sim_prefix.parent
            if not sim_prefix.is_dir():
                raise FileNotFoundError(f"Experiment directory not found: {sim_prefix}")
        shift_path = sim_prefix / "cadence_and_phase.process.csv"
        if shift_path.exists():
            shift_data = pl.read_csv(shift_path)
            return shift_data
        shift_path = sim_prefix / "cadence_and_phase.process.parquet"
        if shift_path.exists() and False:
            shift_data = pl.read_parquet(shift_path)
            return shift_data

        sim_data, sim_param = data_and_param_path(exp_name, read_file=True)
        processed = processor_map["cadence_and_phase"](sim_data, sim_param, ["velocity"], [], prerun_filter=1500)
        processed_df = processed.collect()
        processed_df.write_parquet(shift_path)
        return processed_df
    except Exception as e:
        print(f"[ERROR] {e}")
        raise e


def load_angle_amplitude_processed(exp_name: str) -> Optional[pl.DataFrame]:
    """
    Load or compute angle amplitude data for a given experiment.
    Processes max/min angles and their amplitude, saving results to a file.
    
    Args:
        exp_name (str): Name of the experiment
        
    Returns:
        pl.DataFrame: Processed angle amplitude data or None if processing fails
    """
    try:
        sim_prefix = PROJECT_ROOT / "experiment_data/simulation" / exp_name
        if not sim_prefix.is_dir():
            sim_prefix = sim_prefix.parent
            if not sim_prefix.is_dir():
                raise FileNotFoundError(f"Experiment directory not found: {sim_prefix}")
        
        # Check for existing processed files
        angle_path = sim_prefix / "angle_amplitude.process.parquet"
        if angle_path.exists():
            angle_data = pl.read_parquet(angle_path)
            return angle_data
            
        # Process new data if no cached file exists
        sim_data, sim_param = data_and_param_path(exp_name, read_file=True)
        processed = processor_map["angle_amplitude"](sim_data, sim_param, [], [], prerun_filter=1500)
        processed.write_parquet(angle_path)
        return processed
        
    except Exception as e:
        print(f"[ERROR] {e}")
        raise e


processor_map: Dict[str, Any] = {
    "mean_per_cycle": mean_per_cycle,
    "metabolic": process_meta,
    "metabolic_selected": process_meta_select,
    "activation": process_act,
    "cadence_and_phase": process_cadence_and_phase,
    "angle_amplitude": process_angle_amplitude,
}
