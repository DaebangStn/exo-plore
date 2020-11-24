# TODO: performance improvement
# 1. re-allocate the params according to the job progress
# 2. use the zarr array for the data storage
# 3. parallelize the computation and the data storage
from python import muscle
from python.sample.util import *
import polars as pl
from polars import LazyFrame as Lf
import yaml


TS_PATH = 'ray_results/ts/max_checkpoint_mod_scale_1208_223248'
# IN_CSV = 'experiment_data/params/debug.csv'
IN_CSV = 'experiment_data/params/U200_exo.csv'
RECORD_CONFIG = 'data/record/cot.yaml'
# RECORD_CONFIG = 'data/record/gems_angle0.yaml'
PREPROCESS_MA2COT = False
PREPROCESS_MA15COT = True
WRITE_BATCH = 1000
DEFAULT_OUTPUT_ROOT = 'sampled'

INT_COLS = [
    'cycle', 'step',
    # 'contact_R', 'contact_L'
]

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
    return data.filter(pl.col("cycle") >= cycle_from)

def filter_lastrun(data: Lf) -> Lf:
    """
    Drop rows belonging to the last‑recorded cycle—all done with
    a window expression so we stay lazy (no collect()).
    """
    last_cycle_expr = pl.max("cycle")
    return data.filter(pl.col("cycle") < last_cycle_expr)

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

@ray.remote
class PBar:
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Sampling", position=0)
        self._failure_num = 0

    def inc_failure(self):
        self._failure_num += 1
        self.pbar.set_postfix(failure=self._failure_num)
        self.tick()

    def tick(self, n=1):
        self.pbar.update(n)

    def close(self):
        self.pbar.close()


@ray.remote
class FileWriter:
    def __init__(self, filename: str):
        self.filename = filename
        self.df_buffer = []
        directory = osp.dirname(self.filename)
        os.chdir(PROJECT_ROOT)
        if directory and not osp.exists(directory):
            print(f"[Warning] Directory {PROJECT_ROOT/directory} does not exist. Creating...")
            os.makedirs(directory)

    def write(self, df: pl.DataFrame):
        self.df_buffer.append(df)

    def save(self):
        df = pl.concat(self.df_buffer, how='vertical')
        df.write_parquet(self.filename + '.parquet', compression='zstd')


@ray.remote
def remote_rollout(ts_path: str, force_exo: bool, params: List[Dict[str, float]], pbar: PBar, writer: FileWriter, record_config: str, shortening_multiplier: float = None) -> None:
    rollout = muscle.Rollout(ts_path, force_exo)
    rollout.load_config(record_config)
    if shortening_multiplier is not None:
        rollout.set_shortening_multiplier(shortening_multiplier)
    columns = rollout.get_fields()
    futures = []
    dfss = []
    if PREPROCESS_MA2COT or PREPROCESS_MA15COT:
        paramss = []
        
    def queue_data():
        if PREPROCESS_MA2COT or PREPROCESS_MA15COT:
            data = pl.concat(dfss, how='vertical')
            params = pl.concat(paramss, how='vertical')
            for col in params.columns:
                if params[col].dtype == pl.Float64:
                    params = params.with_columns(pl.col(col).cast(pl.Float32))
            data = filter_prerun_w_step(data)
            data = filter_prerun_w_cycle(data)
            data = filter_lastrun(data)
            if PREPROCESS_MA2COT:
                data = compute_cot(data, meta_cols=['meta_ma2'])
            elif PREPROCESS_MA15COT:
                data = compute_cot(data, meta_cols=['meta_ma15'])
            data = params.join(data, on="param_idx", how="inner")
            if PREPROCESS_MA2COT:
                data = data.drop(['meta_ma2'])
            elif PREPROCESS_MA15COT:
                data = data.drop(['meta_ma15'])
            futures.append(writer.write.remote(data))
            paramss.clear()
        else:
            futures.append(writer.write.remote(pl.concat(dfss, how='vertical')))
        dfss.clear()
    
    for param in params:
        records = rollout.run(param)
        if len(records) == 0:
            print(f"Failed on param({param['param_idx']})")
            pbar.inc_failure.remote()
            continue

        dfs = []
        for record in records:
            data_dict = {col: record.data.T[:, idx].astype(np.int32) if col in INT_COLS else record.data.T[:, idx].astype(np.float32)
                         for idx, col in enumerate(columns)}
            dfs.append(pl.DataFrame(data_dict))

        df = pl.concat(dfs, how='vertical')
        # Capitalize the keys
        df.columns = [col.capitalize() if col in ['phase', 'stride', 'k', 'delay', 'mass'] else col for col in df.columns]
        param_idx_ser = pl.Series(name='param_idx', values=[param['param_idx']] * len(df)).cast(pl.Int32)
        df = df.with_columns(param_idx_ser)
        dfss.append(df)
        if PREPROCESS_MA2COT or PREPROCESS_MA15COT:
            paramss.append(pl.DataFrame(param).with_columns(pl.col('param_idx').cast(pl.Int32)))
        if len(dfss) > WRITE_BATCH:
            queue_data()
        pbar.tick.remote(1)

    if len(dfss) > 0:
        queue_data()
    return futures


def sample(input_csv: str, ts_path: str, num_worker: int = None, output_dir: str = None,
           record_config_path: str = RECORD_CONFIG) -> None:
    global PREPROCESS_MA2COT
    global PREPROCESS_MA15COT
    if PREPROCESS_MA2COT:
        if not record_config_path.endswith('ma2.yaml'):
            print(f"PREPROCESS_MA2COT is True, so the record config must be ma2.yaml, but got {record_config_path}")
            PREPROCESS_MA2COT = False
    if PREPROCESS_MA15COT:
        if not record_config_path.endswith('ma15.yaml'):
            print(f"PREPROCESS_MA15COT is True, so the record config must be ma15.yaml, but got {record_config_path}")
            PREPROCESS_MA15COT = False
    
    ray.init(address="auto")
    input_csv = Path(input_csv)
    if input_csv.suffix == '.parquet':
        if not input_csv.exists():
            input_csv = "experiment_data/params" / input_csv
        params_df = pl.read_parquet(input_csv)
    else:
        if not input_csv.exists():
            input_csv = "experiment_data/params" / input_csv
        params_df = pl.read_csv(input_csv)
    params_df.columns = [col.strip() for col in params_df.columns]
    csv_filename = osp.basename(input_csv)
    
    # Overriding the exo parameters when the input csv is the selected parameters
    overiding_k = None
    overiding_delay = None
    if 'selected_k2_d1' in csv_filename:
        overiding_k = 0.2667
        overiding_delay = 0.05
    elif 'selected_k2_d2' in csv_filename:
        overiding_k = 0.2667
        overiding_delay = 0.15
    elif 'selected_k2_d3' in csv_filename:
        overiding_k = 0.2667
        overiding_delay = 0.25
    elif 'selected_k2_d4' in csv_filename:
        overiding_k = 0.2667
        overiding_delay = 0.35
    elif 'selected_k5_d1' in csv_filename:
        overiding_k = 0.5
        overiding_delay = 0.05
    elif 'selected_k5_d2' in csv_filename:
        overiding_k = 0.5
        overiding_delay = 0.15
    elif 'selected_k5_d3' in csv_filename:
        overiding_k = 0.5
        overiding_delay = 0.25
    elif 'selected_k5_d4' in csv_filename:
        overiding_k = 0.5
        overiding_delay = 0.35
    if '_k0' in csv_filename:
        overiding_k = 0.0
    if overiding_k is not None:
        params_df = params_df.with_columns([
            pl.lit(overiding_k).alias('K'),
        ])
        print(f"=========> Overriding the exo parameters for {csv_filename}: K = {overiding_k}")
    if overiding_delay is not None:
        params_df = params_df.with_columns([
            pl.lit(overiding_delay).alias('Delay')
        ])
        print(f"=========> Overriding the exo parameters for {csv_filename}: Delay = {overiding_delay}")
    
    # Extract shortening multiplier from record config
    record_config_yaml = Path(record_config_path)
    shortening_multiplier = None
    if record_config_yaml.exists():
        with open(record_config_yaml, 'r') as f:
            config = yaml.safe_load(f)
            if 'sample' in config and 'shortening_multiplier' in config['sample']:
                shortening_multiplier = config['sample']['shortening_multiplier']
                print(f"=========> Using shortening multiplier from record config: {shortening_multiplier}")
    
    if 'param_idx' not in params_df.columns:
        params_df['param_idx'] = np.arange(len(params_df))
    params_df = params_df.with_columns(pl.col('param_idx').cast(pl.Int32))
    for col in params_df.columns:
        if params_df[col].dtype == pl.Float64:
            params_df = params_df.with_columns(pl.col(col).cast(pl.Float32))
    params = params_df.to_dicts()

    ts_name = osp.basename(ts_path)
    output_dir = output_dir if output_dir else f'{DEFAULT_OUTPUT_ROOT}/{ts_name}_{timestamp()}'
    os.makedirs(output_dir, exist_ok=True)
    if PREPROCESS_MA2COT:
        filename =  f'{output_dir}/data_ma2cot'
    elif PREPROCESS_MA15COT:
        filename =  f'{output_dir}/data_ma15cot'
    else:
        filename =  f'{output_dir}/data'
        params_df.write_parquet(f'{output_dir}/param.parquet', compression='zstd')
        
    if (Path(ts_path) / "metadata.txt").exists():
        shutil.copy(Path(ts_path) / "metadata.txt", f"{output_dir}/metadata.txt")
    else:
        shutil.copy(Path(ts_path) / "metadata.yaml", f"{output_dir}/metadata.yaml")

    with open(record_config_path, 'r') as f:
        record_config = yaml.safe_load(f)
        _args, _, _, values = getargvalues(currentframe())
        for k in _args:
            record_config[k] = values[k]
        record_config['sampler'] = 'ray'
        record_config['git'] = {
            'hash': current_git_hash(8),
            # 'branch': Repo(PROJECT_ROOT).active_branch.name,
        }
        with open(f'{output_dir}/record_config.yaml', 'w') as f:
            yaml.dump(record_config, f, default_flow_style=False)

    progress_actor = PBar.remote(len(params))
    file_writer = FileWriter.remote(filename)

    avail = ray.available_resources()
    n_cpus = avail['CPU'] if num_worker is None else num_worker
    n_gpus = avail.get('GPU', 0)
    if n_gpus == 0:
        raise RuntimeError("No GPU available")

    print(f"Available resources: {n_cpus} CPUs, {n_gpus} GPUs")
    cpu_p_worker = 1
    gpu_p_worker = 0.005
    # if gpu_p_worker * num_worker > n_gpus:
    #     print(f"[Warning] Insufficient GPU resources: {n_gpus} GPUs for {n_cpus} CPUs. Force the ratio 0.001")
    #     gpu_p_worker = 1e-3
    if gpu_p_worker < 1e-3:
        print(f"[Warning] Insufficient GPU resources: {n_gpus} GPUs for {n_cpus} CPUs. Force the ratio 0.001")
        gpu_p_worker = 1e-3
    work_p_cpu = ceil(len(params) / n_cpus)
    ray_refs = []

    sample_config = record_config['sample']
    force_exo = sample_config.get('force_exo', False)
    for i in range(0, len(params), work_p_cpu):
        batch_params = params[i:min(i + work_p_cpu, len(params))]
        ray_refs.append(
            remote_rollout
            .options(num_cpus=cpu_p_worker, num_gpus=gpu_p_worker)
            .remote(ts_path, force_exo, batch_params, progress_actor, file_writer, record_config_path, shortening_multiplier)
        )
    sample_start_time = time.time()
    write_futures = []
    for refs in ray_refs:
        write_futures.extend(ray.get(refs))
    write_start_time = time.time()
    for f in write_futures:
        ray.get(f)
    fut = file_writer.save.remote()
    ray.get(fut)
    # print(f"Writing done in {time.time() - write_start_time:.2f}s")
    print(f"Sampleing done in {time.time() - sample_start_time:.2f}s")
    print(f"Data saved to {osp.dirname(filename)}")
    ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A script to process checkpoints and metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input_csv', help='input CSV file', default=IN_CSV)
    parser.add_argument('-s', '--torchscript_dir', help='torchscript directory path', default=TS_PATH)
    parser.add_argument('-o', '--output_dir', help='output CSV directory', default=None)
    parser.add_argument('-r', '--record_config', help='record configuration for the simulation', type=str,
                        default=RECORD_CONFIG)
    parser.add_argument('-w', '--num_worker', help='number of ray workers', type=int, default=None)
    args = parser.parse_args()
    sample(input_csv=args.input_csv, ts_path=args.torchscript_dir, output_dir=args.output_dir, num_worker=args.num_worker,
           record_config_path=args.record_config)
