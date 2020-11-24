from python.cvt_ckpt_to_ts import cvt_ckpt_to_ts
from python.sample.export_ray import sample
from python.util import *


CKPT = "experiment_data/checkpoints/no_exo_meta_ma_0730_161202"
IN_CSV = 'experiment_data/params/b20_U500_k0.csv'
RECORD_CONFIG = 'data/record/no_mod_state_ma.yaml'
MEMO = "gait"
EPOCH = 10000

TS_DIR = "ray_results/ts"
SUSTAIN_TS = True

def export_ckpt(input_csv: str, checkpoint: str, record_config: str, output_dir: str = None, sustain: bool = True, memo: str = None, num_worker: int = 16, epoch: int = None):
    ckpt_path = Path(PROJECT_ROOT) / checkpoint
    ts_path = Path(PROJECT_ROOT) / TS_DIR / ckpt_path.name

    if output_dir is None:
        sample_dirname = Path(checkpoint).name
        if memo is not None:
            sample_dirname += f"+memo_{memo}"
        output_dir = str(Path(PROJECT_ROOT) / "sampled" / sample_dirname)

    if not osp.exists(ts_path) or not sustain:
        shutil.rmtree(ts_path, ignore_errors=True)
        os.makedirs(ts_path, exist_ok=True)
        print(f"Creating checkpoint {ts_path}")
        ts_path = str(cvt_ckpt_to_ts(ckpt_path, str(Path(PROJECT_ROOT) / TS_DIR), epoch=epoch))
    ts_path = str(ts_path)
    print("Sampling...")
    sample(input_csv=input_csv, ts_path=ts_path, output_dir=output_dir, record_config_path=record_config, num_worker=num_worker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to sample from checkpoints.")
    parser.add_argument('-i', '--input_csv', help='input CSV file', default=IN_CSV)
    parser.add_argument('-c', '--checkpoint', help='checkpoint path', default=CKPT, type=str)
    parser.add_argument('-r', '--record_config', help='record configuration for the simulation', type=str,
                        default=RECORD_CONFIG)
    parser.add_argument('-s', '--sustain', help='sustain torchscript', action='store_true', default=SUSTAIN_TS)
    parser.add_argument('-m', '--memo', help='memo', type=str, default=MEMO)
    parser.add_argument('-e', '--epoch', help='checkpoint epoch', type=int, default=EPOCH)
    args = parser.parse_args()
    export_ckpt(args.input_csv, args.checkpoint, args.record_config, sustain=args.sustain, memo=args.memo, epoch=args.epoch)
