from python.local.export_ckpt import export_ckpt
from python.util import *


ROOT_DIR = "ray_results"
NUM_WORKER = 16
# Iterate ckpt
CKPTs = [
'no_exo_meta_ma125_10000_0805_020258',
]
IN_CSV = 'experiment_data/params/b20_U500_k0.csv'
RECORD_CONFIG = 'data/record/no_mod_state.yaml'

MEMO = ""

TS_DIR = "ray_results/ts"
SUSTAIN_TS = True
COT_TYPES = [
    "a", "a2", "a3",
    "m05a", "m05a2", "m05a3",
    "m2a", "m2a2", "m2a3",
    "ma", "ma2", "ma3",
    'a15', 'm05a15', 'm2a15', 'ma15',
    'a125', 'm05a125', 'ma125', 'm2a125',
]

def main():
    for ckpt in CKPTs:
        ckpt_path = os.path.join(ROOT_DIR, ckpt)
        cot_type = ckpt.split("_")[1]
        if cot_type not in COT_TYPES:
            raise ValueError(f"Invalid cot type: {cot_type} for logdir {ckpt_path}")
        record_config = Path(RECORD_CONFIG)
        record_config = record_config.parent / f"{record_config.stem}_{cot_type}{record_config.suffix}"
        print(f"=============> Using record config: {record_config}")
        memo = Path(IN_CSV).stem + "_" + record_config.stem + "+_on_" + timestamp()
        if len(MEMO) > 0:
            memo = MEMO + "_" + memo
        export_ckpt(IN_CSV, ckpt_path, record_config.as_posix(), sustain=SUSTAIN_TS, memo=memo, num_worker=NUM_WORKER)

if __name__ == "__main__":
    main()
