from python.surrogate.util import PROJECT_ROOT
from python.surrogate.util import *
from python.surrogate.nn.train import launch_experiment


NUM_WORKERS = 1

# Hardcoded list of experiments to train
EXP_NAMES = [
    "dummy_test",
    # "exo_hei_resist_21000_0807_131406+memo_b21df_LHS5000_no_mod_state_ma15_gems+_on_0902_122806",
]


def main():
    total_exp_num = len(EXP_NAMES)
    failed_exp = []
    
    if NUM_WORKERS > 1:
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_to_exp = {
                executor.submit(launch_experiment, EXP_NAMES[exp_idx]): exp_idx
                for exp_idx in range(total_exp_num)
            }
            for future in as_completed(future_to_exp):
                try:
                    future.result()
                except Exception as e:
                    exp_idx = future_to_exp[future]
                    print(f"Experiment {exp_idx} ({EXP_NAMES[exp_idx]}) failed: {e}")
                    failed_exp.append(EXP_NAMES[exp_idx])
    else:
        for exp_idx in range(total_exp_num):
            launch_experiment(EXP_NAMES[exp_idx])

    if len(failed_exp) > 0:
        print(f"Failed experiments:")
        for exp_name in failed_exp:
            print(f"\'{exp_name}\', ")

if __name__ == "__main__":
    main()
