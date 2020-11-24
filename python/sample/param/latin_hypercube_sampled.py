from skopt.sampler import Lhs
from skopt.space import Space
import pandas as pd
import numpy as np
from pathlib import Path

#============================Configurations============================
NUM_SAMPLE = 100000
PARAM_NAME = 'b21df'

USE_EXO = True
CHANGE_EXO = True
SHOW_PLOT = False

#=============================Parameters==============================
PARAM_RANGE = [
    (0.7, 1.4), # b20 phase
    (0.2, 1.4), # b20 stride
    # (0.6, 1.1), # p20 phase
    # (0.4, 1.1), # p20 stride
    
    # (0.0, 0.6), # da k
    # (0.2, 0.5), # db k
    # (0.2, 0.7), # dc k
    (0.0, 0.7), # df k  
    
    # (0.0, 0.4), # da, db, dc delay
    (0.0, 0.5), # df delay
]
ADDITIONAL_PARAM = {
    # "muscle_force_Calcaneal": 0.0, # 0.2 ~ 1.0
    # "muscle_length_Equinus": 0.75, # 0.75 ~ 1.0
    # "muscle_force_Waddling": 0.4, # 0.3 ~ 1.0
}
FIXED_PARAM = {
    'Phase': 1,
    'Stride': 1,
    'K': 0.26667,
    'Delay': 0.35,
}
#======================================================================

def main():
    sampling_space = Space(PARAM_RANGE)
    lhs = Lhs(criterion='correlation', iterations=200)
    samples = lhs.generate(sampling_space, NUM_SAMPLE, random_state=42)
    param_idx = np.arange(NUM_SAMPLE)

    # convert to dataframe
    df = pd.DataFrame(samples, columns=['Phase', 'Stride', 'K', 'Delay'])
    df['param_idx'] = param_idx
    df = df[['param_idx', 'Phase', 'Stride', 'K', 'Delay']]

    if not USE_EXO:
        df = df[['param_idx', 'Phase', 'Stride']]

    elif not CHANGE_EXO:
        df['K'] = FIXED_PARAM['K']
        df['Delay'] = FIXED_PARAM['Delay']

    for param_name, param_value in ADDITIONAL_PARAM.items():
        df[param_name] = param_value

    # save to csv
    parquet_path = Path(f'experiment_data/params/{PARAM_NAME}_LHS{NUM_SAMPLE}.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"Latin Hypercube sampled parameters saved to {parquet_path}")

    if SHOW_PLOT:
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.scatter(df['Phase'], df['Stride'], c='blue', alpha=0.5)
        plt.xlabel('Phase')
        plt.ylabel('Stride')
        plt.title('Latin Hypercube Sampling')

        def on_key_press(event):
            if event.key == 'q' or event.key == 'escape':
                plt.close()

        plt.connect('key_press_event', on_key_press)
        plt.show()

if __name__ == '__main__':
    main()
