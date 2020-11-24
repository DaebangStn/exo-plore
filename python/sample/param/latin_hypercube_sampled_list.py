from skopt.sampler import Lhs
from skopt.space import Space
import pandas as pd
import numpy as np
from pathlib import Path

#============================Configurations============================
NUM_SAMPLE = 80000
SHOW_PLOT = False

#=============================Parameters==============================
PARAM_RANGE = [
    # (0.7, 1.4), # b20 phase
    # (0.2, 1.4), # b20 stride
    (0.6, 1.1), # p20 phase
    (0.4, 1.1), # p20 stride
    
    # (0.0, 0.6), # da k
    # (0.2, 0.5), # db k
    # (0.2, 0.7), # dc k
    (0.0, 0.7), # df k  
    
    # (0.0, 0.4), # da, db, dc delay
    (0.0, 0.5), # df delay
]
ADDITIONAL_PARAM = [ # List[Tuple[param_name, Dict[key, value]]]
    # ('b21df_equi73', {
    #     'muscle_length_Equinus': 0.73,
    # }),
    # ('b21df_equi75', {
    #     'muscle_length_Equinus': 0.75,
    # }),
    # ('b21df_equi78', {
    #     'muscle_length_Equinus': 0.78,
    # }),
    # ('b21df_equi82', {
    #     'muscle_length_Equinus': 0.82,
    # }),
    # ('b21df_equi85', {
    #     'muscle_length_Equinus': 0.85,
    # }),
    # ('b21df_equi90', {
    #     'muscle_length_Equinus': 0.90,
    # }),
    # ('b21df_equi95', {
    #     'muscle_length_Equinus': 0.95,
    # }),
    # ('b21df_equi00', {
    #     'muscle_length_Equinus': 1.00,
    # }),
    # ('b21df_calc0', {
    #     'muscle_force_Calcaneal': 0.0,
    # }),
    # ('b21df_calc1', {
    #     'muscle_force_Calcaneal': 0.1,
    # }),
    # ('b21df_calc13', {
    #     'muscle_force_Calcaneal': 0.13,
    # }),
    # ('b21df_calc16', {
    #     'muscle_force_Calcaneal': 0.16,
    # }),
    # ('b21df_calc2', {
    #     'muscle_force_Calcaneal': 0.2,
    # }),
    # ('b21df_calc3', {
    #     'muscle_force_Calcaneal': 0.3,
    # }),
    # ('b21df_calc4', {
    #     'muscle_force_Calcaneal': 0.4,
    # }),
    # ('b21df_calc5', {
    #     'muscle_force_Calcaneal': 0.5,
    # }),
    # ('b21df_calc6', {
    #     'muscle_force_Calcaneal': 0.6,
    # }),
    # ('b21df_calc7', {
    #     'muscle_force_Calcaneal': 0.7,
    # }),
    # ('b21df_calc8', {
    #     'muscle_force_Calcaneal': 0.8,
    # }),
    # ('b21df_wad0', {
    #     'muscle_force_Waddling': 0.0,
    # }),
    # ('b21df_wad1', {
    #     'muscle_force_Waddling': 0.1,
    # }),
    # ('b21df_wad2', {
    #     'muscle_force_Waddling': 0.2,
    # }),
    # ('b21df_wad3', {
    #     'muscle_force_Waddling': 0.3,
    # }),
    # ('b21df_wad4', {
    #     'muscle_force_Waddling': 0.4,
    # }),
    # ('b21df_wad5', {
    #     'muscle_force_Waddling': 0.5,
    # }),
    # ('b21df_wad6', {
    #     'muscle_force_Waddling': 0.6,
    # }),
    # ('b21df_wad7', {
    #     'muscle_force_Waddling': 0.7,
    # }),
    # ('b21df_wad8', {
    #     'muscle_force_Waddling': 0.8,
    # }),
    # ('b21df_wad9', {
    #     'muscle_force_Waddling': 0.9,
    # }),
    # ('b21df_foot00', {
    #     'muscle_force_Footdrop': 0.0,
    # }),
    # ('b21df_foot025', {
    #     'muscle_force_Footdrop': 0.025,
    # }),
    # ('b21df_foot05', {
    #     'muscle_force_Footdrop': 0.05,
    # }),
    # ('b21df_foot075', {
    #     'muscle_force_Footdrop': 0.075,
    # }),
    # ('b21df_foot10', {
    #     'muscle_force_Footdrop': 0.10,
    # }),
    # ('b21df_foot15', {
    #     'muscle_force_Footdrop': 0.15,
    # }),
    # ('b21df_foot20', {
    #     'muscle_force_Footdrop': 0.2,
    # }),
    # ('b21df_foot25', {
    #     'muscle_force_Footdrop': 0.25,
    # }),
    # ('b21df_foot30', {
    #     'muscle_force_Footdrop': 0.3,
    # }),
    # ('b21df_foot40', {
    #     'muscle_force_Footdrop': 0.4,
    # }),
    # ('b21df_foot50', {
    #     'muscle_force_Footdrop': 0.5,
    # }),
    # ('b21df_foot60', {
    #     'muscle_force_Footdrop': 0.6,
    # }),
    # ('b21df_hyp55', {
    #     'muscle_length_Hyperlordosis': 0.55,
    # }),
    # ('b21df_hyp57', {
    #     'muscle_length_Hyperlordosis': 0.57,
    # }),
    # ('b21df_hyp58', {
    #     'muscle_length_Hyperlordosis': 0.58,
    # }),
    # ('b21df_hyp60', {
    #     'muscle_length_Hyperlordosis': 0.60,
    # }),
    # ('b21df_hyp625', {
    #     'muscle_length_Hyperlordosis': 0.625,
    # }),
    # ('b21df_hyp65', {
    #     'muscle_length_Hyperlordosis': 0.65,
    # }),
    # ('b21df_hyp675', {
    #     'muscle_length_Hyperlordosis': 0.675,
    # }),
    # ('b21df_hyp70', {
    #     'muscle_length_Hyperlordosis': 0.70,
    # }),
    # ('b21df_hyp75', {
    #     'muscle_length_Hyperlordosis': 0.75,
    # }),
    # ('b21df_hyp80', {
    #     'muscle_length_Hyperlordosis': 0.80,
    # }),
    # ('b21df_hyp85', {
    #     'muscle_length_Hyperlordosis': 0.85,
    # }),
    # ('b21df_hyp90', {
    #     'muscle_length_Hyperlordosis': 0.90,
    # }),
    # ('b21df_hyp95', {
    #     'muscle_length_Hyperlordosis': 0.95,
    # }),
    # ('b21df_hyp00', {
    #     'muscle_length_Hyperlordosis': 1.00,
    # }),
]

#======================================================================

def sample_lhs(param_name, param_value):
    sampling_space = Space(PARAM_RANGE)
    lhs = Lhs(criterion='correlation', iterations=100)
    samples = lhs.generate(sampling_space, NUM_SAMPLE, random_state=42)
    param_idx = np.arange(NUM_SAMPLE)

    # convert to dataframe
    df = pd.DataFrame(samples, columns=['Phase', 'Stride', 'K', 'Delay'])
    df['param_idx'] = param_idx
    df = df[['param_idx', 'Phase', 'Stride', 'K', 'Delay']]

    for param_key, param_value in param_value.items():
        df[param_key] = param_value

    # save to parquet
    parquet_path = Path(f'experiment_data/params/{param_name}_LHS{NUM_SAMPLE}.parquet')
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


def main():
    for param_name, param_value in ADDITIONAL_PARAM:
        print(f"==> Sampling {param_name} with {param_value}")
        sample_lhs(param_name, param_value)

if __name__ == '__main__':
    main()
