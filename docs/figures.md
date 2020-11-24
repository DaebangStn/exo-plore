# Figure Reproduction Guide

All figures from the paper can be reproduced using the scripts in `python/plot/`. Ensure the `exo` conda environment is activated.

## Prerequisites

- All simulation data directories must be present in `experiment_data/simulation/`
- Reference data (GEMS, EMG) must be in `experiment_data/real/`
- Kinematics reference data must be in `data/figs/gait_data/`

## Paper Figures

### Figure 2: Normal Gait Kinematics

```bash
python -m python.plot.figure2
```

**Script**: `python/plot/figure2.py`
**Data**: 11 simulation directories (no_exo_meta_ma15 series with `pf_param_no_mod_state` suffix)
**Reference**: GEMS angle/velocity data from `experiment_data/real/gems/`

### Figure 3: Muscle Activation

```bash
python -m python.plot.figure3
```

**Script**: `python/plot/figure3.py`
**Data**: 11 simulation directories (no_exo_meta_ma15 series with `pf_param_act` suffix)
**Reference**: EMG data from `experiment_data/real/emg/` and `experiment_data/real/gait120/`

### Figure 4: CoT Regression

```bash
python -m python.plot.figure4_left
python -m python.plot.figure4_right
```

**Scripts**: `python/plot/figure4_left.py`, `python/plot/figure4_right.py`
**Data**: 27 simulation directories across three groups:
- 8 x no_exo_meta_ma (baseline muscle model)
- 8 x no_exo_meta_ma3 (3-muscle model)
- 11 x no_exo_meta_ma15 (15-muscle model)

### Figure 5: Exo Kinematics

```bash
python -m python.plot.figure5
```

**Script**: `python/plot/figure5.py`
**Data**: 1 simulation directory (exo_hei_resist with k2_d3 delay)

### Figure 6: Delay Power Moment

```bash
python -m python.plot.figure6
```

**Script**: `python/plot/figure6.py`
**Data**: 12 simulation directories across three groups:
- 4 x exo_hei_resist (delays d1-d4)
- 4 x exo_hei_assist (delays d1-d4)
- 4 x exo_no_hei (delays d1-d4)

### Figure 7: MRR Bar

```bash
python -m python.plot.figure7
```

**Script**: `python/plot/figure7.py`
**Data**: Checkpoint data embedded in the script (minimal raw data needed)

### Figure 8: Exo Optimization

```bash
python -m python.plot.figure8_left
python -m python.plot.figure8_right
```

**Scripts**:
- `python/plot/figure8_left.py`
- `python/plot/figure8_right.py`

**Data**: 2 simulation directories + regression model checkpoints in `experiment_data/simulation/`

### Figure 9: Pathology Trend

```bash
python -m python.plot.figure9
```

**Script**: `python/plot/figure9.py`
**Data**: Uses pre-computed y_data; references pathology gait data from `data/figs/gait_data/`

### Supplementary Figures

```bash
python -m python.plot.supp_b2    # Supp B2
python -m python.plot.supp_c3_left   # Supp C3 (left)
python -m python.plot.supp_c3_right  # Supp C3 (right)
python -m python.plot.supp_d4    # Supp D4
python -m python.plot.supp_e5    # Supp E5
python -m python.plot.supp_f6    # Supp F6: Exo Benefit Supporting
python -m python.plot.supp_k7    # Supp K7
python -m python.plot.supp_l8    # Supp L8
python -m python.plot.supp_m9    # Supp M9
python -m python.plot.supp_o10   # Supp O10
python -m python.plot.supp_p11   # Supp P11
python -m python.plot.supp_p12   # Supp P12
```

## Regression Model Checkpoints

The optimization plots require trained regression models. These checkpoints are stored under simulation directories:

```
experiment_data/simulation/exo_hei_resist_0807_131406_21000+memo__b21df_LHS60000_no_mod_state_ma15+_on_0821_145627/
├── nn_v01/
├── nn_v02/
├── nn_v03/
└── ...
```

## Output

Generated figures are saved to the `plot/` directory at the project root (created automatically).
