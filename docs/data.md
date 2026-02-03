# Data Directory Structure

## `data/` — Configuration & Resources (~105 MB, in repo)

| Directory | Description |
|-----------|-------------|
| `gait_generator/` | RL training metadata YAML configs (10 experiments) |
| `surrogate/` | Surrogate model training configs (regression, GP) |
| `record/` | Rollout configs defining which signals to export |
| `human/` | Skeleton XML definitions |
| `muscle/` | Hill-type muscle model parameters |
| `device/` | Exoskeleton device XMLs (hip GEMS2) |
| `OBJ/` | 3D meshes for OpenGL visualization |
| `figs/` | Pre-generated figure images for plot scripts |
| `res/` | Runtime resources |

## `experiment_data/` — Experiment Data

All pipeline outputs used in the paper (~20 GB). Download separately from [onedrive](https://imolab-my.sharepoint.com/:f:/g/personal/geonholeem_imo_snu_ac_kr/IgBf_s4c-HLHRaNIwgMId1vfAZdPu9mfUIghLRW0Hr4vOTk?e=hsQIvm) and unzip at the project root. Required for figure reproduction and for resuming the pipeline from intermediate steps without re-running training.

| Directory | Description |
|-----------|-------------|
| `checkpoints/` | RL policy checkpoints (Step 1 output) |
| `params/` | Parameter sweep CSVs/parquets |
| `simulation/` | Simulation rollout outputs (Step 2 output) |
| `tfevents/` | TensorBoard training logs |
| `real/` | Real-world reference gait data (GEMS, EMG, Gutenberg) |

### Pipeline data flow

```
data/gait_generator/*.yaml        --> Step 1: RL training configs
experiment_data/checkpoints/      --> Step 2: TorchScript conversion input
experiment_data/params/*.csv      --> Step 2: rollout input parameters
data/record/*.yaml                --> Step 2: rollout recording configs
experiment_data/simulation/       --> Step 3: surrogate training input
data/surrogate/*.yaml             --> Step 3: surrogate training configs
experiment_data/real/             --> figure reproduction
experiment_data/tfevents/         --> figure reproduction
```

## Setup

1. Clone the repository (includes `data/`).
2. Obtain `experiment_data.zip` separately.
3. Unzip at the project root:
   ```bash
   unzip experiment_data.zip -d project-exo-plore/
   ```
