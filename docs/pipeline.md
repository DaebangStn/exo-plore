# Pipeline

Activate the `exo` environment before running any commands. See [installation.md](installation.md) for environment setup and building.

## Overview

```
[Step 1]  data/gait_generator/*.yaml ──> train.py (RL/PPO) ──> policy checkpoint
                                                                      │
[Step 2]  params.csv ─────────────────────────┐                       │
                                              v                       v
                              python/sample/export_ckpt.py (Gait Data Generator rollout)
                                              │
                                              v
                                    simulation data (.parquet)
                                              │
[Step 3]                          surrogate/nn/train_batch.py
                                              │
                                              v
                                        NN checkpoint
                                              │
[Step 4]                 python/surrogate/batched_optimizer.py
                                              │
                                              v
                              optimal (k, dt) per walking speed
```

---

## Step 1: Train RL Policy (Gait Data Generator)

```bash
python -m python.gait_generator.train \
    -o medium_pc \
    -m data/gait_generator/exo_hei_assist.yaml \
    -n my_experiment
```

Trains a musculoskeletal gait policy using [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html).

**This step is computationally expensive.** A single training run typically requires 10,000--24,000 epochs (see `train_epoch` in the data/gait_generator YAML). We strongly recommend running on a cluster with multiple CPU workers for environment rollouts. The trainer calls `ray.init(address="auto")`, so a Ray cluster must be running and reachable. Use the `-o` flag to select a config preset matching your hardware (e.g., `medium_pc` for a workstation with 32 workers, `large_n4` for a 4-node cluster with 512 workers).

**Output:** Policy checkpoints saved to `ray_results/`.

## Step 2: Rollout Gait Data Generator

This step converts the trained policy to TorchScript, samples the exoskeleton parameter space, and runs C++ rollouts to collect gait data.

### 2a. Convert Checkpoint to TorchScript

```bash
python -m python.cvt_ckpt_to_ts \
    -c <checkpoint_dir> -o ray_results/ts -e <epoch>
```

Converts a Ray RLlib checkpoint into TorchScript (`.pt`) files that the C++ simulation engine can load.

### 2b. Sample Parameter Space

```bash
python -m python.sample.param.latin_hypercube_sampled
```

Creates a space-filling set of exoskeleton control parameter combinations via Latin Hypercube Sampling. Each sample is a tuple of (Phase, Stride, K, Delay), where K is the gain scaling torque magnitude and Delay introduces temporal lag in the delayed-feedback controller. Other sampling strategies are available under `python/sample/param/` (uniform grid, random, Cartesian product, etc.).

### 2c. Run Rollouts

```bash
python -m python.sample.export_ckpt \
    -c experiment_data/checkpoints/no_exo_meta_ma_0730_161202 \
    -i experiment_data/params/b20_U500_k0.csv \
    -e 10000
```

Combines checkpoint conversion (2a) and distributed rollout. For each parameter combination in the input CSV/Parquet, a Ray worker runs the C++ simulation with the trained policy and records gait metrics including metabolic cost-of-transport (CoT).

**Output:** Simulation data saved to `sampled/`.

## Step 3: Train NN Surrogate

```bash
python -m python.surrogate.nn.train_batch
```

Trains a neural network surrogate that maps exoskeleton control parameters (Phase, Stride, K, Delay) to predicted metabolic cost-of-transport. The surrogate replaces expensive physics simulation with a fast, differentiable forward pass, enabling gradient-based optimization in Step 4.

Training uses PyTorch Lightning with mixed-precision (float16/float32). The model is a feedforward network with Huber loss, gradient regularization, and L1/L2 penalties. Training config is in `data/surrogate/regression_soft2.yaml`.

**Output:** Checkpoints under `experiment_data/simulation/<exp_name>/nn_v*/`.

## Step 4: Optimize Exoskeleton Parameters

```python
from python.surrogate.batched_optimizer import Optimizer

exp_name = "your_experiment_name/nn_v01"  # path to surrogate checkpoint
optimizer = Optimizer(exp_name, epoch=10000)

# Access surrogate model for predictions
input_tensor = torch.zeros(batch_size, len(optimizer.model.input_col))
normalized = optimizer.model.normalize_tensor(input_tensor, optimizer.model.input_col)
with torch.no_grad():
    output = optimizer.model(normalized)
output = optimizer.model.denormalize_tensor(output, optimizer.model.output_col)
```

Finds the optimal exoskeleton control parameters (K, Delay) per walking speed by running gradient-based optimization through the trained surrogate network. The optimizer backpropagates through the frozen surrogate to minimize predicted metabolic cost, with support for constrained optimization and input regularization for smooth parameter trajectories across walking speeds. See `python/plot/figure8_left.py` for a complete usage example.

**Output:** Optimal (K, Delay) per walking speed.
