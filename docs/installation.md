# Installation Guide

Tested on Ubuntu 24.04, NVIDIA RTX 3070, CUDA 13.1.

## 0. Clone

```bash
git clone --recursive https://github.com/DaebangStn/exo-plore.git
cd exo-plore
# If submodules are missing: git submodule update --init --recursive
```

## 1. Environment Setup

```bash
# Install micromamba or conda

# Create environment from specification
micromamba env create -f data/res/environment.yml

# Activate environment
micromamba activate exo

# Install tensorboard and tbparse separately
# (including them in environment.yml causes pip resolver conflict with ray)
pip install tensorboard==2.14.0
pip install tbparse==0.0.9

```

### OpenGL Symlink Fix

The conda environment may not expose `libGL.so` directly. Create a symlink so the linker can find it:

```bash
ln -s ${CONDA_PREFIX}/lib/libGL.so.1 ${CONDA_PREFIX}/lib/libGL.so
```

## 2. Download Dataset

Download the experiment data from Zenodo and extract it in the project root:

```bash
# Download from Zenodo (placeholder URL — will be updated after registration)
wget https://zenodo.org/records/XXXXXXX/files/experiment_data.7z

# Extract
7z x experiment_data.7z
```

This provides checkpoints, parameter files, and simulation data required for the pipeline.

## 3. Install DART

Build and install DART v6.13.2 into `libs/dart` using the provided script. The `exo` environment must be active — it provides the compiler toolchain and Python 3.8.

```bash
micromamba activate exo
bash scripts/dart_install.sh
```

The top-level `CMakeLists.txt` adds `libs/dart` to `CMAKE_PREFIX_PATH`, so the build finds it automatically.

## 4. Build C++ Components

### GUI Build (with visualization)

```bash
cmake --preset gui
ninja -C build/gui
```

### Headless Build (server/training gait data generator only)

```bash
cmake --preset headless
ninja -C build/headless
```

### Adjusting GPU Architecture

Edit `CMakePresets.json` to set the correct CUDA architecture for your GPU:

| GPU | Architecture |
|-----|-------------|
| RTX 3090 | 8.6 |
| RTX 4090 | 8.9 |
| A6000 | 8.6 |

## 5. Verify Installation

```bash
# Test C++ build
./build/gui/render/render --help

# Test Python imports
python -c "from python.gait_generator.env import MyEnv; print('Train OK')"
python -c "from python.surrogate.nn.module import Regression; print('Surrogate OK')"
python -c "from python.plot.util import *; print('Plot OK')"
```

## Troubleshooting

### Python import errors
Make sure the `exo` environment is activated and run commands from the project root using `python -m` (e.g., `python -m python.surrogate.nn.train_batch`).
