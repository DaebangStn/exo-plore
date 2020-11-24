#!/bin/bash
# scripts/test_repo.sh — Integration test for the full repo
# Usage: bash scripts/test_repo.sh /path/to/micromamba

SKIP_TRAIN=false
SKIP_SAMPLE=false
SKIP_SURROGATE=false

set -o pipefail

PASS=0
FAIL=0
SKIP=0

MMBA="${1:?Usage: bash scripts/test_repo.sh /path/to/micromamba}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

VERBOSE_LOG="$PROJECT_ROOT/test_verbose.log"
SIMPLE_LOG="$PROJECT_ROOT/test_simple.log"
rm -f "$VERBOSE_LOG" "$SIMPLE_LOG"

# ── Helpers ──────────────────────────────────────────

vlog() { echo "$@" >> "$VERBOSE_LOG"; }
slog() { echo "$@" >> "$SIMPLE_LOG"; }

mmba() { $MMBA run -n exo "$@"; }

pass_test() {
    local name="$1"
    ((PASS++))
    echo "PASS"
    vlog ">>> PASS"
    slog "PASS: $name"
}

fail_test() {
    local name="$1"
    local detail="$2"
    ((FAIL++))
    if [ -n "$detail" ]; then
        echo "FAIL ($detail)"
        vlog ">>> FAIL ($detail)"
        slog "FAIL: $name ($detail)"
    else
        echo "FAIL"
        vlog ">>> FAIL"
        slog "FAIL: $name"
    fi
    echo ""
    echo "FATAL: $name failed. See $VERBOSE_LOG"
    slog ""
    slog "Passed: $PASS | Failed: $FAIL | Skipped: $SKIP"
    exit 1
}

run_test() {
    local name="$1"; shift
    echo -n "  $name ... "
    vlog "--- $name ---"
    local output
    output=$(eval "$@" 2>&1)
    local rc=$?
    vlog "$output"
    if [ $rc -eq 0 ]; then
        pass_test "$name"
    else
        fail_test "$name" "exit $rc"
    fi
    vlog ""
}

skip_test() {
    local name="$1"
    local reason="$2"
    ((SKIP++))
    echo "  $name ... SKIP ($reason)"
    vlog "--- $name --- SKIP ($reason)"
    vlog ""
    slog "SKIP: $name ($reason)"
}

banner() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
    vlog ""
    vlog "================================================================"
    vlog "  $1"
    vlog "================================================================"
    slog ""
    slog "── $1 ──"
}

# ══════════════════════════════════════════════════════
#  STAGE 0: ENVIRONMENT
# ══════════════════════════════════════════════════════
banner "STAGE 0: ENVIRONMENT"

vlog "--- 0a. Check exo env ---"
if $MMBA env list 2>/dev/null | awk '{print $1}' | grep -qx "exo"; then
    vlog "exo env already exists"
    skip_test "0a. Create exo env" "already exists"
else
    echo -n "  0a. Create exo env ... "
    vlog "NOT FOUND, creating..."
    output=$($MMBA env create -f data/res/environment.yml --yes 2>&1)
    rc=$?
    vlog "$output"
    if [ $rc -eq 0 ]; then
        pass_test "0a. Create exo env"
    else
        fail_test "0a. Create exo env" "exit $rc"
        echo "FATAL: Cannot proceed without exo env. See $VERBOSE_LOG"
        exit 1
    fi
fi
vlog ""

# Install tensorboard/tbparse (not in environment.yml to avoid pip resolver conflict with ray)
run_test "0a2. Install tensorboard" "mmba pip install tensorboard==2.14.0"
run_test "0a3. Install tbparse" "mmba pip install tbparse==0.0.9"

run_test "0b. Verify exo env (python --version)" "mmba python --version"

# Create missing .so symlinks for cmake
CONDA_PREFIX=$($MMBA run -n exo printenv CONDA_PREFIX 2>/dev/null)
vlog "--- 0c. Create .so symlinks ---"
vlog "CONDA_PREFIX: $CONDA_PREFIX"
for lib in libGL libnvToolsExt; do
    target="$CONDA_PREFIX/lib/${lib}.so"
    if [ ! -e "$target" ]; then
        src=$(ls "$CONDA_PREFIX/lib/${lib}.so."* 2>/dev/null | head -1)
        if [ -n "$src" ]; then
            ln -s "$src" "$target"
            vlog "Created symlink: $target -> $src"
        fi
    fi
done
slog "PASS: 0c. Create .so symlinks"
((PASS++))
echo "  0c. Create .so symlinks ... PASS"
vlog ""

# ══════════════════════════════════════════════════════
#  STAGE 1: BUILD
# ══════════════════════════════════════════════════════
banner "STAGE 1: BUILD"

vlog "--- 1a. DART physics engine ---"
if [ -d "$PROJECT_ROOT/libs/dart" ]; then
    vlog "DART already installed in libs/dart"
    skip_test "1a. DART physics engine" "already installed"
else
    run_test "1a. DART physics engine" "mmba bash scripts/dart_install.sh"
fi

echo -n "  1b. C++ GUI build ... "
vlog "--- 1b. C++ GUI build ---"
output=$(mmba cmake --preset gui 2>&1 && mmba ninja -C build/gui -j 16 2>&1)
rc=$?
vlog "$output"
if [ $rc -eq 0 ] && [ -f build/gui/src/render/render ]; then
    pass_test "1b. C++ GUI build"
else
    fail_test "1b. C++ GUI build" "cmake/ninja error or no executable"
fi
vlog ""

echo -n "  1c. C++ headless build ... "
vlog "--- 1c. C++ headless build ---"
output=$(mmba cmake --preset headless 2>&1 && mmba ninja -C build/headless -j 16 2>&1)
rc=$?
vlog "$output"
if [ $rc -eq 0 ]; then
    pass_test "1c. C++ headless build"
else
    fail_test "1c. C++ headless build"
fi
vlog ""

# ══════════════════════════════════════════════════════
#  STAGE 2: PIPELINE
# ══════════════════════════════════════════════════════
banner "STAGE 2: PIPELINE"

if [ "$SKIP_TRAIN" = true ]; then
    skip_test "2.1. RL train" "SKIP_TRAIN=true"
else
    echo -n "  2.1. RL train (2 epochs) ... "
    vlog "--- 2.1. RL train (2 epochs) ---"
    mmba ray start --head --num-cpus=16 >> "$VERBOSE_LOG" 2>&1
    output=$(mmba python -m python.gait_generator.train -m data/gait_generator/no_exo_meta_ma15.yaml -n test_rl -e 2 2>&1)
    rc=$?
    vlog "$output"
    if [ $rc -eq 0 ] && [ -d ray_results/test_rl ]; then
        pass_test "2.1. RL train"
    else
        fail_test "2.1. RL train"
    fi
    mmba ray stop >> "$VERBOSE_LOG" 2>&1
    vlog ""
fi

TEST_CKPT="experiment_data/checkpoints/exo_hei_assist_0809_211111"
TEST_EPOCH=24000

echo -n "  2.2a. Convert checkpoint to TorchScript ... "
vlog "--- 2.2a. Convert checkpoint to TorchScript ---"
output=$(mmba python -m python.cvt_ckpt_to_ts -c "$TEST_CKPT" -o ray_results/ts -e "$TEST_EPOCH" 2>&1)
rc=$?
vlog "$output"
pt_count=$(find ray_results/ts -name "*.pt" 2>/dev/null | wc -l)
if [ $rc -eq 0 ] && [ "$pt_count" -gt 0 ]; then
    pass_test "2.2a. cvt_ckpt_to_ts ($pt_count .pt files)"
else
    fail_test "2.2a. cvt_ckpt_to_ts"
fi
vlog ""

run_test "2.2b. Latin hypercube sampling" "mmba python -m python.sample.param.latin_hypercube_sampled"

if [ "$SKIP_SAMPLE" = true ]; then
    skip_test "2.2c. Rollout" "SKIP_SAMPLE=true"
else
    echo -n "  2.2c. Rollout (export_ckpt) ... "
    vlog "--- 2.2c. Rollout (export_ckpt) ---"
    mmba ray start --head --num-cpus=16 >> "$VERBOSE_LOG" 2>&1
    output=$(mmba python -m python.sample.export_ckpt -c "$TEST_CKPT" -i experiment_data/params/debug.csv -e "$TEST_EPOCH" 2>&1)
    rc=$?
    vlog "$output"
    if [ $rc -eq 0 ]; then
        pass_test "2.2c. Rollout"
    else
        fail_test "2.2c. Rollout"
    fi
    mmba ray stop >> "$VERBOSE_LOG" 2>&1
    vlog ""
fi

if [ "$SKIP_SURROGATE" = true ]; then
    skip_test "2.3. Surrogate train_batch" "SKIP_SURROGATE=true"
else
    run_test "2.3. Surrogate train_batch" "mmba python -m python.surrogate.nn.train_batch"
fi

# ══════════════════════════════════════════════════════
#  STAGE 3: PLOTS
# ══════════════════════════════════════════════════════
banner "STAGE 3: PLOTS"

PLOT_SCRIPTS=(figure2 figure3 figure4_left figure4_right figure5 figure6 figure7 figure8_left figure8_right figure9 supp_b2 supp_c3_left supp_c3_right supp_d4 supp_e5 supp_f6 supp_k7 supp_l8 supp_m9 supp_o10 supp_p11 supp_p12)

for name in "${PLOT_SCRIPTS[@]}"; do
    # Skip if PNG already exists
    if [ -s "plot/${name}.png" ]; then
        skip_test "plot.$name" "PNG exists"
        continue
    fi
    echo -n "  plot.$name ... "
    vlog "--- plot.$name ---"
    output=$(mmba python -m "python.plot.$name" 2>&1)
    rc=$?
    vlog "$output"
    if [ $rc -eq 0 ] && [ -s "plot/${name}.png" ]; then
        pass_test "plot.$name"
    else
        fail_test "plot.$name"
    fi
    vlog ""
done

# ══════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════
banner "RESULTS"

echo "  Passed:  $PASS"
echo "  Failed:  $FAIL"
echo "  Skipped: $SKIP"
echo ""
echo "  Verbose: $VERBOSE_LOG"
echo "  Simple:  $SIMPLE_LOG"
echo ""

slog ""
slog "Passed: $PASS | Failed: $FAIL | Skipped: $SKIP"

echo "All tests passed."
exit 0
