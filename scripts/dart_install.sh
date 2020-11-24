#!/usr/bin/env bash
error_exit() {
    echo "$1" >&2
    exit 1
}

DARTSIM_VERSION=v6.13.2

if ! command -v python &> /dev/null; then
    error_exit "python command not found. Please activate conda env"
fi

# Get Python version (e.g., Python 3.8.10)
python_version=$(python3 --version 2>&1)

# The version must be 3.8
if [[ $python_version != "Python 3.8"* ]]; then
    error_exit "Unsupported Python version: $python_version. Supported version is 3.8."
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENVDIR=${ENVDIR:-${PROJECT_ROOT}/libs/dart}
SRCDIR=$(mktemp -d)

trap "rm -rf $SRCDIR" EXIT

mkdir -p $ENVDIR
mkdir -p $ENVDIR/include
mkdir -p $ENVDIR/lib
mkdir -p $ENVDIR/lib/cmake

export CC=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++

pushd $SRCDIR
git clone --depth 1 --branch $DARTSIM_VERSION https://github.com/dartsim/dart dart
echo "==== Installing dartsim($DARTSIM_VERSION) at $ENVDIR ===="
mkdir dart/build
pushd dart/build
cmake -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$ENVDIR \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
      -DCMAKE_INSTALL_RPATH="${ENVDIR}/lib;${CONDA_PREFIX}/lib" \
      -DDART_BUILD_DARTPY=false \
      -DBUILD_SHARED_LIBS=true \
      -DDART_BUILD_GUI_OSG=false \
      -DDART_ENABLE_SIMD=true \
      ..
ninja install
