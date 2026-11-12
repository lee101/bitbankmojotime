#!/usr/bin/env bash
# build the Mojo inference binaries into build/. MOJO_ENV = a pixi env with mojo (and max for the GPU path).
set -euo pipefail
cd "$(dirname "$0")"
E=${MOJO_ENV:-/nvme0n1-disk/code/mojo-p2m-toolchain/.pixi/envs/default}
export MODULAR_HOME=$E/share/max
export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda-12.9/bin/ptxas}
mkdir -p build
nice -n 10 "$E/bin/mojo" build -O3 src/bmt.mojo -o build/bmt
if [ "${1:-}" = gpu ]; then
  nice -n 10 "$E/bin/mojo" build -O3 --target-accelerator=nvidia:sm_120 src/bmt_gpu.mojo -o build/bmt_gpu_sm120
fi
