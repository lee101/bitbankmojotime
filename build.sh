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
if [ "${1:-}" = gpu ]; then   # size-t build (model dimensions are compile-time)
  nice -n 10 "$E/bin/mojo" build -O3 -D D=64 -D NL=2 -D NH=4 --target-accelerator=nvidia:sm_120 src/bmt_gpu.mojo -o build/bmt_gpu_t_sm120
fi
if [ "${1:-}" = fcast ]; then   # bmt2 for tools/fcast24.py; portable baseline (host-native builds emit AMD-only instructions: SIGILL on Intel)
  nice -n 10 "$E/bin/mojo" build -O3 --target-cpu "${TARGET_CPU:-x86-64-v3}" src/bmt2.mojo -o build/bmt2
fi
